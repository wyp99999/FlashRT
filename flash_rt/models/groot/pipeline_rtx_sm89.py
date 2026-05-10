"""FlashRT — SM89 FP16 GROOT N1.6 inference pipeline.

SM89 (RTX 4060 Ti/4090) does not fully support FP8 E4M3 GEMM in cuBLAS.
This pipeline uses FP16 GEMM throughout, avoiding FP8 quantization.

Architecture is identical to pipeline_rtx.py, but:
- No FP8 quantization of weights
- Uses fp16_nn GEMM instead of fp8_descale_fp16
- No calibration needed (FP16 doesn't require dynamic scaling)

For production use on SM89 hardware where FP8 GEMM is unsupported.
"""

from __future__ import annotations

import ctypes
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

from flash_rt.core.cuda_buffer import CudaBuffer
from flash_rt.core.cuda_graph import CUDAGraph

logger = logging.getLogger(__name__)


# ── GROOT N1.6 fixed model dimensions (same as pipeline_rtx.py) ──

VIS_L = 27
VIS_D = 1152
VIS_H = 4304
VIS_NH = 16
VIS_HD = 72
VIS_PATCH_FLAT = 14 * 14 * 3
VIS_SPV_RAW = 256
VIS_SPV = 64
VIS_MLP1_IN = 4608

QWEN3_L = 16
QWEN3_D = 2048
QWEN3_H = 6144
QWEN3_NHQ = 16
QWEN3_NHKV = 8
QWEN3_HD = 128
QWEN3_QKV_DIM = QWEN3_NHQ * QWEN3_HD + 2 * QWEN3_NHKV * QWEN3_HD

DIT_L = 32
DIT_D = 1536
DIT_H = 6144
DIT_NH = 32
DIT_HD = 48

ACTION_DIM = 128
STATE_DIM = 128
ACTION_HORIZON_MAX = 50
NUM_FLOW_STEPS = 4
DIT_OUTPUT_DIM = 1024

FP16_NP = np.float16
FP32 = np.float32


def _p(buf) -> int:
    """Extract int pointer from a CudaBuffer."""
    return buf.ptr.value


# ══════════════════════════════════════════════════════════════════
#  Phase A — SigLIP2 vision encoder (FP16, no FP8)
# ══════════════════════════════════════════════════════════════════

class GrootSigLIP2FP16:
    """SigLIP2 27-layer vision encoder using FP16 GEMM (no FP8).

    Same architecture as GrootSigLIP2 but:
    - Uses fp16_nn for all GEMM operations
    - No FP8 quantization/dequantization
    """

    def __init__(self, gemm, fvk, attn_backend, weights, num_views: int):
        self.gemm = gemm
        self.fvk = fvk
        self.attn = attn_backend
        self.weights = weights
        self.num_views = int(num_views)
        self.S_raw = self.num_views * VIS_SPV_RAW
        self.S_img = self.num_views * VIS_SPV

        self._cudart = ctypes.CDLL("libcudart.so")
        self._attn_ptrs = attn_backend.get_ptrs()

        self.bufs = self._allocate_buffers()

    def _allocate_buffers(self) -> dict:
        nv = self.num_views
        S = self.S_raw
        B = {}
        B["input_images"] = CudaBuffer.device_empty(nv * 224 * 224 * 3, FP16_NP)
        B["sig_x"] = CudaBuffer.device_empty(S * VIS_D, FP16_NP)
        B["sig_x_norm"] = CudaBuffer.device_empty(S * VIS_D, FP16_NP)
        B["sig_patches"] = CudaBuffer.device_empty(S * VIS_PATCH_FLAT, FP16_NP)
        B["sig_qkv"] = CudaBuffer.device_empty(S * 3 * VIS_D, FP16_NP)
        B["sig_hidden"] = CudaBuffer.device_empty(S * VIS_H, FP16_NP)
        B["sig_postln"] = CudaBuffer.device_empty(S * VIS_D, FP16_NP)
        B["mlp1_in"] = CudaBuffer.device_empty(self.S_img * VIS_MLP1_IN, FP16_NP)
        B["mlp1_ln"] = CudaBuffer.device_empty(self.S_img * VIS_MLP1_IN, FP16_NP)
        B["mlp1_fc1"] = CudaBuffer.device_empty(self.S_img * QWEN3_D, FP16_NP)
        B["vision_features"] = CudaBuffer.device_empty(self.S_img * QWEN3_D, FP16_NP)
        return B

    def forward(self, stream: int = 0) -> None:
        """Run SigLIP2 → post-LN → pixel_unshuffle → mlp1."""
        self._patch_embed(stream)
        self._siglip_layers(stream)
        self._post_ln_unshuffle_mlp1(stream)

    def _patch_embed(self, stream: int) -> None:
        fvk = self.fvk
        B = self.bufs
        W = self.weights
        S = self.S_raw

        fvk.patch_im2col(
            _p(B["input_images"]),
            _p(B["sig_patches"]),
            self.num_views, stream)

        self.gemm.fp16_nn(
            _p(B["sig_patches"]),
            W["vision_patch_embedding_w"],
            _p(B["sig_x"]),
            S, VIS_D, VIS_PATCH_FLAT, stream)

        fvk.patch_embed_bias_pos(
            _p(B["sig_x"]),
            W["vision_patch_embedding_b"],
            W["vision_position_embedding"],
            S, VIS_D, VIS_SPV_RAW, stream)

    def _siglip_layers(self, stream: int) -> None:
        fvk = self.fvk
        gemm = self.gemm
        B = self.bufs
        W = self.weights
        S = self.S_raw
        attn_ptrs = self._attn_ptrs

        for i in range(VIS_L):
            self._siglip_one_layer(i, S, attn_ptrs, stream)

    def _siglip_one_layer(self, i: int, S: int, attn_ptrs: dict, stream: int) -> None:
        fvk = self.fvk
        gemm = self.gemm
        B = self.bufs
        W = self.weights

        # Attention LayerNorm
        fvk.layer_norm_fp16(
            _p(B["sig_x"]),
            W["vision_pre_attn_norm_w"][i], W["vision_pre_attn_norm_b"][i],
            _p(B["sig_x_norm"]),
            S, VIS_D, 1e-6, stream)

        # QKV GEMM (FP16): (S, D) @ (D, 3D) → (S, 3D)
        self.gemm.fp16_nn(
            _p(B["sig_x_norm"]),
            W["vision_attn_qkv_w"][i],
            _p(B["sig_qkv"]),
            S, 3 * VIS_D, VIS_D, stream)

        fvk.add_bias_fp16(
            _p(B["sig_qkv"]), W["vision_attn_qkv_b"][i],
            S, 3 * VIS_D, stream)

        # Split QKV
        qkv_ptr = _p(B["sig_qkv"])
        fvk.gpu_strided_copy_fp16(
            qkv_ptr, attn_ptrs["vis_Q"],
            S, VIS_D, 3 * VIS_D, 0, stream)
        fvk.gpu_strided_copy_fp16(
            qkv_ptr, attn_ptrs["vis_K"],
            S, VIS_D, 3 * VIS_D, VIS_D, stream)
        fvk.gpu_strided_copy_fp16(
            qkv_ptr, attn_ptrs["vis_V"],
            S, VIS_D, 3 * VIS_D, 2 * VIS_D, stream)

        # FlashAttention
        attn_out_ptr = self.attn.run(
            "siglip", i, q_seq=VIS_SPV_RAW, stream=stream)

        # O projection (FP16)
        self.gemm.fp16_nn(
            attn_out_ptr, W["vision_attn_o_w"][i],
            _p(B["sig_x_norm"]),
            S, VIS_D, VIS_D, stream)

        # x += x_norm + attn_o_bias
        fvk.bias_residual_fp16(
            _p(B["sig_x"]), _p(B["sig_x_norm"]),
            W["vision_attn_o_b"][i],
            S, VIS_D, stream)

        # FFN LayerNorm
        fvk.layer_norm_fp16(
            _p(B["sig_x"]),
            W["vision_pre_ffn_norm_w"][i], W["vision_pre_ffn_norm_b"][i],
            _p(B["sig_x_norm"]),
            S, VIS_D, 1e-6, stream)

        # FFN up (FP16): (S, D) → (S, H)
        self.gemm.fp16_nn(
            _p(B["sig_x_norm"]), W["vision_ffn_up_w"][i],
            _p(B["sig_hidden"]),
            S, VIS_H, VIS_D, stream)
        fvk.add_bias_fp16(
            _p(B["sig_hidden"]), W["vision_ffn_up_b"][i],
            S, VIS_H, stream)
        fvk.gelu_inplace_fp16(_p(B["sig_hidden"]), S * VIS_H, stream)

        # FFN down (FP16): (S, H) → (S, D)
        self.gemm.fp16_nn(
            _p(B["sig_hidden"]), W["vision_ffn_down_w"][i],
            _p(B["sig_x_norm"]),
            S, VIS_D, VIS_H, stream)
        fvk.bias_residual_fp16(
            _p(B["sig_x"]), _p(B["sig_x_norm"]),
            W["vision_ffn_down_b"][i],
            S, VIS_D, stream)

    def _post_ln_unshuffle_mlp1(self, stream: int) -> None:
        fvk = self.fvk
        B = self.bufs
        W = self.weights
        S = self.S_raw
        fvk.layer_norm_fp16(
            _p(B["sig_x"]),
            W["vision_post_norm_w"], W["vision_post_norm_b"],
            _p(B["sig_postln"]),
            S, VIS_D, 1e-6, stream)


# ══════════════════════════════════════════════════════════════════
#  Phase B — Qwen3-1.7B encoder backbone (FP16)
# ══════════════════════════════════════════════════════════════════

class GrootQwen3FP16:
    """Qwen3-1.7B 16-layer GQA encoder using FP16 GEMM."""

    def __init__(self, gemm, fvk, attn_backend, weights, encoder_seq_max: int):
        self.gemm = gemm
        self.fvk = fvk
        self.attn = attn_backend
        self.weights = weights
        self.Se_max = int(encoder_seq_max)
        self._attn_ptrs = attn_backend.get_ptrs()
        self._cudart = ctypes.CDLL("libcudart.so")

        self._rope_cos, self._rope_sin = self._build_rope()
        self.Se = self.Se_max

        self.bufs = self._allocate_buffers()

    def set_seq_len(self, Se: int) -> None:
        assert Se <= self.Se_max
        self.Se = int(Se)

    def _build_rope(self) -> tuple:
        theta = 1e6
        max_seq = max(self.Se_max, 1024)
        HD = QWEN3_HD
        freqs = 1.0 / (theta ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
        positions = np.arange(max_seq, dtype=np.float64)
        angles = positions[:, None] * freqs[None, :]
        cos = np.concatenate([np.cos(angles), np.cos(angles)], axis=-1).astype(FP16_NP)
        sin = np.concatenate([np.sin(angles), np.sin(angles)], axis=-1).astype(FP16_NP)
        return CudaBuffer.from_numpy(cos), CudaBuffer.from_numpy(sin)

    def _allocate_buffers(self) -> dict:
        Se = self.Se_max
        D = QWEN3_D
        H = QWEN3_H
        QKV = QWEN3_QKV_DIM
        B = {}
        B["x"] = CudaBuffer.device_empty(Se * D, FP16_NP)
        B["x_norm"] = CudaBuffer.device_empty(Se * D, FP16_NP)
        B["qkv"] = CudaBuffer.device_empty(Se * QKV, FP16_NP)
        B["attn_out_flat"] = CudaBuffer.device_empty(Se * D, FP16_NP)
        B["o_out"] = CudaBuffer.device_empty(Se * D, FP16_NP)
        B["gate_up"] = CudaBuffer.device_empty(Se * 2 * H, FP16_NP)
        B["gate"] = CudaBuffer.device_empty(Se * H, FP16_NP)
        B["up"] = CudaBuffer.device_empty(Se * H, FP16_NP)
        B["down"] = CudaBuffer.device_empty(Se * D, FP16_NP)
        B["backbone_features"] = CudaBuffer.device_empty(Se * D, FP16_NP)
        return B

    def forward(self, stream: int = 0) -> None:
        Se = self.Se
        D = QWEN3_D
        H = QWEN3_H
        QKV = QWEN3_QKV_DIM
        NHQ, NHKV, HD = QWEN3_NHQ, QWEN3_NHKV, QWEN3_HD

        fvk = self.fvk
        gemm = self.gemm
        B = self.bufs
        W = self.weights
        attn_ptrs = self._attn_ptrs

        cos_ptr = self._rope_cos.ptr.value
        sin_ptr = self._rope_sin.ptr.value

        for i in range(QWEN3_L):
            # input_layernorm (RMSNorm)
            fvk.rms_norm_fp16(
                _p(B["x"]), W["qwen3_ln_attn_w"][i],
                _p(B["x_norm"]),
                Se, D, 1e-6, stream)

            # QKV GEMM (FP16)
            self.gemm.fp16_nn(
                _p(B["x_norm"]), W["qwen3_qkv_w"][i],
                _p(B["qkv"]),
                Se, QKV, D, stream)

            # Split QKV
            qkv_ptr = _p(B["qkv"])
            fvk.gpu_strided_copy_fp16(
                qkv_ptr, attn_ptrs["qwen3_Q"],
                Se, NHQ * HD, QKV, 0, stream)
            fvk.gpu_strided_copy_fp16(
                qkv_ptr, attn_ptrs["qwen3_K"],
                Se, NHKV * HD, QKV, NHQ * HD, stream)
            fvk.gpu_strided_copy_fp16(
                qkv_ptr, attn_ptrs["qwen3_V"],
                Se, NHKV * HD, QKV, NHQ * HD + NHKV * HD, stream)

            # q_norm / k_norm
            fvk.rms_norm_fp16(
                attn_ptrs["qwen3_Q"], W["qwen3_q_norm_w"][i],
                attn_ptrs["qwen3_Q"],
                Se * NHQ, HD, 1e-6, stream)
            fvk.rms_norm_fp16(
                attn_ptrs["qwen3_K"], W["qwen3_k_norm_w"][i],
                attn_ptrs["qwen3_K"],
                Se * NHKV, HD, 1e-6, stream)

            # RoPE
            fvk.rope_rotate_half_fp16(
                attn_ptrs["qwen3_Q"], cos_ptr, sin_ptr,
                Se, NHQ, HD, stream)
            fvk.rope_rotate_half_fp16(
                attn_ptrs["qwen3_K"], cos_ptr, sin_ptr,
                Se, NHKV, HD, stream)

            # GQA self-attention
            attn_out_ptr = self.attn.run(
                "qwen3", i, q_seq=Se, stream=stream)

            # O projection (fp16)
            self.gemm.fp16_nn(
                attn_out_ptr, W["qwen3_o_w_fp16"][i],
                _p(B["o_out"]),
                Se, D, D, stream)
            fvk.residual_add_fp16(
                _p(B["x"]), _p(B["o_out"]),
                Se * D, stream)

            # post_attention_layernorm
            fvk.rms_norm_fp16(
                _p(B["x"]), W["qwen3_ln_ffn_w"][i],
                _p(B["x_norm"]),
                Se, D, 1e-6, stream)

            # FFN gate+up (FP16)
            self.gemm.fp16_nn(
                _p(B["x_norm"]), W["qwen3_gate_up_w"][i],
                _p(B["gate_up"]),
                Se, 2 * H, D, stream)

            # Split gate / up
            gu_ptr = _p(B["gate_up"])
            fvk.gpu_strided_copy_fp16(
                gu_ptr, _p(B["gate"]),
                Se, H, 2 * H, 0, stream)
            fvk.gpu_strided_copy_fp16(
                gu_ptr, _p(B["up"]),
                Se, H, 2 * H, H, stream)

            # SiLU(gate) * up - use torch for element-wise operation
            # Create torch tensors from buffer pointers
            gate_ptr = _p(B["gate"])
            up_ptr = _p(B["up"])
            # Use torch tensor wrapper with the pointer
            gate_tensor = torch.empty(Se * H, dtype=torch.float16, device='cuda')
            up_tensor = torch.empty(Se * H, dtype=torch.float16, device='cuda')
            fvk.gpu_copy(gate_tensor.data_ptr(), gate_ptr, Se * H * 2, stream)
            fvk.gpu_copy(up_tensor.data_ptr(), up_ptr, Se * H * 2, stream)
            # Apply SiLU then multiply
            gate_tensor = F.silu(gate_tensor)
            gate_tensor.mul_(up_tensor)
            # Copy result back to gate buffer (will be used by FFN down)
            fvk.gpu_copy(gate_ptr, gate_tensor.data_ptr(), Se * H * 2, stream)

            # FFN down (FP16)
            self.gemm.fp16_nn(
                _p(B["gate"]), W["qwen3_down_w"][i],
                _p(B["down"]),
                Se, D, H, stream)
            fvk.residual_add_fp16(
                _p(B["x"]), _p(B["down"]),
                Se * D, stream)

        # Final RMSNorm + vlln
        fvk.rms_norm_fp16(
            _p(B["x"]), W["qwen3_final_norm_w"],
            _p(B["x_norm"]),
            Se, D, 1e-6, stream)
        fvk.layer_norm_fp16(
            _p(B["x_norm"]),
            W["vlln_w"], W["vlln_b"],
            _p(B["backbone_features"]),
            Se, D, 1e-5, stream)


# ══════════════════════════════════════════════════════════════════
#  Phase C — AlternateVLDiT action head (FP16, no FP8)
# ══════════════════════════════════════════════════════════════════

class GrootDiTFP16:
    """32-layer AlternateVLDiT action head + 4-step flow matching (FP16).

    Same architecture as GrootDiT but:
    - Uses fp16_nn for all GEMM operations
    - No FP8 quantization/dequantization
    - No per-layer activation scales needed

    Per layer:
      - AdaLayerNorm: layer_norm_no_affine then ada_layer_norm
        x_norm = LN(x) * (1 + scale[step,layer]) + shift[step,layer]
      - Self-attn (odd layers) OR cross-attn (even layers)
      - O projection (fp16) + residual
      - LayerNorm (no params) → FFN
      - GELU FFN: ff_up (FP16 + bias + GELU fused) → ff_down (FP16 + bias) + residual

    Final:
      - Output AdaLN: shift first, scale second
      - proj_out_2 (fp16, 1024)
      - action_decoder: Linear → ReLU → Linear → velocity
      - actions += dt * velocity (gpu_euler_step)
    """

    def __init__(self, gemm, fvk, attn_backend, weights,
                 action_horizon: int, encoder_seq: int):
        self.gemm = gemm
        self.fvk = fvk
        self.attn = attn_backend
        self.weights = weights
        self.T = int(action_horizon)
        self.Sa = 1 + self.T              # 1 state + T actions
        self.Se = int(encoder_seq)        # backbone seq len (for cross-attn KV)
        self._attn_ptrs = attn_backend.get_ptrs()
        self._cudart = ctypes.CDLL("libcudart.so")
        # Argtypes for cudaMemcpy2DAsync
        self._cudart.cudaMemcpy2DAsync.argtypes = [
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_void_p, ctypes.c_size_t,
            ctypes.c_size_t, ctypes.c_size_t,
            ctypes.c_int, ctypes.c_void_p,
        ]
        self._cudart.cudaMemcpy2DAsync.restype = ctypes.c_int

        self.bufs = self._allocate_buffers()

    def _allocate_buffers(self) -> dict:
        D = DIT_D
        H = DIT_H
        T = self.T
        Sa = self.Sa
        Se = self.Se
        B = {}

        # Inputs (frontend writes per-call)
        B["actions"] = CudaBuffer.device_zeros(T * ACTION_DIM, FP32)
        B["state_feat"] = CudaBuffer.device_empty(D, FP16_NP)
        B["kv_text"] = CudaBuffer.device_empty(Se * QWEN3_D, FP16_NP)
        B["kv_img"] = CudaBuffer.device_empty(Se * QWEN3_D, FP16_NP)

        # Action encoder scratch
        B["actions_fp16"] = CudaBuffer.device_empty(T * ACTION_DIM, FP16_NP)
        B["a_emb"] = CudaBuffer.device_empty(T * D, FP16_NP)
        B["concat"] = CudaBuffer.device_empty(T * 2 * D, FP16_NP)
        B["enc_h"] = CudaBuffer.device_empty(T * D, FP16_NP)

        # DiT main residual stream
        B["hidden"] = CudaBuffer.device_empty(Sa * D, FP16_NP)
        B["h_norm"] = CudaBuffer.device_empty(Sa * D, FP16_NP)
        B["qkv"] = CudaBuffer.device_empty(Sa * 3 * D, FP16_NP)
        B["o_out"] = CudaBuffer.device_empty(Sa * D, FP16_NP)
        B["ff_h"] = CudaBuffer.device_empty(Sa * H, FP16_NP)
        B["ff_out"] = CudaBuffer.device_empty(Sa * D, FP16_NP)

        # Final output
        B["model_out"] = CudaBuffer.device_empty(Sa * DIT_OUTPUT_DIM, FP16_NP)
        B["dec_h"] = CudaBuffer.device_empty(Sa * 1024, FP16_NP)
        B["velocity"] = CudaBuffer.device_empty(Sa * ACTION_DIM, FP16_NP)
        return B

    def precompute_cross_kv(self, stream: int = 0) -> None:
        """Project backbone KV through 16 cross-attn K + V GEMMs.

        Writes the projected (Se, D) fp16 features into the backend-owned
        dit_cross_K[idx] / dit_cross_V[idx] slots.
        """
        fvk = self.fvk
        gemm = self.gemm
        B = self.bufs
        W = self.weights
        Se = self.Se
        D = DIT_D
        attn_ptrs = self._attn_ptrs

        for block_idx in range(DIT_L // 2):
            l = block_idx * 2     # cross blocks at l = 0, 2, 4, ..., 30
            kv_src_buf = B["kv_text"] if (l % 4 == 0) else B["kv_img"]
            kv_src_ptr = _p(kv_src_buf)
            # K projection: (Se, D_kv=2048) @ (2048, D=1536) → (Se, D)
            cross_k_ptr = attn_ptrs["dit_cross_K"][block_idx]
            gemm.fp16_nn(
                kv_src_ptr, W["dit_k_w_fp16"][l],
                cross_k_ptr,
                Se, D, QWEN3_D, stream)
            fvk.add_bias_fp16(
                cross_k_ptr, W["dit_k_b"][l],
                Se, D, stream)
            # V projection
            cross_v_ptr = attn_ptrs["dit_cross_V"][block_idx]
            gemm.fp16_nn(
                kv_src_ptr, W["dit_v_w_fp16"][l],
                cross_v_ptr,
                Se, D, QWEN3_D, stream)
            fvk.add_bias_fp16(
                cross_v_ptr, W["dit_v_b"][l],
                Se, D, stream)

    def run_steps(self, stream: int = 0) -> None:
        """Run all 4 flow-matching steps in sequence."""
        for step in range(NUM_FLOW_STEPS):
            self._run_step(step, stream)

    def _run_step(self, step: int, stream: int) -> None:
        D = DIT_D
        H = DIT_H
        T = self.T
        Sa = self.Sa
        NH, HD = DIT_NH, DIT_HD
        Se = self.Se
        dt = 1.0 / NUM_FLOW_STEPS

        fvk = self.fvk
        gemm = self.gemm
        B = self.bufs
        W = self.weights
        attn_ptrs = self._attn_ptrs

        # Per-step pre-computed bufs
        ada_scales_base = W["ada_scales"]
        ada_shifts_base = W["ada_shifts"]
        out_scales_base = W["out_scales"]
        out_shifts_base = W["out_shifts"]
        ate_base = W["action_time_embeds"]

        # ── Action encode ──
        fvk.gpu_cast_fp32_to_fp16(
            _p(B["actions"]), _p(B["actions_fp16"]),
            T * ACTION_DIM, stream)

        gemm.fp16_nn(
            _p(B["actions_fp16"]), W["action_enc_w1"],
            _p(B["a_emb"]),
            T, D, ACTION_DIM, stream)
        fvk.add_bias_fp16(
            _p(B["a_emb"]), W["action_enc_b1"],
            T, D, stream)

        # Concat [a_emb (T,D), time_emb (T,D)] → (T, 2D)
        ate_ptr_step = ate_base + step * T * D * 2
        row_bytes = D * 2
        dst_pitch = 2 * D * 2
        stream_h = ctypes.c_void_p(stream)
        concat_ptr = _p(B["concat"])
        self._cudart.cudaMemcpy2DAsync(
            concat_ptr, dst_pitch,
            _p(B["a_emb"]), row_bytes,
            row_bytes, T,
            3, stream_h)
        self._cudart.cudaMemcpy2DAsync(
            concat_ptr + row_bytes, dst_pitch,
            ate_ptr_step, row_bytes,
            row_bytes, T,
            3, stream_h)

        gemm.fp16_nn(
            _p(B["concat"]), W["action_enc_w2"],
            _p(B["enc_h"]),
            T, D, 2 * D, stream)
        fvk.add_bias_fp16(
            _p(B["enc_h"]), W["action_enc_b2"],
            T, D, stream)
        fvk.silu_inplace_fp16(_p(B["enc_h"]), T * D, stream)

        gemm.fp16_nn(
            _p(B["enc_h"]), W["action_enc_w3"],
            _p(B["a_emb"]),
            T, D, D, stream)
        fvk.add_bias_fp16(
            _p(B["a_emb"]), W["action_enc_b3"],
            T, D, stream)
        fvk.residual_add_fp16(
            _p(B["a_emb"]), W["pos_emb"],
            T * D, stream)

        # ── Build hidden = [state_feat (1,D), a_emb (T,D)] → (Sa, D) ──
        fvk.gpu_copy(
            _p(B["hidden"]), _p(B["state_feat"]),
            D * 2, stream)
        fvk.gpu_copy(
            _p(B["hidden"]) + D * 2,
            _p(B["a_emb"]),
            T * D * 2, stream)

        # ── 32 alternating self/cross blocks ──
        for l in range(DIT_L):
            is_self = (l % 2 == 1)
            ada_scale_ptr = ada_scales_base + (step * DIT_L + l) * D * 2
            ada_shift_ptr = ada_shifts_base + (step * DIT_L + l) * D * 2

            # ── AdaLayerNorm ──
            fvk.ada_layer_norm_fp16(
                _p(B["hidden"]),
                ada_scale_ptr, ada_shift_ptr,
                _p(B["h_norm"]),
                Sa, D, 1e-5, stream)

            if is_self:
                # ── Self-attention (FP16) ──
                gemm.fp16_nn(
                    _p(B["h_norm"]), W["dit_qkv_w_fp16"][l],
                    _p(B["qkv"]),
                    Sa, 3 * D, D, stream)
                fvk.add_bias_fp16(
                    _p(B["qkv"]), W["dit_qkv_b_self"][l],
                    Sa, 3 * D, stream)

                qkv_ptr = _p(B["qkv"])
                fvk.gpu_strided_copy_fp16(
                    qkv_ptr, attn_ptrs["dit_self_Q"],
                    Sa, D, 3 * D, 0, stream)
                fvk.gpu_strided_copy_fp16(
                    qkv_ptr, attn_ptrs["dit_self_K"],
                    Sa, D, 3 * D, D, stream)
                fvk.gpu_strided_copy_fp16(
                    qkv_ptr, attn_ptrs["dit_self_V"],
                    Sa, D, 3 * D, 2 * D, stream)

                attn_out_ptr = self.attn.run(
                    "dit_self", l, q_seq=Sa, stream=stream)
            else:
                # ── Cross-attention (FP16) ──
                gemm.fp16_nn(
                    _p(B["h_norm"]), W["dit_q_w_fp16"][l],
                    attn_ptrs["dit_cross_Q"],
                    Sa, D, D, stream)
                fvk.add_bias_fp16(
                    attn_ptrs["dit_cross_Q"], W["dit_q_b"][l],
                    Sa, D, stream)

                attn_out_ptr = self.attn.run(
                    "dit_cross", l,
                    q_seq=Sa, kv_seq=Se, stream=stream)

            # ── O projection (fp16) + residual ──
            gemm.fp16_nn(
                attn_out_ptr, W["dit_o_w_fp16"][l],
                _p(B["o_out"]),
                Sa, D, D, stream)
            fvk.add_bias_fp16(
                _p(B["o_out"]), W["dit_o_b"][l],
                Sa, D, stream)
            fvk.residual_add_fp16(
                _p(B["hidden"]), _p(B["o_out"]),
                Sa * D, stream)

            # ── FFN (FP16) ──
            fvk.layer_norm_no_affine_fp16(
                _p(B["hidden"]), _p(B["h_norm"]),
                Sa, D, 1e-5, stream)

            # FFN-up (FP16)
            gemm.fp16_nn(
                _p(B["h_norm"]), W["dit_ff_up_w"][l],
                _p(B["ff_h"]),
                Sa, H, D, stream)
            fvk.add_bias_fp16(
                _p(B["ff_h"]), W["dit_ff_up_b"][l],
                Sa, H, stream)
            fvk.gelu_inplace_fp16(_p(B["ff_h"]), Sa * H, stream)

            # FFN-down (FP16)
            gemm.fp16_nn(
                _p(B["ff_h"]), W["dit_ff_down_w"][l],
                _p(B["ff_out"]),
                Sa, D, H, stream)
            fvk.add_bias_fp16(
                _p(B["ff_out"]), W["dit_ff_down_b"][l],
                Sa, D, stream)
            fvk.residual_add_fp16(
                _p(B["hidden"]), _p(B["ff_out"]),
                Sa * D, stream)

        # ── Final AdaLayerNorm ──
        out_scale_ptr = out_scales_base + step * D * 2
        out_shift_ptr = out_shifts_base + step * D * 2
        fvk.ada_layer_norm_fp16(
            _p(B["hidden"]), out_scale_ptr, out_shift_ptr,
            _p(B["h_norm"]),
            Sa, D, 1e-6, stream)

        # ── proj_out_2 (fp16): (Sa, D) → (Sa, output_dim=1024) ──
        gemm.fp16_nn(
            _p(B["h_norm"]), W["proj_out_2_w"],
            _p(B["model_out"]),
            Sa, DIT_OUTPUT_DIM, D, stream)
        fvk.add_bias_fp16(
            _p(B["model_out"]), W["proj_out_2_b"],
            Sa, DIT_OUTPUT_DIM, stream)

        # ── action_decoder: 2-layer MLP ──
        gemm.fp16_nn(
            _p(B["model_out"]), W["action_dec_w1"],
            _p(B["dec_h"]),
            Sa, 1024, DIT_OUTPUT_DIM, stream)
        fvk.add_bias_fp16(
            _p(B["dec_h"]), W["action_dec_b1"],
            Sa, 1024, stream)
        fvk.relu_inplace_fp16(_p(B["dec_h"]), Sa * 1024, stream)
        gemm.fp16_nn(
            _p(B["dec_h"]), W["action_dec_w2"],
            _p(B["velocity"]),
            Sa, ACTION_DIM, 1024, stream)
        fvk.add_bias_fp16(
            _p(B["velocity"]), W["action_dec_b2"],
            Sa, ACTION_DIM, stream)

        # ── actions += dt * velocity (Euler step) ──
        vel_offset = (Sa - self.T) * ACTION_DIM
        fvk.gpu_euler_step(
            _p(B["actions"]), _p(B["velocity"]),
            self.T, ACTION_DIM, dt, vel_offset, stream)