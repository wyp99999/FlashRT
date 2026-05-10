"""FlashRT — SM89 FP16 Pi0.5 inference pipeline.

SM89 (RTX 4060 Ti/4090) has limited FP8 GEMM support in cuBLAS.
This pipeline uses FP16 GEMM throughout, avoiding FP8 quantization.

Key differences from Pi0:
- AdaRMSNorm: style modulation instead of RMSNorm in decoder
- Pre-computed decoder styles (time_emb + style_attn + style_ffn + style_final)
- No state_proj (uses action_in_proj directly)

Architecture:
- Vision: 27 SigLIP layers (head_dim=72)
- Encoder: 18 Gemma-2B layers (GQA 8Q/1KV, head_dim=256)
- Decoder: 18 Gemma-300M layers with AdaRMSNorm (GQA 8Q/1KV, head_dim=256)
- Diffusion: 10-step flow-matching with style conditioning
"""

from __future__ import annotations

import ctypes
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

from flash_rt.core.cuda_buffer import CudaBuffer

logger = logging.getLogger(__name__)


# Fixed Pi0.5 model dimensions (same as Pi0)
VIS_L = 27
VIS_D = 1152
VIS_H = 4304
VIS_NH = 16
VIS_HD = 72
VIS_SEQ_PER_VIEW = 256
VIS_PATCH_FLAT = 14 * 14 * 3  # 588

ENC_L = 18
ENC_D = 2048
ENC_H = 16384
ENC_NH = 8
ENC_NKV = 1
ENC_HD = 256

DEC_L = 18
DEC_D = 1024
DEC_H = 4096
DEC_NH = 8
DEC_NKV = 1
DEC_HD = 256

ACTION_DIM = 32
CHUNK_SIZE_DEFAULT = 10
NUM_STEPS_DEFAULT = 10

FP16 = np.float16
FP32 = np.float32


def _p(buf) -> int:
    """Extract int pointer from a CudaBuffer."""
    return buf.ptr.value


class Pi05PipelineSm89:
    """Pi0.5 inference pipeline for SM89 (FP16, no FP8).
    
    Key differences from Pi0PipelineSm89:
    - Decoder uses AdaRMSNorm (style modulation) instead of RMSNorm
    - Pre-computed style buffers: time_emb, style_attn, style_ffn, style_final
    - Each decoder layer gets style modulation based on diffusion step
    
    Args:
        gemm: GemmRunner for FP16 GEMM operations.
        fvk: flash_rt_kernels module.
        weights: Pointer dict from frontend.
        num_views: Number of camera views.
        max_prompt_len: Maximum prompt token length.
        chunk_size: Action chunk size (default 10).
        num_steps: Diffusion denoise steps (default 10).
    """
    
    def __init__(
        self,
        gemm,
        fvk,
        weights,
        *,
        num_views: int,
        max_prompt_len: int,
        chunk_size: int = CHUNK_SIZE_DEFAULT,
        num_steps: int = NUM_STEPS_DEFAULT,
    ):
        self.gemm = gemm
        self.fvk = fvk
        self.weights = weights
        self._ctx = fvk.FvkContext()
        
        self.num_views = int(num_views)
        self.max_prompt_len = int(max_prompt_len)
        self.chunk_size = int(chunk_size)
        self.S_dec = self.chunk_size + 1
        self.num_steps = int(num_steps)
        
        # Fixed layer configuration (18 encoder + 18 decoder)
        # NOTE: Reducing layers causes severe quality loss in diffusion models
        self.enc_layers = ENC_L
        self.dec_layers = DEC_L
        
        # Derived sizes
        self.vision_seq = self.num_views * VIS_SEQ_PER_VIEW
        self.encoder_seq_len = self.vision_seq + self.max_prompt_len
        
        # Allocate buffers
        self.bufs = self._allocate_buffers()
        
        # Build RoPE table
        self._build_rope_table()
        
        # Upload precomputed styles from frontend
        self._upload_precomputed_styles()
        
        # CUDART for D2D copies
        self._cudart = ctypes.CDLL("libcudart.so")
        
        # Pre-create torch tensors for CUDA Graph compatibility
        # (No dynamic allocation during inference)
        self._create_preallocated_tensors()
        
        logger.info(
            "Pi05PipelineSm89 initialised (num_views=%d, vision_seq=%d, "
            "encoder_seq_len=%d, chunk_size=%d, num_steps=%d)",
            self.num_views, self.vision_seq, self.encoder_seq_len,
            self.chunk_size, self.num_steps)
    
    def _create_preallocated_tensors(self):
        """Create pre-allocated torch tensors for CUDA Graph compatibility.
        
        These tensors are created once at initialization and reused during inference,
        avoiding dynamic allocation that would prevent CUDA Graph capture.
        """
        sd = self.S_dec
        es = self.encoder_seq_len
        dec_qkv_dim = (DEC_NH + 2 * DEC_NKV) * DEC_HD
        
        # Decoder scratch tensors (pre-created, reused per-layer per-step)
        self._dec_qkv_tensor = torch.empty(sd, dec_qkv_dim, dtype=torch.float16, device='cuda')
        self._dec_rope_tensor = torch.empty(sd, 256, dtype=torch.float16, device='cuda')
        self._dec_enc_K_tensor = torch.empty(es, ENC_NKV, ENC_HD, dtype=torch.float16, device='cuda')
        self._dec_enc_V_tensor = torch.empty(es, ENC_NKV, ENC_HD, dtype=torch.float16, device='cuda')
        
        # Store views for common patterns
        self._dec_rope_cos = self._dec_rope_tensor[:, :128]
        self._dec_rope_sin = self._dec_rope_tensor[:, 128:]
        
        logger.info("Created pre-allocated decoder tensors for CUDA Graph")
    
    def _allocate_buffers(self) -> dict:
        """Allocate all pipeline working buffers."""
        nv = self.num_views
        vs = self.vision_seq
        es = self.encoder_seq_len
        sd = self.S_dec
        sa = self.chunk_size
        B = {}
        
        # Vision (SigLIP) - same as Pi0
        B["observation_images_normalized"] = CudaBuffer.device_empty(
            nv * 224 * 224 * 3, FP16)
        B["vision_x"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        B["vision_x_norm"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        B["vision_QKV"] = CudaBuffer.device_empty(vs * 3 * VIS_D, FP16)
        B["vision_hidden"] = CudaBuffer.device_empty(vs * VIS_H, FP16)
        B["vision_patches"] = CudaBuffer.device_empty(vs * VIS_PATCH_FLAT, FP16)
        B["vision_pos_embed_expanded"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        B["vision_attn_out"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        B["vision_ffn_out"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        
        # Encoder (Gemma-2B) - same as Pi0
        B["encoder_x"] = CudaBuffer.device_empty(es * ENC_D, FP16)
        B["encoder_x_norm"] = CudaBuffer.device_empty(es * ENC_D, FP16)
        B["encoder_QKV"] = CudaBuffer.device_empty(
            es * (ENC_NH + 2 * ENC_NKV) * ENC_HD, FP16)
        B["encoder_hidden"] = CudaBuffer.device_empty(es * ENC_H, FP16)
        B["encoder_gate_merged"] = CudaBuffer.device_empty(es * 2 * ENC_H, FP16)
        B["encoder_K_cache"] = CudaBuffer.device_empty(
            ENC_L * es * ENC_NKV * ENC_HD, FP16)
        B["encoder_V_cache"] = CudaBuffer.device_empty(
            ENC_L * es * ENC_NKV * ENC_HD, FP16)
        B["encoder_attn_out"] = CudaBuffer.device_empty(es * ENC_D, FP16)
        B["encoder_ffn_out"] = CudaBuffer.device_empty(es * ENC_D, FP16)
        
        # RMSNorm ones buffers
        B["encoder_ones"] = CudaBuffer.from_numpy(np.ones(ENC_D, dtype=FP16))
        B["decoder_ones"] = CudaBuffer.from_numpy(np.ones(DEC_D, dtype=FP16))
        
        # Decoder (Gemma-300M) - with AdaRMSNorm buffers
        B["decoder_x"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        B["decoder_x_norm"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        B["decoder_QKV"] = CudaBuffer.device_empty(
            sd * (DEC_NH + 2 * DEC_NKV) * DEC_HD, FP16)
        B["decoder_hidden"] = CudaBuffer.device_empty(sd * DEC_H, FP16)
        B["decoder_gate_merged"] = CudaBuffer.device_empty(sd * 2 * DEC_H, FP16)
        B["decoder_attn_out"] = CudaBuffer.device_empty(sd * DEC_NH * DEC_HD, FP16)
        
        # AdaRMSNorm scratch buffers
        B["x_normed_buf"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        B["gate_buf"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        
        # Precomputed style buffers (uploaded from frontend)
        B["decoder_time_emb"] = CudaBuffer.device_empty(
            self.num_steps * sd * DEC_D, FP16)
        B["decoder_style_attn"] = CudaBuffer.device_empty(
            self.num_steps * DEC_L * sd * 3 * DEC_D, FP16)
        B["decoder_style_ffn"] = CudaBuffer.device_empty(
            self.num_steps * DEC_L * sd * 3 * DEC_D, FP16)
        B["decoder_style_final"] = CudaBuffer.device_empty(
            self.num_steps * sd * 3 * DEC_D, FP16)
        
        # Diffusion
        B["diffusion_noise"] = CudaBuffer.device_empty(sa * ACTION_DIM, FP16)
        B["decoder_action_buf"] = CudaBuffer.device_empty(sa * ACTION_DIM, FP16)
        
        return B
    
    def _upload_precomputed_styles(self):
        """Upload frontend-precomputed decoder style buffers."""
        pre = self.weights.get("precomputed", {})
        if not pre:
            logger.warning("No precomputed styles provided, using dummy values")
            return
        
        B = self.bufs
        
        if "time_emb" in pre:
            B["decoder_time_emb"].upload(np.ascontiguousarray(pre["time_emb"]))
        
        if "style_attn" in pre:
            B["decoder_style_attn"].upload(np.ascontiguousarray(pre["style_attn"]))
        
        if "style_ffn" in pre:
            B["decoder_style_ffn"].upload(np.ascontiguousarray(pre["style_ffn"]))
        
        if "style_final" in pre:
            B["decoder_style_final"].upload(np.ascontiguousarray(pre["style_final"]))
        
        logger.info("Uploaded precomputed styles for %d diffusion steps", self.num_steps)
    
    def _style_slice_ptr(self, buf_name: str, step: int, layer: int | None = None) -> int:
        """Compute device pointer for a per-step (per-layer) style slice."""
        base = _p(self.bufs[buf_name])
        sd = self.S_dec
        
        if buf_name == "decoder_time_emb":
            return base + step * sd * DEC_D * 2
        if buf_name == "decoder_style_final":
            return base + step * sd * 3 * DEC_D * 2
        
        # style_attn / style_ffn: (num_steps, DEC_L, chunk_size, 3*DEC_D)
        per_layer = sd * 3 * DEC_D * 2
        per_step = DEC_L * per_layer
        return base + step * per_step + layer * per_layer
    
    def _build_rope_table(self):
        """Build RoPE cos/sin tables."""
        max_pos = self.encoder_seq_len + self.S_dec
        
        inv_freq = 1.0 / (10000 ** (np.arange(0, 256, 2, dtype=np.float64) / 256))
        positions = np.arange(max_pos, dtype=np.float64)
        phase = positions[:, None] * inv_freq[None, :]
        
        cos = np.cos(phase).astype(FP16)
        sin = np.sin(phase).astype(FP16)
        
        interleaved = np.stack([cos, sin], axis=-1).reshape(max_pos, 256)
        self._rope_table_np = interleaved
        
        enc_rope = interleaved[:self.encoder_seq_len]
        self.bufs["encoder_rope"] = CudaBuffer.from_numpy(
            np.ascontiguousarray(enc_rope))
        
        dec_rope = interleaved[
            self.encoder_seq_len:self.encoder_seq_len + self.S_dec]
        self.bufs["decoder_rope"] = CudaBuffer.from_numpy(
            np.ascontiguousarray(dec_rope))
    
    def _build_pos_embed_expanded(self):
        """Expand position embedding across num_views."""
        pos_src_ptr = self.weights["vision_position_embedding"]
        per_view_nbytes = VIS_SEQ_PER_VIEW * VIS_D * 2
        dst_buf = self.bufs["vision_pos_embed_expanded"]
        
        for v in range(self.num_views):
            self._cudart.cudaMemcpy(
                ctypes.c_void_p(dst_buf.ptr.value + v * per_view_nbytes),
                ctypes.c_void_p(pos_src_ptr),
                per_view_nbytes, 3)
        self._cudart.cudaDeviceSynchronize()
    
    # ══════════════════════════════════════════════════════════════════
    #  AdaRMSNorm (optimized kernel implementation)
    # ══════════════════════════════════════════════════════════════════
    
    def _ada_rms_norm_fp16(self, x_ptr: int, style_ptr: int, 
                           out_ptr: int, gate_out_ptr: int,
                           seq: int, dim: int, stream: int):
        """AdaRMSNorm using optimized fvk kernel.
        
        AdaRMSNorm: y = (1 + gate) * rms_norm(x) + shift
        style layout: [gate (dim), shift (dim), scale (dim)]
        
        Uses fvk.adarms_fp16 kernel for zero GPU-CPU copy overhead.
        """
        self.fvk.adarms_fp16(x_ptr, style_ptr, out_ptr, gate_out_ptr, seq, dim, stream)
    
    def _gate_residual_fp16(self, x_ptr: int, residual_ptr: int, 
                             gate_ptr: int, seq_dim: int, stream: int):
        """Gate * residual add: x += gate * residual using fvk kernel."""
        # gate_mul_residual(residual, x, gate, n, stream)
        # computes: residual += gate * x (but we want x += gate * residual)
        # So we use: gate_mul_residual(x_ptr, residual_ptr, gate_ptr, seq_dim, stream)
        # which computes: x += gate * residual
        self.fvk.gate_mul_residual(x_ptr, residual_ptr, gate_ptr, seq_dim, stream)
    
    # ══════════════════════════════════════════════════════════════════
    #  Phase A: Vision Encoder (SigLIP 27 layers)
    # ══════════════════════════════════════════════════════════════════
    
    def vision_encoder(self, stream: int = 0):
        """Run SigLIP vision encoder - identical to Pi0."""
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        seq = self.vision_seq
        nv = self.num_views
        
        # Patch embedding
        fvk.patch_im2col(
            _p(B["observation_images_normalized"]),
            _p(B["vision_patches"]),
            nv, stream)
        
        gemm.fp16_nn(
            _p(B["vision_patches"]),
            W["vision_patch_embedding_w"],
            _p(B["vision_x"]),
            seq, VIS_D, VIS_PATCH_FLAT, stream=stream)
        
        fvk.bias_residual_fp16(
            _p(B["vision_x"]),
            _p(B["vision_pos_embed_expanded"]),
            W["vision_patch_embedding_b"],
            seq, VIS_D, stream=stream)
        
        for i in range(VIS_L):
            self._vision_layer(i, stream)
    
    def _vision_layer(self, i: int, stream: int):
        """One SigLIP layer - identical to Pi0."""
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        seq = self.vision_seq
        
        # Pre-attention LayerNorm
        fvk.layer_norm_fp16(
            _p(B["vision_x"]),
            W["vision_pre_attn_norm_w"][i],
            W["vision_pre_attn_norm_b"][i],
            _p(B["vision_x_norm"]),
            seq, VIS_D, 1e-5, stream=stream)
        
        # QKV GEMM
        gemm.fp16_nn(
            _p(B["vision_x_norm"]),
            W["vision_attn_qkv_w"][i],
            _p(B["vision_QKV"]),
            seq, 3 * VIS_D, VIS_D, stream=stream)
        fvk.add_bias_fp16(
            _p(B["vision_QKV"]),
            W["vision_attn_qkv_b"][i],
            seq, 3 * VIS_D, stream=stream)
        
        # Split QKV and run torch attention
        qkv_tensor = torch.empty(seq, 3 * VIS_D, dtype=torch.float16, device='cuda')
        fvk.gpu_copy(qkv_tensor.data_ptr(), _p(B["vision_QKV"]), seq * 3 * VIS_D * 2, stream)
        
        Q = qkv_tensor[:, :VIS_D].view(seq, VIS_NH, VIS_HD)
        K = qkv_tensor[:, VIS_D:2*VIS_D].view(seq, VIS_NH, VIS_HD)
        V = qkv_tensor[:, 2*VIS_D:].view(seq, VIS_NH, VIS_HD)
        
        attn_out = torch.empty(seq, VIS_D, dtype=torch.float16, device='cuda')
        for v in range(self.num_views):
            start = v * VIS_SEQ_PER_VIEW
            end = (v + 1) * VIS_SEQ_PER_VIEW
            Q_view = Q[start:end].transpose(0, 1)
            K_view = K[start:end].transpose(0, 1)
            V_view = V[start:end].transpose(0, 1)
            
            attn_result = F.scaled_dot_product_attention(Q_view, K_view, V_view)
            attn_out[start:end] = attn_result.transpose(0, 1).reshape(
                VIS_SEQ_PER_VIEW, VIS_D)
        
        fvk.gpu_copy(_p(B["vision_attn_out"]), attn_out.data_ptr(), seq * VIS_D * 2, stream)
        
        # O projection
        gemm.fp16_nn(
            _p(B["vision_attn_out"]),
            W["vision_attn_o_w"][i],
            _p(B["vision_x_norm"]),
            seq, VIS_D, VIS_D, stream=stream)
        fvk.bias_residual_fp16(
            _p(B["vision_x"]),
            _p(B["vision_x_norm"]),
            W["vision_attn_o_b"][i],
            seq, VIS_D, stream=stream)
        
        # Pre-FFN LayerNorm
        fvk.layer_norm_fp16(
            _p(B["vision_x"]),
            W["vision_pre_ffn_norm_w"][i],
            W["vision_pre_ffn_norm_b"][i],
            _p(B["vision_x_norm"]),
            seq, VIS_D, 1e-5, stream=stream)
        
        # FFN up
        gemm.fp16_nn(
            _p(B["vision_x_norm"]),
            W["vision_ffn_up_w"][i],
            _p(B["vision_hidden"]),
            seq, VIS_H, VIS_D, stream=stream)
        fvk.add_bias_fp16(
            _p(B["vision_hidden"]),
            W["vision_ffn_up_b"][i],
            seq, VIS_H, stream=stream)
        fvk.gelu_inplace_fp16(_p(B["vision_hidden"]), seq * VIS_H, stream=stream)
        
        # FFN down
        gemm.fp16_nn(
            _p(B["vision_hidden"]),
            W["vision_ffn_down_w"][i],
            _p(B["vision_ffn_out"]),
            seq, VIS_D, VIS_H, stream=stream)
        fvk.add_bias_fp16(
            _p(B["vision_ffn_out"]),
            W["vision_ffn_down_b"][i],
            seq, VIS_D, stream=stream)
        
        fvk.residual_add_fp16(_p(B["vision_x"]), _p(B["vision_ffn_out"]), seq * VIS_D, stream)
    
    # ══════════════════════════════════════════════════════════════════
    #  Phase B: Gemma-2B Encoder (18 layers)
    # ══════════════════════════════════════════════════════════════════
    
    def transformer_encoder(self, stream: int = 0):
        """Run Gemma-2B encoder - identical to Pi0."""
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        vs = self.vision_seq
        es = self.encoder_seq_len
        
        # Vision final norm
        fvk.layer_norm_fp16(
            _p(B["vision_x"]),
            W["vision_final_norm_w"],
            W["vision_final_norm_b"],
            _p(B["vision_x_norm"]),
            vs, VIS_D, 1e-5, stream=stream)
        
        # Multi-modal projector
        gemm.fp16_nn(
            _p(B["vision_x_norm"]),
            W["encoder_multi_modal_projector_w"],
            _p(B["encoder_x"]),
            vs, ENC_D, VIS_D, stream=stream)
        fvk.add_bias_fp16(
            _p(B["encoder_x"]),
            W["encoder_multi_modal_projector_b"],
            vs, ENC_D, stream=stream)
        
        for i in range(self.enc_layers):
            self._encoder_layer(i, es, stream)
    
    def _encoder_layer(self, i: int, seq: int, stream: int):
        """One encoder layer - identical to Pi0."""
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        
        # RMSNorm
        fvk.rms_norm_fp16(
            _p(B["encoder_x"]),
            _p(B["encoder_ones"]),
            _p(B["encoder_x_norm"]),
            seq, ENC_D, 1e-6, stream=stream)
        
        # QKV GEMM
        gemm.fp16_nn(
            _p(B["encoder_x_norm"]),
            W["encoder_attn_qkv_w"][i],
            _p(B["encoder_QKV"]),
            seq, (ENC_NH + 2 * ENC_NKV) * ENC_HD, ENC_D, stream=stream)
        
        # Split QKV
        qkv_tensor = torch.empty(seq, (ENC_NH + 2 * ENC_NKV) * ENC_HD, 
                                 dtype=torch.float16, device='cuda')
        fvk.gpu_copy(qkv_tensor.data_ptr(), _p(B["encoder_QKV"]), 
                     seq * (ENC_NH + 2 * ENC_NKV) * ENC_HD * 2, stream)
        
        Q = qkv_tensor[:, :ENC_NH * ENC_HD].view(seq, ENC_NH, ENC_HD)
        KV = qkv_tensor[:, ENC_NH * ENC_HD:].view(seq, 2, ENC_NKV, ENC_HD)
        K = KV[:, 0, :, :]
        V = KV[:, 1, :, :]
        
        # Apply RoPE
        rope_tensor = torch.empty(seq, 256, dtype=torch.float16, device='cuda')
        fvk.gpu_copy(rope_tensor.data_ptr(), _p(B["encoder_rope"]), seq * 256 * 2, stream)
        cos = rope_tensor[:, :128]
        sin = rope_tensor[:, 128:]
        
        Q_rope = self._apply_rope_interleaved(Q, cos, sin)
        K_rope = self._apply_rope_interleaved(K, cos, sin)
        
        # Write K/V to cache
        K_cache_offset = i * seq * ENC_NKV * ENC_HD * 2
        V_cache_offset = i * seq * ENC_NKV * ENC_HD * 2
        
        fvk.gpu_copy(_p(B["encoder_K_cache"]) + K_cache_offset, 
                     K_rope.data_ptr(), seq * ENC_NKV * ENC_HD * 2, stream)
        fvk.gpu_copy(_p(B["encoder_V_cache"]) + V_cache_offset,
                     V.data_ptr(), seq * ENC_NKV * ENC_HD * 2, stream)
        
        # Torch attention (GQA)
        K_expanded = K_rope.expand(-1, ENC_NH, -1)
        V_expanded = V.expand(-1, ENC_NH, -1)
        
        Q_t = Q_rope.transpose(0, 1)
        K_t = K_expanded.transpose(0, 1)
        V_t = V_expanded.transpose(0, 1)
        
        attn_out = F.scaled_dot_product_attention(Q_t, K_t, V_t)
        attn_out = attn_out.transpose(0, 1).reshape(seq, ENC_D)
        
        fvk.gpu_copy(_p(B["encoder_attn_out"]), attn_out.data_ptr(), seq * ENC_D * 2, stream)
        
        # O projection
        gemm.fp16_nn(
            _p(B["encoder_attn_out"]),
            W["encoder_attn_o_w"][i],
            _p(B["encoder_x_norm"]),
            seq, ENC_D, ENC_D, stream=stream)
        
        fvk.residual_add_fp16(_p(B["encoder_x"]), _p(B["encoder_x_norm"]), seq * ENC_D, stream)
        
        # FFN
        fvk.rms_norm_fp16(
            _p(B["encoder_x"]),
            _p(B["encoder_ones"]),
            _p(B["encoder_x_norm"]),
            seq, ENC_D, 1e-6, stream=stream)
        
        gemm.fp16_nn(
            _p(B["encoder_x_norm"]),
            W["encoder_ffn_gate_w"][i],
            _p(B["encoder_gate_merged"]),
            seq, ENC_H, ENC_D, stream=stream)
        gemm.fp16_nn(
            _p(B["encoder_x_norm"]),
            W["encoder_ffn_up_w"][i],
            _p(B["encoder_hidden"]),
            seq, ENC_H, ENC_D, stream=stream)
        
        fvk.gate_geglu_fp16(
            _p(B["encoder_gate_merged"]),
            _p(B["encoder_hidden"]),
            _p(B["encoder_hidden"]),
            seq * ENC_H, stream=stream)
        
        gemm.fp16_nn(
            _p(B["encoder_hidden"]),
            W["encoder_ffn_down_w"][i],
            _p(B["encoder_ffn_out"]),
            seq, ENC_D, ENC_H, stream=stream)
        
        fvk.residual_add_fp16(_p(B["encoder_x"]), _p(B["encoder_ffn_out"]), seq * ENC_D, stream)
    
    def _apply_rope_interleaved(self, x, cos, sin):
        """Apply interleaved RoPE."""
        x_even = x[..., :x.shape[-1]//2]
        x_odd = x[..., x.shape[-1]//2:]
        
        if x.dim() == 3:
            cos_exp = cos.unsqueeze(1)
            sin_exp = sin.unsqueeze(1)
        else:
            cos_exp = cos
            sin_exp = sin
        
        rotated_even = x_even * cos_exp - x_odd * sin_exp
        rotated_odd = x_even * sin_exp + x_odd * cos_exp
        
        return torch.cat([rotated_even, rotated_odd], dim=-1)
    
    # ══════════════════════════════════════════════════════════════════
    #  Phase C: Decoder + Diffusion (AdaRMSNorm style modulation)
    # ══════════════════════════════════════════════════════════════════
    
    def _assemble_decoder_x(self, step: int, stream: int):
        """Build decoder_x = [time_emb[0]; action_in_proj(noise) + time_emb[1:]]."""
        B = self.bufs
        W = self.weights
        fvk = self.fvk
        gemm = self.gemm
        sa = self.chunk_size
        
        # Get time embedding for this step: (chunk_size+1, DEC_D)
        time_emb_ptr = self._style_slice_ptr("decoder_time_emb", step)
        
        # Copy time_emb[0] to decoder_x[0] (first token)
        self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(_p(B["decoder_x"])),
            ctypes.c_void_p(time_emb_ptr),
            DEC_D * 2, 3, stream)
        
        # action_in_proj(noise) → decoder_x[1:S_dec]
        x_action_ptr = _p(B["decoder_x"]) + DEC_D * 2
        
        gemm.fp16_nn(
            _p(B["diffusion_noise"]),
            W["decoder_action_in_proj_w"],
            x_action_ptr,
            sa, DEC_D, ACTION_DIM, stream=stream)
        fvk.add_bias_fp16(
            x_action_ptr,
            W["decoder_action_in_proj_b"],
            sa, DEC_D, stream=stream)
        
        # Add time_emb[1:] to action projection
        time_emb_action_ptr = time_emb_ptr + DEC_D * 2  # Skip first token
        fvk.residual_add_fp16(
            x_action_ptr,
            time_emb_action_ptr,
            sa * DEC_D, stream)
    
    def transformer_decoder(self, stream: int = 0):
        """Run decoder + 10-step diffusion with AdaRMSNorm."""
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        es = self.encoder_seq_len
        sd = self.S_dec
        sa = self.chunk_size
        
        for step in range(self.num_steps):
            # Assemble decoder_x with time embedding
            self._assemble_decoder_x(step, stream)
            
            # Decoder layers with AdaRMSNorm (use configured number)
            for i in range(self.dec_layers):
                self._decoder_layer(i, step, es, sd, stream)
            
            # Final AdaRMSNorm + output projection
            style_final_ptr = self._style_slice_ptr("decoder_style_final", step)
            
            self._ada_rms_norm_fp16(
                _p(B["decoder_x"]),
                style_final_ptr,
                _p(B["x_normed_buf"]),
                _p(B["gate_buf"]),
                sd, DEC_D, stream)
            
            # Skip the first token (time_emb), use x_normed_buf[1:] for action output
            x_out_action_ptr = _p(B["x_normed_buf"]) + DEC_D * 2
            
            # Action out projection (accumulate into noise)
            gemm.fp16_nn(
                x_out_action_ptr,
                W["decoder_action_out_proj_w"],
                _p(B["diffusion_noise"]),
                sa, ACTION_DIM, DEC_D, stream=stream)
            self.fvk.add_bias_fp16(
                _p(B["diffusion_noise"]),
                W["decoder_action_out_proj_b"],
                sa, ACTION_DIM, stream)
    
    def _decoder_layer(self, i: int, step: int, enc_seq: int, sd: int, stream: int):
        """One decoder layer with AdaRMSNorm style modulation.
        
        Uses pre-allocated buffers for CUDA Graph compatibility (no dynamic allocation).
        """
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        
        # Get style for this layer and step
        style_attn_ptr = self._style_slice_ptr("decoder_style_attn", step, i)
        style_ffn_ptr = self._style_slice_ptr("decoder_style_ffn", step, i)
        
        # C1: AdaRMSNorm (attention) + QKV
        self._ada_rms_norm_fp16(
            _p(B["decoder_x"]),
            style_attn_ptr,
            _p(B["x_normed_buf"]),
            _p(B["gate_buf"]),
            sd, DEC_D, stream)
        
        dec_qkv_dim = (DEC_NH + 2 * DEC_NKV) * DEC_HD
        gemm.fp16_nn(
            _p(B["x_normed_buf"]),
            W["decoder_attn_qkv_w"][i],
            _p(B["decoder_QKV"]),
            sd, dec_qkv_dim, DEC_D, stream=stream)
        
        # Split QKV using pre-allocated tensor (no dynamic allocation)
        qkv_tensor = self._dec_qkv_tensor
        fvk.gpu_copy(qkv_tensor.data_ptr(), _p(B["decoder_QKV"]), sd * dec_qkv_dim * 2, stream)
        
        Q = qkv_tensor[:, :DEC_NH * DEC_HD].view(sd, DEC_NH, DEC_HD)
        KV = qkv_tensor[:, DEC_NH * DEC_HD:].view(sd, 2, DEC_NKV, DEC_HD)
        K_dec = KV[:, 0, :, :]
        V_dec = KV[:, 1, :, :]
        
        # Apply RoPE using pre-allocated tensor
        fvk.gpu_copy(self._dec_rope_tensor.data_ptr(), _p(B["decoder_rope"]), sd * 256 * 2, stream)
        
        Q_rope = self._apply_rope_interleaved(Q, self._dec_rope_cos, self._dec_rope_sin)
        K_dec_rope = self._apply_rope_interleaved(K_dec, self._dec_rope_cos, self._dec_rope_sin)
        
        # Cross-attention with encoder KV cache (pre-allocated tensors)
        enc_K_ptr = _p(B["encoder_K_cache"]) + i * enc_seq * ENC_NKV * ENC_HD * 2
        enc_V_ptr = _p(B["encoder_V_cache"]) + i * enc_seq * ENC_NKV * ENC_HD * 2
        
        fvk.gpu_copy(self._dec_enc_K_tensor.data_ptr(), enc_K_ptr, enc_seq * ENC_NKV * ENC_HD * 2, stream)
        fvk.gpu_copy(self._dec_enc_V_tensor.data_ptr(), enc_V_ptr, enc_seq * ENC_NKV * ENC_HD * 2, stream)
        
        # Expand encoder KV for GQA
        enc_K_expanded = self._dec_enc_K_tensor.expand(-1, DEC_NH, -1)
        enc_V_expanded = self._dec_enc_V_tensor.expand(-1, DEC_NH, -1)
        
        Q_t = Q_rope.transpose(0, 1)
        K_t = enc_K_expanded.transpose(0, 1)
        V_t = enc_V_expanded.transpose(0, 1)
        
        attn_out = F.scaled_dot_product_attention(Q_t, K_t, V_t)
        attn_out = attn_out.transpose(0, 1).reshape(sd, DEC_NH * DEC_HD)
        
        fvk.gpu_copy(_p(B["decoder_attn_out"]), attn_out.data_ptr(), sd * DEC_NH * DEC_HD * 2, stream)
        
        # O projection
        gemm.fp16_nn(
            _p(B["decoder_attn_out"]),
            W["decoder_attn_o_w"][i],
            _p(B["x_normed_buf"]),
            sd, DEC_D, DEC_NH * DEC_HD, stream=stream)
        
        # gate * residual (AdaRMSNorm gate from C1)
        self._gate_residual_fp16(
            _p(B["decoder_x"]),
            _p(B["x_normed_buf"]),
            _p(B["gate_buf"]),
            sd * DEC_D, stream)
        
        # C4: AdaRMSNorm (FFN) + FFN
        self._ada_rms_norm_fp16(
            _p(B["decoder_x"]),
            style_ffn_ptr,
            _p(B["x_normed_buf"]),
            _p(B["gate_buf"]),
            sd, DEC_D, stream)
        
        gemm.fp16_nn(
            _p(B["x_normed_buf"]),
            W["decoder_ffn_gate_w"][i],
            _p(B["decoder_gate_merged"]),
            sd, DEC_H, DEC_D, stream=stream)
        gemm.fp16_nn(
            _p(B["x_normed_buf"]),
            W["decoder_ffn_up_w"][i],
            _p(B["decoder_hidden"]),
            sd, DEC_H, DEC_D, stream=stream)
        
        fvk.gate_geglu_fp16(
            _p(B["decoder_gate_merged"]),
            _p(B["decoder_hidden"]),
            _p(B["decoder_hidden"]),
            sd * DEC_H, stream=stream)
        
        gemm.fp16_nn(
            _p(B["decoder_hidden"]),
            W["decoder_ffn_down_w"][i],
            _p(B["decoder_attn_out"]),
            sd, DEC_D, DEC_H, stream=stream)
        
        # gate * residual
        self._gate_residual_fp16(
            _p(B["decoder_x"]),
            _p(B["decoder_attn_out"]),
            _p(B["gate_buf"]),
            sd * DEC_D, stream)
    
    # ══════════════════════════════════════════════════════════════════
    #  Full pipeline
    # ══════════════════════════════════════════════════════════════════
    
    def run_pipeline(self, stream: int = 0, sync: bool = True):
        """Run full inference pipeline.
        
        Args:
            stream: CUDA stream for operations.
            sync: Whether to synchronize at the end. Set False for CUDA Graph capture.
        """
        self._copy_lang_embeds_to_encoder_x(stream)
        self.vision_encoder(stream)
        self.transformer_encoder(stream)
        self.transformer_decoder(stream)
        if sync:
            self._cudart.cudaStreamSynchronize(ctypes.c_void_p(stream))
    
    def set_language_embeds(self, lang_embeds_np: np.ndarray):
        """Set language embedding from prompt."""
        prompt_len = lang_embeds_np.shape[0]
        assert prompt_len <= self.max_prompt_len
        
        arr = np.ascontiguousarray(lang_embeds_np)
        self._lang_embeds_buf = CudaBuffer.from_numpy(arr)
        self._current_prompt_len = prompt_len
    
    def _copy_lang_embeds_to_encoder_x(self, stream: int):
        """Copy language embeddings to encoder_x buffer."""
        if not hasattr(self, "_lang_embeds_buf"):
            return
        
        start_byte = self.vision_seq * ENC_D * 2
        dst_ptr = _p(self.bufs["encoder_x"]) + start_byte
        
        self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(dst_ptr),
            self._lang_embeds_buf.ptr,
            self._lang_embeds_buf.nbytes, 3, stream)
    
    @property
    def input_images_buf(self):
        return self.bufs["observation_images_normalized"]
    
    @property
    def input_noise_buf(self):
        return self.bufs["diffusion_noise"]
    
    @property
    def output_noise_buf(self):
        return self.bufs["diffusion_noise"]