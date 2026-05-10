"""FlashRT — SM89 FP16 Pi0 inference pipeline.

SM89 (RTX 4060 Ti/4090) has limited FP8 GEMM support in cuBLAS.
This pipeline uses FP16 GEMM throughout, avoiding FP8 quantization.

Key differences from pipeline_rtx.py:
- No FP8 quantization of weights or activations
- Uses torch.nn.functional.scaled_dot_product_attention (avoids FA2 alignment issues)
- No calibration needed (FP16 doesn't require dynamic scaling)
- Compatible with Pi0 architecture (state_proj + action_time_mlp + flow-matching)

Architecture matches pi0_rtx.py:
- Vision: 27 SigLIP layers (head_dim=72, uses torch attention)
- Encoder: 18 Gemma-2B layers (GQA 8Q/1KV, head_dim=256)
- Decoder: 18 Gemma-300M layers (GQA 8Q/1KV, head_dim=256)
- Diffusion: 10-step flow-matching with state conditioning
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


# Fixed Pi0 model dimensions (same as pipeline_rtx.py)
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
DEC_D = 1024  # hidden_dim
DEC_H = 4096  # FFN hidden
DEC_NH = 8    # num_heads for Q
DEC_NKV = 1   # num_kv_heads (GQA)
DEC_HD = 256  # head_dim = 256 (q_proj: 2048/8, k_proj: 256/1)

ACTION_DIM = 32
CHUNK_SIZE_DEFAULT = 10
NUM_STEPS_DEFAULT = 10

FP16 = np.float16
FP32 = np.float32


def _p(buf) -> int:
    """Extract int pointer from a CudaBuffer."""
    return buf.ptr.value


class Pi0PipelineSm89:
    """Pi0 inference pipeline for SM89 (FP16, no FP8).
    
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
        
        # CUDART for D2D copies
        self._cudart = ctypes.CDLL("libcudart.so")
        
        # Pre-create torch tensors for CUDA Graph compatibility
        # (No dynamic allocation during inference)
        self._create_preallocated_tensors()
        
        logger.info(
            "Pi0PipelineSm89 initialised (num_views=%d, vision_seq=%d, "
            "encoder_seq=%d, chunk_size=%d, num_steps=%d)",
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
        enc_qkv_dim = (ENC_NH + 2 * ENC_NKV) * ENC_HD
        
        # Decoder scratch tensors (pre-created, reused per-layer per-step)
        self._dec_qkv_tensor = torch.empty(sd, dec_qkv_dim, dtype=torch.float16, device='cuda')
        self._dec_rope_tensor = torch.empty(sd, 256, dtype=torch.float16, device='cuda')
        self._dec_enc_K_tensor = torch.empty(es, ENC_NKV, ENC_HD, dtype=torch.float16, device='cuda')
        self._dec_enc_V_tensor = torch.empty(es, ENC_NKV, ENC_HD, dtype=torch.float16, device='cuda')
        
        # Encoder scratch tensors (pre-created, reused per-layer)
        self._enc_qkv_tensor = torch.empty(es, enc_qkv_dim, dtype=torch.float16, device='cuda')
        self._enc_rope_tensor = torch.empty(es, 256, dtype=torch.float16, device='cuda')
        
        # Store views for common patterns
        self._dec_rope_cos = self._dec_rope_tensor[:, :128]
        self._dec_rope_sin = self._dec_rope_tensor[:, 128:]
        self._enc_rope_cos = self._enc_rope_tensor[:, :128]
        self._enc_rope_sin = self._enc_rope_tensor[:, 128:]
        
        logger.info("Created pre-allocated decoder/encoder tensors for CUDA Graph")
    
    def _allocate_buffers(self) -> dict:
        """Allocate all pipeline working buffers."""
        nv = self.num_views
        vs = self.vision_seq
        es = self.encoder_seq_len
        sd = self.S_dec
        sa = self.chunk_size
        B = {}
        
        # Vision (SigLIP)
        B["observation_images_normalized"] = CudaBuffer.device_empty(
            nv * 224 * 224 * 3, FP16)
        B["vision_x"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        B["vision_x_norm"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        B["vision_QKV"] = CudaBuffer.device_empty(vs * 3 * VIS_D, FP16)
        B["vision_hidden"] = CudaBuffer.device_empty(vs * VIS_H, FP16)
        B["vision_patches"] = CudaBuffer.device_empty(vs * VIS_PATCH_FLAT, FP16)
        B["vision_pos_embed_expanded"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        B["vision_attn_out"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        B["vision_ffn_out"] = CudaBuffer.device_empty(vs * VIS_D, FP16)  # Temporary buffer for FFN down
        
        # Encoder (Gemma-2B)
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
        
        # RMSNorm ones buffers (pre-allocated for reuse)
        B["encoder_ones"] = CudaBuffer.from_numpy(np.ones(ENC_D, dtype=FP16))
        B["decoder_ones"] = CudaBuffer.from_numpy(np.ones(DEC_D, dtype=FP16))
        
        # Encoder FFN down output buffer
        B["encoder_ffn_out"] = CudaBuffer.device_empty(es * ENC_D, FP16)
        
        # Decoder pre-processing
        B["state_buf"] = CudaBuffer.device_empty(1 * ACTION_DIM, FP16)
        B["state_token"] = CudaBuffer.device_empty(1 * DEC_D, FP16)
        B["action_time_temp"] = CudaBuffer.device_empty(sa * DEC_D, FP16)
        
        # Decoder (Gemma-300M)
        B["decoder_x"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        B["decoder_x_norm"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        B["decoder_QKV"] = CudaBuffer.device_empty(
            sd * (DEC_NH + 2 * DEC_NKV) * DEC_HD, FP16)
        B["decoder_hidden"] = CudaBuffer.device_empty(sd * DEC_H, FP16)
        B["decoder_gate_merged"] = CudaBuffer.device_empty(sd * 2 * DEC_H, FP16)
        B["decoder_attn_out"] = CudaBuffer.device_empty(sd * DEC_NH * DEC_HD, FP16)  # (sd, 2048) for attention output before O proj
        
        # Diffusion
        B["diffusion_noise"] = CudaBuffer.device_empty(sa * ACTION_DIM, FP16)
        B["decoder_action_buf"] = CudaBuffer.device_empty(sa * ACTION_DIM, FP16)
        
        return B
    
    def _build_rope_table(self):
        """Build RoPE cos/sin tables."""
        max_pos = self.encoder_seq_len + self.S_dec
        
        # Standard RoPE formula
        inv_freq = 1.0 / (10000 ** (np.arange(0, 256, 2, dtype=np.float64) / 256))
        positions = np.arange(max_pos, dtype=np.float64)
        phase = positions[:, None] * inv_freq[None, :]
        
        cos = np.cos(phase).astype(FP16)
        sin = np.sin(phase).astype(FP16)
        
        # Interleaved format: [cos, sin] pairs
        interleaved = np.stack([cos, sin], axis=-1).reshape(max_pos, 256)
        self._rope_table_np = interleaved
        
        # Encoder RoPE
        enc_rope = interleaved[:self.encoder_seq_len]
        self.bufs["encoder_rope"] = CudaBuffer.from_numpy(
            np.ascontiguousarray(enc_rope))
        
        # Decoder RoPE (starts after encoder)
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
    #  Phase A: Vision Encoder (SigLIP 27 layers)
    # ══════════════════════════════════════════════════════════════════
    
    def vision_encoder(self, stream: int = 0):
        """Run SigLIP vision encoder."""
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
        
        # 27 SigLIP layers
        for i in range(VIS_L):
            self._vision_layer(i, stream)
    
    def _vision_layer(self, i: int, stream: int):
        """One SigLIP layer with torch attention."""
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
        
        # Multi-view batched attention (each view attends independently)
        attn_out = torch.empty(seq, VIS_D, dtype=torch.float16, device='cuda')
        for v in range(self.num_views):
            start = v * VIS_SEQ_PER_VIEW
            end = (v + 1) * VIS_SEQ_PER_VIEW
            Q_view = Q[start:end].transpose(0, 1)  # (NH, Sper, HD)
            K_view = K[start:end].transpose(0, 1)
            V_view = V[start:end].transpose(0, 1)
            
            # Torch scaled_dot_product_attention (Flash attention)
            attn_result = F.scaled_dot_product_attention(Q_view, K_view, V_view)
            attn_out[start:end] = attn_result.transpose(0, 1).reshape(
                VIS_SEQ_PER_VIEW, VIS_D)
        
        # Copy back to buffer
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
        
        # FFN up (GEMM + GELU)
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
        
        # FFN down (use temporary buffer to avoid overlap)
        gemm.fp16_nn(
            _p(B["vision_hidden"]),
            W["vision_ffn_down_w"][i],
            _p(B["vision_ffn_out"]),
            seq, VIS_D, VIS_H, stream=stream)
        fvk.add_bias_fp16(
            _p(B["vision_ffn_out"]),
            W["vision_ffn_down_b"][i],
            seq, VIS_D, stream=stream)
        
        # Final residual
        fvk.residual_add_fp16(_p(B["vision_x"]), _p(B["vision_ffn_out"]), seq * VIS_D, stream)
    
    # ══════════════════════════════════════════════════════════════════
    #  Phase B: Gemma-2B Encoder (18 layers)
    # ══════════════════════════════════════════════════════════════════
    
    def transformer_encoder(self, stream: int = 0):
        """Run Gemma-2B encoder."""
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
        
        # Language embeddings (already copied by frontend)
        
        # Encoder layers (use configured number)
        for i in range(self.enc_layers):
            self._encoder_layer(i, es, stream)
    
    def _encoder_layer(self, i: int, seq: int, stream: int):
        """One Gemma-2B encoder layer with torch attention + RoPE."""
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        
        # RMSNorm (Gemma-style) - use pre-allocated ones buffer
        fvk.rms_norm_fp16(
            _p(B["encoder_x"]),
            _p(B["encoder_ones"]),
            _p(B["encoder_x_norm"]),
            seq, ENC_D, 1e-6, stream=stream)
        
        # QKV GEMM (GQA: 8Q heads + 1KV head)
        gemm.fp16_nn(
            _p(B["encoder_x_norm"]),
            W["encoder_attn_qkv_w"][i],
            _p(B["encoder_QKV"]),
            seq, (ENC_NH + 2 * ENC_NKV) * ENC_HD, ENC_D, stream=stream)
        
        # Split QKV - use pre-allocated tensor
        qkv_tensor = self._enc_qkv_tensor
        fvk.gpu_copy(qkv_tensor.data_ptr(), _p(B["encoder_QKV"]), 
                     seq * (ENC_NH + 2 * ENC_NKV) * ENC_HD * 2, stream)
        
        Q = qkv_tensor[:, :ENC_NH * ENC_HD].view(seq, ENC_NH, ENC_HD)
        KV = qkv_tensor[:, ENC_NH * ENC_HD:].view(seq, 2, ENC_NKV, ENC_HD)
        K = KV[:, 0, :, :]  # (seq, 1, 256)
        V = KV[:, 1, :, :]
        
        # Apply RoPE - use pre-allocated tensor
        rope_tensor = self._enc_rope_tensor
        fvk.gpu_copy(rope_tensor.data_ptr(), _p(B["encoder_rope"]), seq * 256 * 2, stream)
        cos = self._enc_rope_cos
        sin = self._enc_rope_sin
        
        # RoPE rotation (interleaved)
        Q_rope = self._apply_rope_interleaved(Q, cos, sin)  # (seq, NH, HD)
        K_rope = self._apply_rope_interleaved(K, cos, sin)  # K is (seq, NKV, HD), returns (seq, NKV, HD)
        
        # Write K/V to cache (for cross-attention) - separate buffers for K and V
        K_cache_offset = i * seq * ENC_NKV * ENC_HD * 2
        V_cache_offset = i * seq * ENC_NKV * ENC_HD * 2  # Relative to V_cache buffer
        
        fvk.gpu_copy(_p(B["encoder_K_cache"]) + K_cache_offset, 
                     K_rope.data_ptr(), seq * ENC_NKV * ENC_HD * 2, stream)
        fvk.gpu_copy(_p(B["encoder_V_cache"]) + V_cache_offset,
                     V.data_ptr(), seq * ENC_NKV * ENC_HD * 2, stream)
        
# Torch attention (GQA: expand KV to match Q heads)
        # K_rope: (seq, NKV=1, HD=256), expand to (seq, NH=8, HD=256)
        K_expanded = K_rope.expand(-1, ENC_NH, -1)  # (seq, NH, HD)
        V_expanded = V.expand(-1, ENC_NH, -1)  # V is (seq, 1, 256), expand to (seq, 8, 256)
        
        Q_t = Q_rope.transpose(0, 1)  # (NH, seq, HD)
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
        
        # Residual
        fvk.residual_add_fp16(_p(B["encoder_x"]), _p(B["encoder_x_norm"]), seq * ENC_D, stream)
        
        # FFN (SwiGLU: gate + up) - use pre-allocated ones buffer
        fvk.rms_norm_fp16(
            _p(B["encoder_x"]),
            _p(B["encoder_ones"]),
            _p(B["encoder_x_norm"]),
            seq, ENC_D, 1e-6, stream=stream)
        
        # Gate+Up merged GEMM
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
        
        # SwiGLU: SiLU(gate) * up
        fvk.gate_geglu_fp16(
            _p(B["encoder_gate_merged"]),
            _p(B["encoder_hidden"]),
            _p(B["encoder_hidden"]),
            seq * ENC_H, stream=stream)
        
        # Down projection (use temporary buffer to avoid overlap)
        gemm.fp16_nn(
            _p(B["encoder_hidden"]),
            W["encoder_ffn_down_w"][i],
            _p(B["encoder_ffn_out"]),
            seq, ENC_D, ENC_H, stream=stream)
        
        # Residual
        fvk.residual_add_fp16(_p(B["encoder_x"]), _p(B["encoder_ffn_out"]), seq * ENC_D, stream)
    
    def _apply_rope_interleaved(self, x, cos, sin):
        """Apply interleaved RoPE."""
        # x: (seq, heads, head_dim) or (seq, head_dim)
        # cos, sin: (seq, head_dim // 2)
        x_even = x[..., :x.shape[-1]//2]
        x_odd = x[..., x.shape[-1]//2:]
        
        # Handle broadcasting based on input dimensions
        # cos/sin: (seq, 128) -> need to match x_even/x_odd dimensions
        if x.dim() == 3:  # (seq, heads, head_dim)
            # cos.unsqueeze(1): (seq, 1, 128) broadcasts to (seq, heads, 128)
            cos_exp = cos.unsqueeze(1)
            sin_exp = sin.unsqueeze(1)
        else:  # (seq, head_dim)
            # No extra dimension needed
            cos_exp = cos
            sin_exp = sin
        
        # Rotate
        rotated_even = x_even * cos_exp - x_odd * sin_exp
        rotated_odd = x_even * sin_exp + x_odd * cos_exp
        
        return torch.cat([rotated_even, rotated_odd], dim=-1)
    
    # ══════════════════════════════════════════════════════════════════
    #  Phase C: Decoder + Diffusion (flow-matching)
    # ══════════════════════════════════════════════════════════════════
    
    def _state_project(self, stream: int):
        """Project state vector to decoder initial token."""
        B = self.bufs
        W = self.weights
        fvk = self.fvk
        
        fvk.gmm_fp16(
            self._ctx,
            _p(B["state_buf"]),
            W["state_proj_w"],
            _p(B["state_token"]),
            1, DEC_D, ACTION_DIM, 0.0, stream)
        fvk.add_bias_fp16(
            _p(B["state_token"]),
            W["state_proj_b"],
            1, DEC_D, stream)
    
    def _assemble_decoder_x(self, step: int, stream: int):
        """Build decoder_x = [state_token; action_time_mlp(noise, t)]."""
        B = self.bufs
        W = self.weights
        fvk = self.fvk
        sa = self.chunk_size
        
        # 1) Copy state_token to decoder_x[0]
        self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(_p(B["decoder_x"])),
            ctypes.c_void_p(_p(B["state_token"])),
            DEC_D * 2, 3, stream)
        
        # 2) action_in_proj(noise) → decoder_x[1:S_dec]
        x_action_ptr = _p(B["decoder_x"]) + DEC_D * 2
        fvk.gmm_fp16(
            self._ctx,
            _p(B["diffusion_noise"]),
            W["decoder_action_in_proj_w"],
            x_action_ptr,
            sa, DEC_D, ACTION_DIM, 0.0, stream)
        fvk.add_bias_fp16(
            x_action_ptr,
            W["decoder_action_in_proj_b"],
            sa, DEC_D, stream)
        
        # 3) action_time_mlp
        fvk.gmm_fp16(
            self._ctx,
            x_action_ptr,
            W["action_time_mlp_in_wa_w"],
            _p(B["action_time_temp"]),
            sa, DEC_D, DEC_D, 0.0, stream)
        
        time_proj_ptr = W["time_proj_all"] + step * sa * DEC_D * 2
        fvk.fused_add_silu_fp16(
            _p(B["action_time_temp"]),
            time_proj_ptr,
            sa * DEC_D, stream)
        
        fvk.gmm_fp16(
            self._ctx,
            _p(B["action_time_temp"]),
            W["action_time_mlp_out_w"],
            x_action_ptr,
            sa, DEC_D, DEC_D, 0.0, stream)
        fvk.add_bias_fp16(
            x_action_ptr,
            W["action_time_mlp_out_b"],
            sa, DEC_D, stream)
    
    def transformer_decoder(self, stream: int = 0):
        """Run decoder + 10-step diffusion."""
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        es = self.encoder_seq_len
        sd = self.S_dec
        sa = self.chunk_size
        
        for step in range(self.num_steps):
            # Assemble decoder_x for this step
            self._assemble_decoder_x(step, stream)
            
            # Decoder layers (use configured number)
            for i in range(self.dec_layers):
                self._decoder_layer(i, es, sd, stream)
            
            # Final norm - use pre-allocated ones buffer
            fvk.rms_norm_fp16(
                _p(B["decoder_x"]),
                _p(B["decoder_ones"]),
                _p(B["decoder_x_norm"]),
                sd, DEC_D, 1e-6, stream=stream)
            
            # Action out projection (accumulate into noise)
            x_out_action_ptr = _p(B["decoder_x_norm"]) + DEC_D * 2
            fvk.gmm_fp16(
                self._ctx,
                x_out_action_ptr,
                W["decoder_action_out_proj_w"],
                _p(B["diffusion_noise"]),
                sa, ACTION_DIM, DEC_D, 1.0, stream)  # beta=1.0 (accumulate)
            fvk.add_bias_fp16(
                _p(B["diffusion_noise"]),
                W["decoder_action_out_proj_b"],
                sa, ACTION_DIM, stream)
    
    def _decoder_layer(self, i: int, enc_seq: int, sd: int, stream: int):
        """One decoder layer with cross-attention."""
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        
        # RMSNorm - use pre-allocated ones buffer
        fvk.rms_norm_fp16(
            _p(B["decoder_x"]),
            _p(B["decoder_ones"]),
            _p(B["decoder_x_norm"]),
            sd, DEC_D, 1e-6, stream=stream)
        
        # QKV GEMM - QKV dim = (NH + 2*NKV) * HD = (8 + 2*1) * 256 = 2560
        dec_qkv_dim = (DEC_NH + 2 * DEC_NKV) * DEC_HD  # 2560
        gemm.fp16_nn(
            _p(B["decoder_x_norm"]),
            W["decoder_attn_qkv_w"][i],
            _p(B["decoder_QKV"]),
            sd, dec_qkv_dim, DEC_D, stream=stream)
        
        # Split QKV - use pre-allocated tensor
        qkv_tensor = self._dec_qkv_tensor
        fvk.gpu_copy(qkv_tensor.data_ptr(), _p(B["decoder_QKV"]), sd * dec_qkv_dim * 2, stream)
        
        Q = qkv_tensor[:, :DEC_NH * DEC_HD].view(sd, DEC_NH, DEC_HD)  # (sd, 8, 256)
        KV = qkv_tensor[:, DEC_NH * DEC_HD:].view(sd, 2, DEC_NKV, DEC_HD)  # (sd, 2, 1, 256)
        K_dec = KV[:, 0, :, :]  # (sd, 1, 256)
        V_dec = KV[:, 1, :, :]
        
        # Apply RoPE - use pre-allocated tensor
        rope_tensor = self._dec_rope_tensor
        fvk.gpu_copy(rope_tensor.data_ptr(), _p(B["decoder_rope"]), sd * 256 * 2, stream)
        cos = self._dec_rope_cos  # (sd, 128) = HD//2
        sin = self._dec_rope_sin  # (sd, 128)
        
        Q_rope = self._apply_rope_interleaved(Q, cos, sin)  # Q is (sd, NH, HD=256)
        K_dec_rope = self._apply_rope_interleaved(K_dec, cos, sin)  # K_dec is (sd, NKV, HD=256)
        
        # Cross-attention: decoder queries attend to encoder KV
        # For simplicity, use torch tensors directly without cache writes
        
        # Load encoder K/V from cache - use pre-allocated tensors
        enc_K_tensor = self._dec_enc_K_tensor
        enc_V_tensor = self._dec_enc_V_tensor
        
        enc_K_ptr = _p(B["encoder_K_cache"]) + i * enc_seq * ENC_NKV * ENC_HD * 2
        enc_V_ptr = _p(B["encoder_V_cache"]) + i * enc_seq * ENC_NKV * ENC_HD * 2
        
        fvk.gpu_copy(enc_K_tensor.data_ptr(), enc_K_ptr, enc_seq * ENC_NKV * ENC_HD * 2, stream)
        fvk.gpu_copy(enc_V_tensor.data_ptr(), enc_V_ptr, enc_seq * ENC_NKV * ENC_HD * 2, stream)
        
        # Expand encoder KV for GQA
        enc_K_expanded = enc_K_tensor.expand(-1, DEC_NH, -1)  # (enc_seq, NH, HD)
        enc_V_expanded = enc_V_tensor.expand(-1, DEC_NH, -1)
        
        # For cross-attention: Q from decoder, K/V from encoder
        Q_t = Q_rope.transpose(0, 1)  # (NH, sd, HD)
        K_t = enc_K_expanded.transpose(0, 1)  # (NH, enc_seq, HD)
        V_t = enc_V_expanded.transpose(0, 1)
        
        # Cross-attention
        attn_out = F.scaled_dot_product_attention(Q_t, K_t, V_t)
        attn_out = attn_out.transpose(0, 1).reshape(sd, DEC_NH * DEC_HD)  # (sd, 2048)
        
        fvk.gpu_copy(_p(B["decoder_attn_out"]), attn_out.data_ptr(), sd * DEC_NH * DEC_HD * 2, stream)
        
        # O projection: (sd, 2048) @ (2048, 1024) -> (sd, 1024)
        gemm.fp16_nn(
            _p(B["decoder_attn_out"]),
            W["decoder_attn_o_w"][i],
            _p(B["decoder_x_norm"]),
            sd, DEC_D, DEC_NH * DEC_HD, stream=stream)
        
        # Residual
        fvk.residual_add_fp16(_p(B["decoder_x"]), _p(B["decoder_x_norm"]), sd * DEC_D, stream)
        
        # FFN - use pre-allocated ones buffer
        fvk.rms_norm_fp16(
            _p(B["decoder_x"]),
            _p(B["decoder_ones"]),
            _p(B["decoder_x_norm"]),
            sd, DEC_D, 1e-6, stream=stream)
        
        gemm.fp16_nn(
            _p(B["decoder_x_norm"]),
            W["decoder_ffn_gate_w"][i],
            _p(B["decoder_gate_merged"]),
            sd, DEC_H, DEC_D, stream=stream)
        gemm.fp16_nn(
            _p(B["decoder_x_norm"]),
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
        
        fvk.residual_add_fp16(_p(B["decoder_x"]), _p(B["decoder_attn_out"]), sd * DEC_D, stream)
    
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
        self._state_project(stream)
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
    def input_state_buf(self):
        return self.bufs["state_buf"]
    
    @property
    def output_noise_buf(self):
        """Return final action output buffer."""
        return self.bufs["diffusion_noise"]