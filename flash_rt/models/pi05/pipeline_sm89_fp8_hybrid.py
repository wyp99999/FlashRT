"""FlashRT — SM89 FP8 Pi0.5 inference pipeline (简化稳定版).

关键优化:
- Encoder FFN大矩阵使用FP8 GEMM (当M被16整除时)
- Vision/Attention/Decoder使用FP16 (避免alignment问题)
- 权重预量化到FP8 ColumnMajor格式

性能目标: <50ms
"""

from __future__ import annotations

import ctypes
import logging
import math

import numpy as np
import torch
import torch.nn.functional as F

from flash_rt.core.cuda_buffer import CudaBuffer
import flash_rt.flash_rt_kernels as frk

logger = logging.getLogger(__name__)

# Fixed Pi0.5 model dimensions
VIS_L = 27
VIS_D = 1152
VIS_H = 4304
VIS_NH = 16
VIS_HD = 72
VIS_SEQ_PER_VIEW = 256
VIS_PATCH_FLAT = 14 * 14 * 3

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
fp8_e4m3 = torch.float8_e4m3fn


def _p(buf) -> int:
    return buf.ptr.value


class Pi05PipelineSm89Fp8Hybrid:
    """Pi0.5 hybrid pipeline: FP16 main + FP8 for Encoder FFN.
    
    简化版优化策略:
    - Vision: FP16 (27层, seq=256)
    - Encoder: FP16 attention + FP8 FFN (当M被16整除时)
    - Decoder: FP16 (seq=11很小)
    
    这样可以避免FP8 kernel alignment问题，同时获得Encoder FFN的FP8加速。
    """
    
    def __init__(
        self,
        fvk,
        weights_fp16,
        weights_fp8_optional,  # FP8权重(可选，用于FFN)
        fp8_scales_optional,
        *,
        num_views: int,
        max_prompt_len: int,
        chunk_size: int = CHUNK_SIZE_DEFAULT,
        num_steps: int = NUM_STEPS_DEFAULT,
    ):
        self.fvk = fvk
        self.weights = weights_fp16  # 主要使用FP16权重
        self.weights_fp8 = weights_fp8_optional  # FP8权重(可选)
        self.fp8_scales = fp8_scales_optional
        
        self.num_views = num_views
        self.max_prompt_len = max_prompt_len
        self.chunk_size = chunk_size
        self.S_dec = chunk_size + 1
        self.num_steps = num_steps
        
        self.vision_seq = num_views * VIS_SEQ_PER_VIEW
        self.encoder_seq_len = self.vision_seq + max_prompt_len
        
        # Check if FP8 can be used for Encoder FFN
        self.use_fp8_encoder_ffn = (self.encoder_seq_len % 16 == 0)
        if self.use_fp8_encoder_ffn:
            logger.info("FP8 enabled for Encoder FFN (seq=%d divisible by 16)", self.encoder_seq_len)
        else:
            logger.info("FP8 disabled for Encoder FFN (seq=%d not divisible by 16, using FP16)", self.encoder_seq_len)
        
        self.enc_layers = ENC_L
        self.dec_layers = DEC_L
        
        self.bufs = self._allocate_buffers()
        self.gemm = fvk.GemmRunner()
        self._cudart = ctypes.CDLL("libcudart.so")
        
        # Pre-allocated tensors for CUDA Graph
        self._create_preallocated_tensors()
        
        # FP8 activation buffers (only if FP8 enabled)
        if self.use_fp8_encoder_ffn:
            self.encoder_x_fp8 = torch.empty(self.encoder_seq_len, ENC_D, dtype=fp8_e4m3, device='cuda')
            self.encoder_hidden_fp8 = torch.empty(self.encoder_seq_len, ENC_H, dtype=fp8_e4m3, device='cuda')
            self.scale_one = torch.tensor([1.0], dtype=torch.float32, device='cuda')
        
        logger.info(
            "Pi05PipelineSm89Fp8Hybrid initialised (num_views=%d, vision_seq=%d, "
            "encoder_seq_len=%d, fp8_encoder=%s)",
            num_views, self.vision_seq, self.encoder_seq_len, self.use_fp8_encoder_ffn)
    
    def _create_preallocated_tensors(self):
        """Create pre-allocated tensors for CUDA Graph."""
        sd = self.S_dec
        es = self.encoder_seq_len
        
        self._dec_qkv_tensor = torch.empty(sd, (DEC_NH + 2 * DEC_NKV) * DEC_HD, 
                                           dtype=torch.float16, device='cuda')
        self._dec_rope_tensor = torch.empty(sd, 256, dtype=torch.float16, device='cuda')
        self._dec_rope_cos = self._dec_rope_tensor[:, :128]
        self._dec_rope_sin = self._dec_rope_tensor[:, 128:]
        
        self._enc_K_tensor = torch.empty(es, ENC_NKV, ENC_HD, dtype=torch.float16, device='cuda')
        self._enc_V_tensor = torch.empty(es, ENC_NKV, ENC_HD, dtype=torch.float16, device='cuda')
        
        self._vis_qkv_tensor = torch.empty(self.vision_seq, 3 * VIS_D, dtype=torch.float16, device='cuda')
        self._encoder_qkv_tensor = torch.empty(es, (ENC_NH + 2 * ENC_NKV) * ENC_HD,
                                               dtype=torch.float16, device='cuda')
        self._encoder_rope_tensor = torch.empty(es, 256, dtype=torch.float16, device='cuda')
    
    def _allocate_buffers(self) -> dict:
        """Allocate working buffers."""
        nv = self.num_views
        vs = self.vision_seq
        es = self.encoder_seq_len
        sd = self.S_dec
        sa = self.chunk_size
        
        B = {}
        
        # Vision buffers
        B["observation_images_normalized"] = CudaBuffer.device_empty(nv * 224 * 224 * 3, FP16)
        B["vision_x"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        B["vision_x_norm"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        B["vision_QKV"] = CudaBuffer.device_empty(vs * 3 * VIS_D, FP16)
        B["vision_hidden"] = CudaBuffer.device_empty(vs * VIS_H, FP16)
        B["vision_patches"] = CudaBuffer.device_empty(vs * VIS_PATCH_FLAT, FP16)
        B["vision_pos_embed_expanded"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        B["vision_attn_out"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        B["vision_ffn_out"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        
        # Encoder buffers
        B["encoder_x"] = CudaBuffer.device_empty(es * ENC_D, FP16)
        B["encoder_x_norm"] = CudaBuffer.device_empty(es * ENC_D, FP16)
        B["encoder_QKV"] = CudaBuffer.device_empty(es * (ENC_NH + 2 * ENC_NKV) * ENC_HD, FP16)
        B["encoder_hidden"] = CudaBuffer.device_empty(es * ENC_H, FP16)
        B["encoder_gate_merged"] = CudaBuffer.device_empty(es * 2 * ENC_H, FP16)
        B["encoder_K_cache"] = CudaBuffer.device_empty(ENC_L * es * ENC_NKV * ENC_HD, FP16)
        B["encoder_V_cache"] = CudaBuffer.device_empty(ENC_L * es * ENC_NKV * ENC_HD, FP16)
        B["encoder_attn_out"] = CudaBuffer.device_empty(es * ENC_D, FP16)
        B["encoder_ffn_out"] = CudaBuffer.device_empty(es * ENC_D, FP16)
        
        # Decoder buffers
        B["decoder_x"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        B["decoder_x_norm"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        B["decoder_QKV"] = CudaBuffer.device_empty(sd * (DEC_NH + 2 * DEC_NKV) * DEC_HD, FP16)
        B["decoder_hidden"] = CudaBuffer.device_empty(sd * DEC_H, FP16)
        B["decoder_gate_merged"] = CudaBuffer.device_empty(sd * 2 * DEC_H, FP16)
        B["decoder_attn_out"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        
        # AdaRMSNorm buffers
        B["encoder_ones"] = CudaBuffer.from_numpy(np.ones(ENC_D, dtype=FP16))
        B["decoder_ones"] = CudaBuffer.from_numpy(np.ones(DEC_D, dtype=FP16))
        B["x_normed_buf"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        B["gate_buf"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        
        # Precomputed styles
        B["decoder_time_emb"] = CudaBuffer.device_empty(self.num_steps * sd * DEC_D, FP16)
        B["decoder_style_attn"] = CudaBuffer.device_empty(self.num_steps * DEC_L * sd * 3 * DEC_D, FP16)
        B["decoder_style_ffn"] = CudaBuffer.device_empty(self.num_steps * DEC_L * sd * 3 * DEC_D, FP16)
        B["decoder_style_final"] = CudaBuffer.device_empty(self.num_steps * sd * 3 * DEC_D, FP16)
        
        B["diffusion_noise"] = CudaBuffer.device_empty(sa * ACTION_DIM, FP16)
        
        return B
    
    def _build_rope_table(self):
        """Build RoPE cos/sin tables."""
        max_pos = self.encoder_seq_len + self.S_dec
        
        inv_freq = 1.0 / (10000 ** (np.arange(0, 256, 2, dtype=np.float64) / 256))
        positions = np.arange(max_pos, dtype=np.float64)
        phase = positions[:, None] * inv_freq[None, :]
        
        cos = np.cos(phase).astype(FP16)
        sin = np.sin(phase).astype(FP16)
        
        interleaved = np.stack([cos, sin], axis=-1).reshape(max_pos, 256)
        
        enc_rope = interleaved[:self.encoder_seq_len]
        self.bufs["encoder_rope"] = CudaBuffer.from_numpy(np.ascontiguousarray(enc_rope))
        
        dec_rope = interleaved[self.encoder_seq_len:self.encoder_seq_len + self.S_dec]
        self.bufs["decoder_rope"] = CudaBuffer.from_numpy(np.ascontiguousarray(dec_rope))
    
    def _style_slice_ptr(self, buf_name: str, step: int, layer: int | None = None) -> int:
        base = _p(self.bufs[buf_name])
        sd = self.S_dec
        
        if buf_name == "decoder_time_emb":
            return base + step * sd * DEC_D * 2
        if buf_name == "decoder_style_final":
            return base + step * sd * 3 * DEC_D * 2
        
        per_layer = sd * 3 * DEC_D * 2
        per_step = DEC_L * per_layer
        return base + step * per_step + layer * per_layer
    
    # ========== Vision Encoder (FP16) ==========
    
    def vision_encoder(self, stream: int = 0):
        """Run SigLIP vision encoder (FP16)."""
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        seq = self.vision_seq
        
        fvk.patch_im2col(_p(B["observation_images_normalized"]),
                        _p(B["vision_patches"]), self.num_views, stream)
        
        gemm.fp16_nn(_p(B["vision_patches"]), W["vision_patch_embedding_w"],
                    _p(B["vision_x"]), seq, VIS_D, VIS_PATCH_FLAT, stream=stream)
        
        fvk.bias_residual_fp16(_p(B["vision_x"]), _p(B["vision_pos_embed_expanded"]),
                               W["vision_patch_embedding_b"], seq, VIS_D, stream=stream)
        
        for i in range(VIS_L):
            self._vision_layer_fp16(i, stream)
    
    def _vision_layer_fp16(self, i: int, stream: int):
        """Vision layer using FP16."""
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        seq = self.vision_seq
        
        fvk.layer_norm_fp16(_p(B["vision_x"]), W["vision_pre_attn_norm_w"][i],
                           W["vision_pre_attn_norm_b"][i], _p(B["vision_x_norm"]),
                           seq, VIS_D, 1e-5, stream=stream)
        
        gemm.fp16_nn(_p(B["vision_x_norm"]), W["vision_attn_qkv_w"][i],
                    _p(B["vision_QKV"]), seq, 3 * VIS_D, VIS_D, stream=stream)
        fvk.add_bias_fp16(_p(B["vision_QKV"]), W["vision_attn_qkv_b"][i],
                         seq, 3 * VIS_D, stream=stream)
        
        fvk.gpu_copy(self._vis_qkv_tensor.data_ptr(), _p(B["vision_QKV"]),
                    seq * 3 * VIS_D * 2, stream)
        
        Q = self._vis_qkv_tensor[:, :VIS_D].view(seq, VIS_NH, VIS_HD)
        K = self._vis_qkv_tensor[:, VIS_D:2*VIS_D].view(seq, VIS_NH, VIS_HD)
        V = self._vis_qkv_tensor[:, 2*VIS_D:].view(seq, VIS_NH, VIS_HD)
        
        attn_out = torch.empty(seq, VIS_D, dtype=torch.float16, device='cuda')
        for v in range(self.num_views):
            start = v * VIS_SEQ_PER_VIEW
            end = (v + 1) * VIS_SEQ_PER_VIEW
            Q_view = Q[start:end].transpose(0, 1)
            K_view = K[start:end].transpose(0, 1)
            V_view = V[start:end].transpose(0, 1)
            attn_result = F.scaled_dot_product_attention(Q_view, K_view, V_view)
            attn_out[start:end] = attn_result.transpose(0, 1).reshape(VIS_SEQ_PER_VIEW, VIS_D)
        
        fvk.gpu_copy(_p(B["vision_attn_out"]), attn_out.data_ptr(), seq * VIS_D * 2, stream)
        
        gemm.fp16_nn(_p(B["vision_attn_out"]), W["vision_attn_o_w"][i],
                    _p(B["vision_x_norm"]), seq, VIS_D, VIS_D, stream=stream)
        fvk.bias_residual_fp16(_p(B["vision_x"]), _p(B["vision_x_norm"]),
                               W["vision_attn_o_b"][i], seq, VIS_D, stream=stream)
        
        fvk.layer_norm_fp16(_p(B["vision_x"]), W["vision_pre_ffn_norm_w"][i],
                           W["vision_pre_ffn_norm_b"][i], _p(B["vision_x_norm"]),
                           seq, VIS_D, 1e-5, stream=stream)
        
        gemm.fp16_nn(_p(B["vision_x_norm"]), W["vision_ffn_up_w"][i],
                    _p(B["vision_hidden"]), seq, VIS_H, VIS_D, stream=stream)
        fvk.add_bias_fp16(_p(B["vision_hidden"]), W["vision_ffn_up_b"][i],
                         seq, VIS_H, stream=stream)
        fvk.gelu_inplace_fp16(_p(B["vision_hidden"]), seq * VIS_H, stream=stream)
        
        gemm.fp16_nn(_p(B["vision_hidden"]), W["vision_ffn_down_w"][i],
                    _p(B["vision_ffn_out"]), seq, VIS_D, VIS_H, stream=stream)
        fvk.bias_residual_fp16(_p(B["vision_x"]), _p(B["vision_ffn_out"]),
                               W["vision_ffn_down_b"][i], seq, VIS_D, stream=stream)
    
    # ========== Encoder (FP16 Attention + FP8 FFN optional) ==========
    
    def transformer_encoder(self, stream: int = 0):
        """Run Gemma-2B encoder."""
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        vs = self.vision_seq
        es = self.encoder_seq_len
        
        fvk.layer_norm_fp16(_p(B["vision_x"]), W["vision_final_norm_w"],
                           W["vision_final_norm_b"], _p(B["vision_x_norm"]),
                           vs, VIS_D, 1e-5, stream=stream)
        
        gemm.fp16_nn(_p(B["vision_x_norm"]), W["encoder_multi_modal_projector_w"],
                    _p(B["encoder_x"]), vs, ENC_D, VIS_D, stream=stream)
        fvk.add_bias_fp16(_p(B["encoder_x"]), W["encoder_multi_modal_projector_b"],
                         vs, ENC_D, stream=stream)
        
        for i in range(self.enc_layers):
            self._encoder_layer(i, es, stream)
    
    def _encoder_layer(self, i: int, seq: int, stream: int):
        """Encoder layer: FP16 Attention + FP8/FP16 FFN."""
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        W8 = self.weights_fp8
        S8 = self.fp8_scales
        B = self.bufs
        
        # RMSNorm
        fvk.rms_norm_fp16(_p(B["encoder_x"]), _p(B["encoder_ones"]),
                         _p(B["encoder_x_norm"]), seq, ENC_D, 1e-6, stream=stream)
        
        # QKV GEMM (FP16)
        gemm.fp16_nn(_p(B["encoder_x_norm"]), W["encoder_attn_qkv_w"][i],
                    _p(B["encoder_QKV"]), seq, (ENC_NH + 2 * ENC_NKV) * ENC_HD, ENC_D, stream=stream)
        
        # Attention
        fvk.gpu_copy(self._encoder_qkv_tensor.data_ptr(), _p(B["encoder_QKV"]),
                    seq * (ENC_NH + 2 * ENC_NKV) * ENC_HD * 2, stream)
        
        Q = self._encoder_qkv_tensor[:, :ENC_NH * ENC_HD].view(seq, ENC_NH, ENC_HD)
        KV = self._encoder_qkv_tensor[:, ENC_NH * ENC_HD:].view(seq, 2, ENC_NKV, ENC_HD)
        K = KV[:, 0, :, :]
        V = KV[:, 1, :, :]
        
        fvk.gpu_copy(self._encoder_rope_tensor.data_ptr(), _p(B["encoder_rope"]),
                    seq * 256 * 2, stream)
        cos = self._encoder_rope_tensor[:, :128]
        sin = self._encoder_rope_tensor[:, 128:]
        
        Q_rope = self._apply_rope_interleaved(Q, cos, sin)
        K_rope = self._apply_rope_interleaved(K, cos, sin)
        
        K_cache_offset = i * seq * ENC_NKV * ENC_HD * 2
        V_cache_offset = i * seq * ENC_NKV * ENC_HD * 2
        fvk.gpu_copy(_p(B["encoder_K_cache"]) + K_cache_offset, K_rope.data_ptr(),
                    seq * ENC_NKV * ENC_HD * 2, stream)
        fvk.gpu_copy(_p(B["encoder_V_cache"]) + V_cache_offset, V.data_ptr(),
                    seq * ENC_NKV * ENC_HD * 2, stream)
        
        K_expanded = K_rope.expand(-1, ENC_NH, -1)
        V_expanded = V.expand(-1, ENC_NH, -1)
        
        Q_t = Q_rope.transpose(0, 1)
        K_t = K_expanded.transpose(0, 1)
        V_t = V_expanded.transpose(0, 1)
        
        attn_out = F.scaled_dot_product_attention(Q_t, K_t, V_t)
        attn_out = attn_out.transpose(0, 1).reshape(seq, ENC_D)
        
        fvk.gpu_copy(_p(B["encoder_attn_out"]), attn_out.data_ptr(), seq * ENC_D * 2, stream)
        
        # O projection (FP16)
        gemm.fp16_nn(_p(B["encoder_attn_out"]), W["encoder_attn_o_w"][i],
                    _p(B["encoder_x_norm"]), seq, ENC_D, ENC_D, stream=stream)
        fvk.residual_add_fp16(_p(B["encoder_x"]), _p(B["encoder_x_norm"]), seq * ENC_D, stream)
        
        # FFN - use FP8 if enabled, else FP16
        fvk.rms_norm_fp16(_p(B["encoder_x"]), _p(B["encoder_ones"]),
                         _p(B["encoder_x_norm"]), seq, ENC_D, 1e-6, stream=stream)
        
        if self.use_fp8_encoder_ffn and W8 and i < len(W8.get("encoder_ffn_gate_w", [])):
            self._encoder_ffn_fp8(i, seq, stream)
        else:
            self._encoder_ffn_fp16(i, seq, stream)
    
    def _encoder_ffn_fp8(self, i: int, seq: int, stream: int):
        """Encoder FFN using FP8 GEMM."""
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights  # FP16 weights for fallback
        W8 = self.weights_fp8
        S8 = self.fp8_scales
        B = self.bufs
        
        # Quantize activation
        frk.quantize_fp8_static_fp16(
            _p(B["encoder_x_norm"]), self.encoder_x_fp8.data_ptr(),
            self.scale_one.data_ptr(), seq * ENC_D, stream)
        
        # Gate (FP8)
        ret = frk.cutlass_ada_fp8_gemm_bf16_simple(
            self.encoder_x_fp8.data_ptr(), W8["encoder_ffn_gate_w"][i],
            _p(B["encoder_gate_merged"]), seq, ENC_H, ENC_D,
            1.0, S8["encoder_ffn_gate_w"][i], stream)
        
        if ret < 0:
            # Fallback to FP16
            gemm.fp16_nn(_p(B["encoder_x_norm"]), W["encoder_ffn_gate_w"][i],
                        _p(B["encoder_gate_merged"]), seq, ENC_H, ENC_D, stream=stream)
        
        # Up (FP8)
        ret = frk.cutlass_ada_fp8_gemm_bf16_simple(
            self.encoder_x_fp8.data_ptr(), W8["encoder_ffn_up_w"][i],
            _p(B["encoder_hidden"]), seq, ENC_H, ENC_D,
            1.0, S8["encoder_ffn_up_w"][i], stream)
        
        if ret < 0:
            gemm.fp16_nn(_p(B["encoder_x_norm"]), W["encoder_ffn_up_w"][i],
                        _p(B["encoder_hidden"]), seq, ENC_H, ENC_D, stream=stream)
        
        # GeGLU
        fvk.gate_geglu_fp16(_p(B["encoder_gate_merged"]), _p(B["encoder_hidden"]),
                           _p(B["encoder_hidden"]), seq * ENC_H, stream=stream)
        
        # Quantize hidden
        frk.quantize_fp8_static_fp16(
            _p(B["encoder_hidden"]), self.encoder_hidden_fp8.data_ptr(),
            self.scale_one.data_ptr(), seq * ENC_H, stream)
        
        # Down (FP8)
        ret = frk.cutlass_ada_fp8_gemm_bf16_simple(
            self.encoder_hidden_fp8.data_ptr(), W8["encoder_ffn_down_w"][i],
            _p(B["encoder_ffn_out"]), seq, ENC_D, ENC_H,
            1.0, S8["encoder_ffn_down_w"][i], stream)
        
        if ret < 0:
            gemm.fp16_nn(_p(B["encoder_hidden"]), W["encoder_ffn_down_w"][i],
                        _p(B["encoder_ffn_out"]), seq, ENC_D, ENC_H, stream=stream)
        
        fvk.residual_add_fp16(_p(B["encoder_x"]), _p(B["encoder_ffn_out"]), seq * ENC_D, stream)
    
    def _encoder_ffn_fp16(self, i: int, seq: int, stream: int):
        """Encoder FFN using FP16."""
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        
        gemm.fp16_nn(_p(B["encoder_x_norm"]), W["encoder_ffn_gate_w"][i],
                    _p(B["encoder_gate_merged"]), seq, ENC_H, ENC_D, stream=stream)
        gemm.fp16_nn(_p(B["encoder_x_norm"]), W["encoder_ffn_up_w"][i],
                    _p(B["encoder_hidden"]), seq, ENC_H, ENC_D, stream=stream)
        
        fvk.gate_geglu_fp16(_p(B["encoder_gate_merged"]), _p(B["encoder_hidden"]),
                           _p(B["encoder_hidden"]), seq * ENC_H, stream=stream)
        
        gemm.fp16_nn(_p(B["encoder_hidden"]), W["encoder_ffn_down_w"][i],
                    _p(B["encoder_ffn_out"]), seq, ENC_D, ENC_H, stream=stream)
        
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
    
    # ========== Decoder (FP16) ==========
    
    def transformer_decoder(self, stream: int = 0):
        """Run decoder diffusion (FP16)."""
        gemm = self.gemm
        fvk = self.fvk
        W = self.weights
        B = self.bufs
        es = self.encoder_seq_len
        sd = self.S_dec
        sa = self.chunk_size
        
        for step in range(self.num_steps):
            self._decoder_step(step, es, sd, stream)
    
    def _decoder_step(self, step: int, enc_seq: int, sd: int, stream: int):
        """Decoder step using FP16."""
        gemm = self.gemm
        fvk = self.fvk
        W = self.weights
        B = self.bufs
        sa = self.chunk_size
        
        time_emb_ptr = self._style_slice_ptr("decoder_time_emb", step)
        
        self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(_p(B["decoder_x"])),
            ctypes.c_void_p(time_emb_ptr),
            DEC_D * 2, 3, stream)
        
        x_action_ptr = _p(B["decoder_x"]) + DEC_D * 2
        gemm.fp16_nn(_p(B["diffusion_noise"]), W["decoder_action_in_proj_w"],
                    x_action_ptr, sa, DEC_D, ACTION_DIM, stream=stream)
        fvk.add_bias_fp16(x_action_ptr, W["decoder_action_in_proj_b"], sa, DEC_D, stream)
        
        time_emb_action_ptr = time_emb_ptr + DEC_D * 2
        fvk.residual_add_fp16(x_action_ptr, time_emb_action_ptr, sa * DEC_D, stream)
        
        for i in range(self.dec_layers):
            self._decoder_layer(i, step, enc_seq, sd, stream)
        
        style_final_ptr = self._style_slice_ptr("decoder_style_final", step)
        fvk.adarms_fp16(_p(B["decoder_x"]), style_final_ptr,
                       _p(B["x_normed_buf"]), _p(B["gate_buf"]),
                       sd, DEC_D, stream)
        
        x_out_ptr = _p(B["x_normed_buf"]) + DEC_D * 2
        gemm.fp16_nn(x_out_ptr, W["decoder_action_out_proj_w"],
                    _p(B["diffusion_noise"]), sa, ACTION_DIM, DEC_D, stream=stream)
        fvk.add_bias_fp16(_p(B["diffusion_noise"]), W["decoder_action_out_proj_b"],
                         sa, ACTION_DIM, stream)
    
    def _decoder_layer(self, i: int, step: int, enc_seq: int, sd: int, stream: int):
        """Decoder layer using FP16."""
        gemm = self.gemm
        fvk = self.fvk
        W = self.weights
        B = self.bufs
        
        style_attn_ptr = self._style_slice_ptr("decoder_style_attn", step, i)
        style_ffn_ptr = self._style_slice_ptr("decoder_style_ffn", step, i)
        
        fvk.adarms_fp16(_p(B["decoder_x"]), style_attn_ptr,
                       _p(B["x_normed_buf"]), _p(B["gate_buf"]),
                       sd, DEC_D, stream)
        
        gemm.fp16_nn(_p(B["x_normed_buf"]), W["decoder_attn_qkv_w"][i],
                    _p(B["decoder_QKV"]), sd, (DEC_NH + 2 * DEC_NKV) * DEC_HD, DEC_D, stream=stream)
        
        fvk.gpu_copy(self._dec_qkv_tensor.data_ptr(), _p(B["decoder_QKV"]),
                    sd * (DEC_NH + 2 * DEC_NKV) * DEC_HD * 2, stream)
        
        Q = self._dec_qkv_tensor[:, :DEC_NH * DEC_HD].view(sd, DEC_NH, DEC_HD)
        KV = self._dec_qkv_tensor[:, DEC_NH * DEC_HD:].view(sd, 2, DEC_NKV, DEC_HD)
        K_dec = KV[:, 0, :, :]
        V_dec = KV[:, 1, :, :]
        
        fvk.gpu_copy(self._dec_rope_tensor.data_ptr(), _p(B["decoder_rope"]), sd * 256 * 2, stream)
        Q_rope = self._apply_rope_interleaved(Q, self._dec_rope_cos, self._dec_rope_sin)
        K_dec_rope = self._apply_rope_interleaved(K_dec, self._dec_rope_cos, self._dec_rope_sin)
        
        enc_K_ptr = _p(B["encoder_K_cache"]) + i * enc_seq * ENC_NKV * ENC_HD * 2
        enc_V_ptr = _p(B["encoder_V_cache"]) + i * enc_seq * ENC_NKV * ENC_HD * 2
        
        fvk.gpu_copy(self._enc_K_tensor.data_ptr(), enc_K_ptr, enc_seq * ENC_NKV * ENC_HD * 2, stream)
        fvk.gpu_copy(self._enc_V_tensor.data_ptr(), enc_V_ptr, enc_seq * ENC_NKV * ENC_HD * 2, stream)
        
        enc_K_expanded = self._enc_K_tensor.expand(-1, DEC_NH, -1)
        enc_V_expanded = self._enc_V_tensor.expand(-1, DEC_NH, -1)
        
        Q_t = Q_rope.transpose(0, 1)
        K_t = enc_K_expanded.transpose(0, 1)
        V_t = enc_V_expanded.transpose(0, 1)
        
        attn_out = F.scaled_dot_product_attention(Q_t, K_t, V_t)
        attn_out = attn_out.transpose(0, 1).reshape(sd, DEC_NH * DEC_HD)
        
        fvk.gpu_copy(_p(B["decoder_attn_out"]), attn_out.data_ptr(), sd * DEC_NH * DEC_HD * 2, stream)
        
        gemm.fp16_nn(_p(B["decoder_attn_out"]), W["decoder_attn_o_w"][i],
                    _p(B["x_normed_buf"]), sd, DEC_D, DEC_NH * DEC_HD, stream=stream)
        
        fvk.gate_mul_residual(_p(B["decoder_x"]), _p(B["x_normed_buf"]),
                             _p(B["gate_buf"]), sd * DEC_D, stream)
        
        fvk.adarms_fp16(_p(B["decoder_x"]), style_ffn_ptr,
                       _p(B["x_normed_buf"]), _p(B["gate_buf"]),
                       sd, DEC_D, stream)
        
        gemm.fp16_nn(_p(B["x_normed_buf"]), W["decoder_ffn_gate_w"][i],
                    _p(B["decoder_gate_merged"]), sd, DEC_H, DEC_D, stream=stream)
        gemm.fp16_nn(_p(B["x_normed_buf"]), W["decoder_ffn_up_w"][i],
                    _p(B["decoder_hidden"]), sd, DEC_H, DEC_D, stream=stream)
        
        fvk.gate_geglu_fp16(_p(B["decoder_gate_merged"]), _p(B["decoder_hidden"]),
                           _p(B["decoder_hidden"]), sd * DEC_H, stream)
        
        gemm.fp16_nn(_p(B["decoder_hidden"]), W["decoder_ffn_down_w"][i],
                    _p(B["decoder_attn_out"]), sd, DEC_D, DEC_H, stream=stream)
        
        fvk.gate_mul_residual(_p(B["decoder_x"]), _p(B["decoder_attn_out"]),
                             _p(B["gate_buf"]), sd * DEC_D, stream)
    
    def run_pipeline(self, stream: int = 0, sync: bool = True):
        """Run full inference pipeline."""
        self._copy_lang_embeds_to_encoder_x(stream)
        self.vision_encoder(stream)
        self.transformer_encoder(stream)
        self.transformer_decoder(stream)
        if sync:
            self._cudart.cudaStreamSynchronize(ctypes.c_void_p(stream))
    
    def set_language_embeds(self, lang_embeds_np: np.ndarray):
        arr = np.ascontiguousarray(lang_embeds_np)
        self._lang_embeds_buf = CudaBuffer.from_numpy(arr)
    
    def _copy_lang_embeds_to_encoder_x(self, stream: int):
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