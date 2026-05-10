"""FlashRT — SM89 Pi0.5 inference pipeline with FP8 optimization for Vision and Encoder FFN.

Key optimizations:
- Vision FFN: FP8 GEMM (seq=512, 18x speedup)
- Encoder FFN: FP8 GEMM (seq=528, 1.57x speedup)
- Decoder: FP16 (seq=11 too small for FP8)

Performance target: <50ms (from 117ms baseline for v=2,s=10)
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
    return buf.ptr.value


class Pi05PipelineSm89Fp8VisionEncoder:
    """Pi0.5 pipeline with FP8 optimization for Vision and Encoder FFN."""
    
    def __init__(
        self,
        gemm,
        fvk,
        weights,
        weights_fp8=None,
        fp8_scales=None,
        *,
        num_views: int,
        max_prompt_len: int,
        chunk_size: int = CHUNK_SIZE_DEFAULT,
        num_steps: int = NUM_STEPS_DEFAULT,
    ):
        self.gemm = gemm
        self.fvk = fvk
        self.weights = weights
        self.weights_fp8 = weights_fp8 or {}
        self.fp8_scales = fp8_scales or {}
        
        self.num_views = int(num_views)
        self.max_prompt_len = int(max_prompt_len)
        self.chunk_size = int(chunk_size)
        self.S_dec = self.chunk_size + 1
        self.num_steps = int(num_steps)
        
        self.enc_layers = ENC_L
        self.dec_layers = DEC_L
        
        # Derived sizes
        self.vision_seq = self.num_views * VIS_SEQ_PER_VIEW
        encoder_seq_raw = self.vision_seq + self.max_prompt_len
        self.encoder_seq_padded = ((encoder_seq_raw + 15) // 16) * 16
        self.encoder_seq_len = self.encoder_seq_padded
        
        # FP8 support
        self.use_fp8_vision = bool(weights_fp8 and "vision_ffn_up" in weights_fp8)
        self.use_fp8_encoder = bool(weights_fp8 and "gate" in weights_fp8)
        
        if self.use_fp8_vision or self.use_fp8_encoder:
            # Pre-allocated FP8 activation buffers
            self.vision_x_fp8 = torch.empty(
                self.vision_seq, VIS_D,
                dtype=torch.float8_e4m3fn, device='cuda')
            self.vision_hidden_fp8 = torch.empty(
                self.vision_seq, VIS_H,
                dtype=torch.float8_e4m3fn, device='cuda')
            self.encoder_x_fp8 = torch.empty(
                self.encoder_seq_padded, ENC_D,
                dtype=torch.float8_e4m3fn, device='cuda')
            self.encoder_hidden_fp8 = torch.empty(
                self.encoder_seq_padded, ENC_H,
                dtype=torch.float8_e4m3fn, device='cuda')
            
            self.scale_one = torch.tensor([1.0], dtype=torch.float32, device='cuda')
            self.scale_one_ptr = self.scale_one.data_ptr()
        
        # Allocate buffers
        self.bufs = self._allocate_buffers()
        self._build_rope_table()
        self._cudart = ctypes.CDLL("libcudart.so")
        self._create_preallocated_tensors()
        
        logger.info(
            "Pi05PipelineSm89Fp8VisionEncoder (num_views=%d, vision_seq=%d, "
            "encoder_seq=%d, fp8_vision=%s, fp8_encoder=%s)",
            self.num_views, self.vision_seq, self.encoder_seq_len,
            self.use_fp8_vision, self.use_fp8_encoder)
    
    def _create_preallocated_tensors(self):
        sd = self.S_dec
        es = self.encoder_seq_len
        
        # Encoder scratch
        enc_qkv_dim = (ENC_NH + 2 * ENC_NKV) * ENC_HD
        self._enc_qkv_tensor = torch.empty(es, enc_qkv_dim, dtype=torch.float16, device='cuda')
        self._enc_rope_tensor = torch.empty(es, 256, dtype=torch.float16, device='cuda')
        self._enc_rope_cos = self._enc_rope_tensor[:, :128]
        self._enc_rope_sin = self._enc_rope_tensor[:, 128:]
        
        # Decoder scratch
        dec_qkv_dim = (DEC_NH + 2 * DEC_NKV) * DEC_HD
        self._dec_qkv_tensor = torch.empty(sd, dec_qkv_dim, dtype=torch.float16, device='cuda')
        self._dec_rope_tensor = torch.empty(sd, 256, dtype=torch.float16, device='cuda')
        self._dec_rope_cos = self._dec_rope_tensor[:, :128]
        self._dec_rope_sin = self._dec_rope_tensor[:, 128:]
        self._dec_enc_K_tensor = torch.empty(es, ENC_NKV, ENC_HD, dtype=torch.float16, device='cuda')
        self._dec_enc_V_tensor = torch.empty(es, ENC_NKV, ENC_HD, dtype=torch.float16, device='cuda')
    
    def _allocate_buffers(self) -> dict:
        nv = self.num_views
        vs = self.vision_seq
        es = self.encoder_seq_len
        sd = self.S_dec
        sa = self.chunk_size
        B = {}
        
        # Vision
        B["observation_images_normalized"] = CudaBuffer.device_empty(nv * 224 * 224 * 3, FP16)
        B["vision_x"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        B["vision_x_norm"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        B["vision_QKV"] = CudaBuffer.device_empty(vs * 3 * VIS_D, FP16)
        B["vision_hidden"] = CudaBuffer.device_empty(vs * VIS_H, FP16)
        B["vision_patches"] = CudaBuffer.device_empty(vs * VIS_PATCH_FLAT, FP16)
        B["vision_pos_embed_expanded"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        B["vision_attn_out"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        B["vision_ffn_out"] = CudaBuffer.device_empty(vs * VIS_D, FP16)
        
        # Encoder
        B["encoder_x"] = CudaBuffer.device_empty(es * ENC_D, FP16)
        B["encoder_x_norm"] = CudaBuffer.device_empty(es * ENC_D, FP16)
        B["encoder_QKV"] = CudaBuffer.device_empty(es * (ENC_NH + 2 * ENC_NKV) * ENC_HD, FP16)
        B["encoder_hidden"] = CudaBuffer.device_empty(es * ENC_H, FP16)
        B["encoder_gate_merged"] = CudaBuffer.device_empty(es * 2 * ENC_H, FP16)
        B["encoder_K_cache"] = CudaBuffer.device_empty(ENC_L * es * ENC_NKV * ENC_HD, FP16)
        B["encoder_V_cache"] = CudaBuffer.device_empty(ENC_L * es * ENC_NKV * ENC_HD, FP16)
        B["encoder_attn_out"] = CudaBuffer.device_empty(es * ENC_D, FP16)
        B["encoder_ffn_out"] = CudaBuffer.device_empty(es * ENC_D, FP16)
        
        B["encoder_ones"] = CudaBuffer.from_numpy(np.ones(ENC_D, dtype=FP16))
        B["decoder_ones"] = CudaBuffer.from_numpy(np.ones(DEC_D, dtype=FP16))
        
        # Decoder
        B["decoder_x"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        B["decoder_x_norm"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        B["decoder_QKV"] = CudaBuffer.device_empty(sd * (DEC_NH + 2 * DEC_NKV) * DEC_HD, FP16)
        B["decoder_hidden"] = CudaBuffer.device_empty(sd * DEC_H, FP16)
        B["decoder_gate_merged"] = CudaBuffer.device_empty(sd * 2 * DEC_H, FP16)
        B["decoder_attn_out"] = CudaBuffer.device_empty(sd * DEC_NH * DEC_HD, FP16)
        B["x_normed_buf"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        B["gate_buf"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        
        B["decoder_time_emb"] = CudaBuffer.device_empty(self.num_steps * sd * DEC_D, FP16)
        B["decoder_style_attn"] = CudaBuffer.device_empty(self.num_steps * DEC_L * sd * 3 * DEC_D, FP16)
        B["decoder_style_ffn"] = CudaBuffer.device_empty(self.num_steps * DEC_L * sd * 3 * DEC_D, FP16)
        B["decoder_style_final"] = CudaBuffer.device_empty(self.num_steps * sd * 3 * DEC_D, FP16)
        
        B["diffusion_noise"] = CudaBuffer.device_empty(sa * ACTION_DIM, FP16)
        return B
    
    def _build_rope_table(self):
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
    
    def _build_pos_embed_expanded(self):
        pos_src_ptr = self.weights["vision_position_embedding"]
        per_view_nbytes = VIS_SEQ_PER_VIEW * VIS_D * 2
        dst_buf = self.bufs["vision_pos_embed_expanded"]
        for v in range(self.num_views):
            self._cudart.cudaMemcpy(
                ctypes.c_void_p(dst_buf.ptr.value + v * per_view_nbytes),
                ctypes.c_void_p(pos_src_ptr), per_view_nbytes, 3)
        self._cudart.cudaDeviceSynchronize()
    
    def upload_precomputed_styles(self, styles: dict):
        B = self.bufs
        if "time_emb" in styles:
            B["decoder_time_emb"].upload(np.ascontiguousarray(styles["time_emb"]))
        if "style_attn" in styles:
            B["decoder_style_attn"].upload(np.ascontiguousarray(styles["style_attn"]))
        if "style_ffn" in styles:
            B["decoder_style_ffn"].upload(np.ascontiguousarray(styles["style_ffn"]))
        if "style_final" in styles:
            B["decoder_style_final"].upload(np.ascontiguousarray(styles["style_final"]))
    
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
    
    # Vision Encoder with FP8 FFN
    def vision_encoder(self, stream: int = 0):
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        W8 = self.weights_fp8
        S8 = self.fp8_scales
        B = self.bufs
        seq = self.vision_seq
        
        # Patch embedding (FP16 - small)
        fvk.patch_im2col(_p(B["observation_images_normalized"]), _p(B["vision_patches"]), self.num_views, stream)
        gemm.fp16_nn(_p(B["vision_patches"]), W["vision_patch_embedding_w"], _p(B["vision_x"]),
                     seq, VIS_D, VIS_PATCH_FLAT, stream=stream)
        fvk.bias_residual_fp16(_p(B["vision_x"]), _p(B["vision_pos_embed_expanded"]),
                               W["vision_patch_embedding_b"], seq, VIS_D, stream=stream)
        
        for i in range(VIS_L):
            self._vision_layer(i, stream)
    
    def _vision_layer(self, i: int, stream: int):
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        W8 = self.weights_fp8
        S8 = self.fp8_scales
        B = self.bufs
        seq = self.vision_seq
        
        # Pre-attention LayerNorm
        fvk.layer_norm_fp16(_p(B["vision_x"]), W["vision_pre_attn_norm_w"][i],
                            W["vision_pre_attn_norm_b"][i], _p(B["vision_x_norm"]),
                            seq, VIS_D, 1e-5, stream=stream)
        
        # QKV (FP16)
        gemm.fp16_nn(_p(B["vision_x_norm"]), W["vision_attn_qkv_w"][i],
                     _p(B["vision_QKV"]), seq, 3 * VIS_D, VIS_D, stream=stream)
        fvk.add_bias_fp16(_p(B["vision_QKV"]), W["vision_attn_qkv_b"][i], seq, 3 * VIS_D, stream=stream)
        
        # Attention
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
            attn_out[start:end] = attn_result.transpose(0, 1).reshape(VIS_SEQ_PER_VIEW, VIS_D)
        
        fvk.gpu_copy(_p(B["vision_attn_out"]), attn_out.data_ptr(), seq * VIS_D * 2, stream)
        
        # O projection (FP16)
        gemm.fp16_nn(_p(B["vision_attn_out"]), W["vision_attn_o_w"][i],
                     _p(B["vision_x_norm"]), seq, VIS_D, VIS_D, stream=stream)
        fvk.bias_residual_fp16(_p(B["vision_x"]), _p(B["vision_x_norm"]),
                               W["vision_attn_o_b"][i], seq, VIS_D, stream=stream)
        
        # Pre-FFN LayerNorm
        fvk.layer_norm_fp16(_p(B["vision_x"]), W["vision_pre_ffn_norm_w"][i],
                            W["vision_pre_ffn_norm_b"][i], _p(B["vision_x_norm"]),
                            seq, VIS_D, 1e-5, stream=stream)
        
        # FFN - use FP8 if weights available
        if self.use_fp8_vision and i in W8.get("vision_ffn_up", {}):
            # FP8 FFN Up
            fvk.quantize_fp8_static_fp16(_p(B["vision_x_norm"]), self.vision_x_fp8.data_ptr(),
                                         self.scale_one_ptr, seq * VIS_D, stream)
            frk.cutlass_ada_fp8_gemm_bf16_simple(
                self.vision_x_fp8.data_ptr(), W8["vision_ffn_up"][i],
                _p(B["vision_hidden"]), seq, VIS_H, VIS_D,
                1.0, S8["vision_ffn_up"][i], stream)
            
            fvk.add_bias_fp16(_p(B["vision_hidden"]), W["vision_ffn_up_b"][i], seq, VIS_H, stream=stream)
            fvk.gelu_inplace_fp16(_p(B["vision_hidden"]), seq * VIS_H, stream=stream)
            
            # FP8 FFN Down
            fvk.quantize_fp8_static_fp16(_p(B["vision_hidden"]), self.vision_hidden_fp8.data_ptr(),
                                         self.scale_one_ptr, seq * VIS_H, stream)
            frk.cutlass_ada_fp8_gemm_bf16_simple(
                self.vision_hidden_fp8.data_ptr(), W8["vision_ffn_down"][i],
                _p(B["vision_ffn_out"]), seq, VIS_D, VIS_H,
                1.0, S8["vision_ffn_down"][i], stream)
            
            fvk.bias_residual_fp16(_p(B["vision_x"]), _p(B["vision_ffn_out"]),
                                   W["vision_ffn_down_b"][i], seq, VIS_D, stream=stream)
        else:
            # FP16 FFN fallback
            gemm.fp16_nn(_p(B["vision_x_norm"]), W["vision_ffn_up_w"][i],
                         _p(B["vision_hidden"]), seq, VIS_H, VIS_D, stream=stream)
            fvk.add_bias_fp16(_p(B["vision_hidden"]), W["vision_ffn_up_b"][i], seq, VIS_H, stream=stream)
            fvk.gelu_inplace_fp16(_p(B["vision_hidden"]), seq * VIS_H, stream=stream)
            gemm.fp16_nn(_p(B["vision_hidden"]), W["vision_ffn_down_w"][i],
                         _p(B["vision_ffn_out"]), seq, VIS_D, VIS_H, stream=stream)
            fvk.bias_residual_fp16(_p(B["vision_x"]), _p(B["vision_ffn_out"]),
                                   W["vision_ffn_down_b"][i], seq, VIS_D, stream=stream)
    
    # Encoder with FP8 FFN
    def transformer_encoder(self, stream: int = 0):
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
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        W8 = self.weights_fp8
        S8 = self.fp8_scales
        B = self.bufs
        
        # RMSNorm
        fvk.rms_norm_fp16(_p(B["encoder_x"]), _p(B["encoder_ones"]),
                          _p(B["encoder_x_norm"]), seq, ENC_D, 1e-6, stream=stream)
        
        # QKV (FP16)
        gemm.fp16_nn(_p(B["encoder_x_norm"]), W["encoder_attn_qkv_w"][i],
                     _p(B["encoder_QKV"]), seq, (ENC_NH + 2 * ENC_NKV) * ENC_HD, ENC_D, stream=stream)
        
        # Attention with RoPE
        enc_qkv_dim = (ENC_NH + 2 * ENC_NKV) * ENC_HD
        fvk.gpu_copy(self._enc_qkv_tensor.data_ptr(), _p(B["encoder_QKV"]), seq * enc_qkv_dim * 2, stream)
        
        Q = self._enc_qkv_tensor[:, :ENC_NH * ENC_HD].view(seq, ENC_NH, ENC_HD)
        KV = self._enc_qkv_tensor[:, ENC_NH * ENC_HD:].view(seq, 2, ENC_NKV, ENC_HD)
        K = KV[:, 0, :, :]
        V = KV[:, 1, :, :]
        
        fvk.gpu_copy(self._enc_rope_tensor.data_ptr(), _p(B["encoder_rope"]), seq * 256 * 2, stream)
        Q_rope = self._apply_rope_interleaved(Q, self._enc_rope_cos, self._enc_rope_sin)
        K_rope = self._apply_rope_interleaved(K, self._enc_rope_cos, self._enc_rope_sin)
        
        K_cache_offset = i * seq * ENC_NKV * ENC_HD * 2
        V_cache_offset = i * seq * ENC_NKV * ENC_HD * 2
        fvk.gpu_copy(_p(B["encoder_K_cache"]) + K_cache_offset, K_rope.data_ptr(), seq * ENC_NKV * ENC_HD * 2, stream)
        fvk.gpu_copy(_p(B["encoder_V_cache"]) + V_cache_offset, V.data_ptr(), seq * ENC_NKV * ENC_HD * 2, stream)
        
        K_expanded = K_rope.expand(-1, ENC_NH, -1)
        V_expanded = V.expand(-1, ENC_NH, -1)
        
        attn_out = F.scaled_dot_product_attention(Q_rope.transpose(0, 1), K_expanded.transpose(0, 1), V_expanded.transpose(0, 1))
        attn_out = attn_out.transpose(0, 1).reshape(seq, ENC_D)
        fvk.gpu_copy(_p(B["encoder_attn_out"]), attn_out.data_ptr(), seq * ENC_D * 2, stream)
        
        gemm.fp16_nn(_p(B["encoder_attn_out"]), W["encoder_attn_o_w"][i],
                     _p(B["encoder_x_norm"]), seq, ENC_D, ENC_D, stream=stream)
        fvk.residual_add_fp16(_p(B["encoder_x"]), _p(B["encoder_x_norm"]), seq * ENC_D, stream)
        
        # FFN - FP8 if available
        fvk.rms_norm_fp16(_p(B["encoder_x"]), _p(B["encoder_ones"]),
                          _p(B["encoder_x_norm"]), seq, ENC_D, 1e-6, stream=stream)
        
        if self.use_fp8_encoder and i in W8.get("gate", {}):
            fvk.quantize_fp8_static_fp16(_p(B["encoder_x_norm"]), self.encoder_x_fp8.data_ptr(),
                                         self.scale_one_ptr, seq * ENC_D, stream)
            frk.cutlass_ada_fp8_gemm_bf16_simple(
                self.encoder_x_fp8.data_ptr(), W8["gate"][i], _p(B["encoder_gate_merged"]),
                seq, ENC_H, ENC_D, 1.0, S8["gate"][i], stream)
            frk.cutlass_ada_fp8_gemm_bf16_simple(
                self.encoder_x_fp8.data_ptr(), W8["up"][i], _p(B["encoder_hidden"]),
                seq, ENC_H, ENC_D, 1.0, S8["up"][i], stream)
            fvk.gate_geglu_fp16(_p(B["encoder_gate_merged"]), _p(B["encoder_hidden"]),
                                _p(B["encoder_hidden"]), seq * ENC_H, stream=stream)
            fvk.quantize_fp8_static_fp16(_p(B["encoder_hidden"]), self.encoder_hidden_fp8.data_ptr(),
                                         self.scale_one_ptr, seq * ENC_H, stream)
            frk.cutlass_ada_fp8_gemm_bf16_simple(
                self.encoder_hidden_fp8.data_ptr(), W8["down"][i], _p(B["encoder_ffn_out"]),
                seq, ENC_D, ENC_H, 1.0, S8["down"][i], stream)
            fvk.residual_add_fp16(_p(B["encoder_x"]), _p(B["encoder_ffn_out"]), seq * ENC_D, stream)
        else:
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
    
    # Decoder (FP16 only - seq=11 too small for FP8)
    def transformer_decoder(self, stream: int = 0):
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        es = self.encoder_seq_len
        sd = self.S_dec
        
        for step in range(self.num_steps):
            self._assemble_decoder_x(step, stream)
            for i in range(self.dec_layers):
                self._decoder_layer(i, step, es, sd, stream)
            
            style_final_ptr = self._style_slice_ptr("decoder_style_final", step)
            self.fvk.adarms_fp16(_p(B["decoder_x"]), style_final_ptr,
                                 _p(B["x_normed_buf"]), _p(B["gate_buf"]), sd, DEC_D, stream)
            
            x_out_ptr = _p(B["x_normed_buf"]) + DEC_D * 2
            gemm.fp16_nn(x_out_ptr, W["decoder_action_out_proj_w"],
                         _p(B["diffusion_noise"]), sd - 1, ACTION_DIM, DEC_D, stream=stream)
            self.fvk.add_bias_fp16(_p(B["diffusion_noise"]), W["decoder_action_out_proj_b"],
                                   sd - 1, ACTION_DIM, stream)
    
    def _assemble_decoder_x(self, step: int, stream: int):
        B = self.bufs
        W = self.weights
        gemm = self.gemm
        sa = self.chunk_size
        
        time_emb_ptr = self._style_slice_ptr("decoder_time_emb", step)
        self._cudart.cudaMemcpyAsync(ctypes.c_void_p(_p(B["decoder_x"])),
                                     ctypes.c_void_p(time_emb_ptr), DEC_D * 2, 3, stream)
        
        x_action_ptr = _p(B["decoder_x"]) + DEC_D * 2
        gemm.fp16_nn(_p(B["diffusion_noise"]), W["decoder_action_in_proj_w"],
                     x_action_ptr, sa, DEC_D, ACTION_DIM, stream=stream)
        self.fvk.add_bias_fp16(x_action_ptr, W["decoder_action_in_proj_b"], sa, DEC_D, stream)
        
        time_emb_action_ptr = time_emb_ptr + DEC_D * 2
        self.fvk.residual_add_fp16(x_action_ptr, time_emb_action_ptr, sa * DEC_D, stream)
    
    def _decoder_layer(self, i: int, step: int, enc_seq: int, sd: int, stream: int):
        fvk = self.fvk
        gemm = self.gemm
        W = self.weights
        B = self.bufs
        
        style_attn_ptr = self._style_slice_ptr("decoder_style_attn", step, i)
        style_ffn_ptr = self._style_slice_ptr("decoder_style_ffn", step, i)
        
        fvk.adarms_fp16(_p(B["decoder_x"]), style_attn_ptr,
                        _p(B["x_normed_buf"]), _p(B["gate_buf"]), sd, DEC_D, stream)
        
        dec_qkv_dim = (DEC_NH + 2 * DEC_NKV) * DEC_HD
        gemm.fp16_nn(_p(B["x_normed_buf"]), W["decoder_attn_qkv_w"][i],
                     _p(B["decoder_QKV"]), sd, dec_qkv_dim, DEC_D, stream=stream)
        
        qkv_tensor = self._dec_qkv_tensor
        fvk.gpu_copy(qkv_tensor.data_ptr(), _p(B["decoder_QKV"]), sd * dec_qkv_dim * 2, stream)
        
        Q = qkv_tensor[:, :DEC_NH * DEC_HD].view(sd, DEC_NH, DEC_HD)
        KV = qkv_tensor[:, DEC_NH * DEC_HD:].view(sd, 2, DEC_NKV, DEC_HD)
        K_dec = KV[:, 0, :, :]
        V_dec = KV[:, 1, :, :]
        
        fvk.gpu_copy(self._dec_rope_tensor.data_ptr(), _p(B["decoder_rope"]), sd * 256 * 2, stream)
        Q_rope = self._apply_rope_interleaved(Q, self._dec_rope_cos, self._dec_rope_sin)
        K_dec_rope = self._apply_rope_interleaved(K_dec, self._dec_rope_cos, self._dec_rope_sin)
        
        enc_K_ptr = _p(B["encoder_K_cache"]) + i * enc_seq * ENC_NKV * ENC_HD * 2
        enc_V_ptr = _p(B["encoder_V_cache"]) + i * enc_seq * ENC_NKV * ENC_HD * 2
        fvk.gpu_copy(self._dec_enc_K_tensor.data_ptr(), enc_K_ptr, enc_seq * ENC_NKV * ENC_HD * 2, stream)
        fvk.gpu_copy(self._dec_enc_V_tensor.data_ptr(), enc_V_ptr, enc_seq * ENC_NKV * ENC_HD * 2, stream)
        
        enc_K_expanded = self._dec_enc_K_tensor.expand(-1, DEC_NH, -1)
        enc_V_expanded = self._dec_enc_V_tensor.expand(-1, DEC_NH, -1)
        
        attn_out = F.scaled_dot_product_attention(Q_rope.transpose(0, 1),
                                                   enc_K_expanded.transpose(0, 1),
                                                   enc_V_expanded.transpose(0, 1))
        attn_out = attn_out.transpose(0, 1).reshape(sd, DEC_NH * DEC_HD)
        fvk.gpu_copy(_p(B["decoder_attn_out"]), attn_out.data_ptr(), sd * DEC_NH * DEC_HD * 2, stream)
        
        gemm.fp16_nn(_p(B["decoder_attn_out"]), W["decoder_attn_o_w"][i],
                     _p(B["x_normed_buf"]), sd, DEC_D, DEC_NH * DEC_HD, stream=stream)
        fvk.gate_mul_residual(_p(B["decoder_x"]), _p(B["x_normed_buf"]), _p(B["gate_buf"]), sd * DEC_D, stream)
        
        fvk.adarms_fp16(_p(B["decoder_x"]), style_ffn_ptr,
                        _p(B["x_normed_buf"]), _p(B["gate_buf"]), sd, DEC_D, stream)
        
        gemm.fp16_nn(_p(B["x_normed_buf"]), W["decoder_ffn_gate_w"][i],
                     _p(B["decoder_gate_merged"]), sd, DEC_H, DEC_D, stream=stream)
        gemm.fp16_nn(_p(B["x_normed_buf"]), W["decoder_ffn_up_w"][i],
                     _p(B["decoder_hidden"]), sd, DEC_H, DEC_D, stream=stream)
        fvk.gate_geglu_fp16(_p(B["decoder_gate_merged"]), _p(B["decoder_hidden"]),
                            _p(B["decoder_hidden"]), sd * DEC_H, stream=stream)
        gemm.fp16_nn(_p(B["decoder_hidden"]), W["decoder_ffn_down_w"][i],
                     _p(B["decoder_attn_out"]), sd, DEC_D, DEC_H, stream=stream)
        fvk.gate_mul_residual(_p(B["decoder_x"]), _p(B["decoder_attn_out"]), _p(B["gate_buf"]), sd * DEC_D, stream)
    
    def run_pipeline(self, stream: int = 0, sync: bool = True):
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
        self._cudart.cudaMemcpyAsync(ctypes.c_void_p(dst_ptr),
                                     self._lang_embeds_buf.ptr, self._lang_embeds_buf.nbytes, 3, stream)
    
    @property
    def input_images_buf(self):
        return self.bufs["observation_images_normalized"]
    
    @property
    def input_noise_buf(self):
        return self.bufs["diffusion_noise"]
    
    @property
    def output_noise_buf(self):
        return self.bufs["diffusion_noise"]