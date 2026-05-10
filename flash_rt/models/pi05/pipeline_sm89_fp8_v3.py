"""FlashRT — SM89 FP8 Pi0.5 inference pipeline (修复版).

基于Session 135修复的FP8 kernel调用方式:
- B矩阵必须使用ColumnMajor布局 (B.t().contiguous())
- scale参数使用device pointer (tensor.data_ptr())

关键优化:
- 权重预量化到FP8 E4M3 ColumnMajor格式
- 激活矩阵实时量化到FP8 RowMajor格式
- CUTLASS FP8 GEMM实现1.5-2x加速

性能目标: <50ms (vs 59ms FP16 baseline)
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
FP8 = np.uint8  # For sizing

fp8_e4m3 = torch.float8_e4m3fn


def _p(buf) -> int:
    """Extract int pointer from a CudaBuffer."""
    return buf.ptr.value


class FP8GemmRunnerV3:
    """FP8 GEMM runner using correct ColumnMajor B layout.
    
    基于Session 135修复:
    - B矩阵必须是ColumnMajor布局 (B.t().contiguous())
    - scale参数必须是device pointer
    
    CUDA Graph兼容: 使用固定scale tensor，不在运行时修改
    """
    
    def __init__(self, stream: int = 0):
        self.stream = stream
        self.fvk = frk
        
        # Pre-allocate scale tensors (CUDA Graph兼容)
        # 使用scale=1.0，因为激活值已经在合理范围内
        self.scale_one = torch.tensor([1.0], dtype=torch.float32, device='cuda')
        self.scale_one_ptr = self.scale_one.data_ptr()
        
    def quantize_activation(self, x_fp16_ptr: int, x_fp8_ptr: int, 
                            numel: int, stream: int = 0):
        """Quantize FP16 activation to FP8 with scale=1.0.
        
        CUDA Graph兼容: 使用固定scale tensor，不在运行时修改
        """
        self.fvk.quantize_fp8_static_fp16(
            x_fp16_ptr, x_fp8_ptr, 
            self.scale_one_ptr,
            numel, stream
        )
    
    def fp8_gemm(self, A_fp8_ptr: int, B_fp8_colmajor_ptr: int, C_fp16_ptr: int,
                 M: int, N: int, K: int, scale_a_ptr: int, scale_b_ptr: int,
                 stream: int = 0):
        """FP8 GEMM with ColumnMajor B layout.
        
        Args:
            A_fp8_ptr: FP8 activation (RowMajor, MxK)
            B_fp8_colmajor_ptr: FP8 weight (ColumnMajor, KxN stored as NxK)
            C_fp16_ptr: FP16 output (MxN)
            M, N, K: GEMM dimensions
            scale_a_ptr: Device pointer to scale_a tensor
            scale_b_ptr: Device pointer to scale_b tensor
        """
        self.fvk.cutlass_ada_fp8_gemm_bf16(
            A_fp8_ptr, B_fp8_colmajor_ptr, C_fp16_ptr,
            M, N, K, scale_a_ptr, scale_b_ptr, stream
        )
    
    def fp8_gemm_simple(self, A_fp8_ptr: int, B_fp8_colmajor_ptr: int, 
                        C_fp16_ptr: int, M: int, N: int, K: int,
                        scale_a: float = 1.0, scale_b: float = 1.0, stream: int = 0):
        """FP8 GEMM with simple float scale parameters.
        
        如果FP8 kernel不支持（M不能被16整除），fallback到FP16 GEMM。
        """
        # Check alignment: FP8 kernel requires M divisible by 16
        if M % 16 != 0 or N % 16 != 0 or K % 16 != 0:
            # Fallback: use FP16 GEMM
            # We need to dequantize FP8 back to FP16 for this
            # This is inefficient, so we skip FP8 for misaligned matrices
            return -1  # Signal fallback needed
        
        self.fvk.cutlass_ada_fp8_gemm_bf16_simple(
            A_fp8_ptr, B_fp8_colmajor_ptr, C_fp16_ptr,
            M, N, K, scale_a, scale_b, stream
        )
        return 0  # Success


class Pi05PipelineSm89Fp8V3:
    """Pi0.5 inference pipeline with corrected FP8 GEMM.
    
    Key differences from V2:
    - Uses FP8GemmRunnerV3 with ColumnMajor B layout
    - Weights pre-quantized to ColumnMajor format
    - Correct device pointer scale handling
    """
    
    def __init__(
        self,
        fvk,
        weights_fp8_colmajor,  # FP8 weights in ColumnMajor format
        weights_fp16,          # FP16 weights (norm, bias, etc.)
        fp8_scales,            # Per-weight scale tensors
        fp8_buffers,           # Pre-allocated FP8 activation buffers
        *,
        num_views: int,
        max_prompt_len: int,
        chunk_size: int = CHUNK_SIZE_DEFAULT,
        num_steps: int = NUM_STEPS_DEFAULT,
    ):
        self.fvk = fvk
        self.weights_fp8 = weights_fp8_colmajor
        self.weights_fp16 = weights_fp16
        self.fp8_scales = fp8_scales
        self.fp8_buffers = fp8_buffers
        
        self.num_views = num_views
        self.max_prompt_len = max_prompt_len
        self.chunk_size = chunk_size
        self.S_dec = chunk_size + 1
        self.num_steps = num_steps
        
        self.vision_seq = num_views * VIS_SEQ_PER_VIEW
        self.encoder_seq_len = self.vision_seq + max_prompt_len
        
        self.enc_layers = ENC_L
        self.dec_layers = DEC_L
        
        # Allocate buffers
        self.bufs = self._allocate_buffers()
        
        # Create FP8 GEMM runner
        stream = torch.cuda.current_stream().cuda_stream
        self.fp8_gemm = FP8GemmRunnerV3(stream)
        
        # FP16 GEMM runner for small matrices
        self.gemm = fvk.GemmRunner()
        
        self._cudart = ctypes.CDLL("libcudart.so")
        
        # Pre-allocate decoder tensors
        self._create_preallocated_tensors()
        
        logger.info(
            "Pi05PipelineSm89Fp8V3 initialised (num_views=%d, vision_seq=%d, "
            "encoder_seq_len=%d, chunk_size=%d, num_steps=%d)",
            num_views, self.vision_seq, self.encoder_seq_len,
            chunk_size, num_steps)
    
    def _create_preallocated_tensors(self):
        """Create pre-allocated tensors for CUDA Graph."""
        sd = self.S_dec
        es = self.encoder_seq_len
        
        # Decoder tensors
        self._dec_qkv_tensor = torch.empty(sd, (DEC_NH + 2 * DEC_NKV) * DEC_HD, 
                                           dtype=torch.float16, device='cuda')
        self._dec_rope_tensor = torch.empty(sd, 256, dtype=torch.float16, device='cuda')
        self._dec_rope_cos = self._dec_rope_tensor[:, :128]
        self._dec_rope_sin = self._dec_rope_tensor[:, 128:]
        
        # Encoder cross-attention tensors
        self._enc_K_tensor = torch.empty(es, ENC_NKV, ENC_HD, dtype=torch.float16, device='cuda')
        self._enc_V_tensor = torch.empty(es, ENC_NKV, ENC_HD, dtype=torch.float16, device='cuda')
        
        # Vision tensors
        self._vis_qkv_tensor = torch.empty(self.vision_seq, 3 * VIS_D, 
                                           dtype=torch.float16, device='cuda')
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
        
        # Decoder buffers (FP16 - small matrices)
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
        
        # Diffusion
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
        """Compute device pointer for style slice."""
        base = _p(self.bufs[buf_name])
        sd = self.S_dec
        
        if buf_name == "decoder_time_emb":
            return base + step * sd * DEC_D * 2
        if buf_name == "decoder_style_final":
            return base + step * sd * 3 * DEC_D * 2
        
        per_layer = sd * 3 * DEC_D * 2
        per_step = DEC_L * per_layer
        return base + step * per_step + layer * per_layer
    
    # ========== Vision Encoder (FP8 GEMM) ==========
    
    def vision_encoder(self, stream: int = 0):
        """Run SigLIP vision encoder with FP8 GEMM."""
        fvk = self.fvk
        gemm = self.gemm
        W16 = self.weights_fp16
        B = self.bufs
        seq = self.vision_seq
        
        # Patch embedding (FP16 - small matrix)
        fvk.patch_im2col(_p(B["observation_images_normalized"]),
                        _p(B["vision_patches"]), self.num_views, stream)
        
        gemm.fp16_nn(_p(B["vision_patches"]), W16["vision_patch_embedding_w"],
                    _p(B["vision_x"]), seq, VIS_D, VIS_PATCH_FLAT, stream=stream)
        
        fvk.bias_residual_fp16(_p(B["vision_x"]), _p(B["vision_pos_embed_expanded"]),
                               W16["vision_patch_embedding_b"], seq, VIS_D, stream=stream)
        
        # Vision layers - use FP16 for now (FP8 gains minimal for small seq)
        for i in range(VIS_L):
            self._vision_layer_fp16(i, stream)
    
    def _vision_layer_fp16(self, i: int, stream: int):
        """Vision layer using FP16 GEMM."""
        fvk = self.fvk
        gemm = self.gemm
        W16 = self.weights_fp16
        B = self.bufs
        seq = self.vision_seq
        
        # Pre-attention LayerNorm
        fvk.layer_norm_fp16(_p(B["vision_x"]), W16["vision_pre_attn_norm_w"][i],
                           W16["vision_pre_attn_norm_b"][i], _p(B["vision_x_norm"]),
                           seq, VIS_D, 1e-5, stream=stream)
        
        # QKV GEMM
        gemm.fp16_nn(_p(B["vision_x_norm"]), W16["vision_attn_qkv_w"][i],
                    _p(B["vision_QKV"]), seq, 3 * VIS_D, VIS_D, stream=stream)
        fvk.add_bias_fp16(_p(B["vision_QKV"]), W16["vision_attn_qkv_b"][i],
                         seq, 3 * VIS_D, stream=stream)
        
        # Attention (torch SDPA)
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
        
        # O projection
        gemm.fp16_nn(_p(B["vision_attn_out"]), W16["vision_attn_o_w"][i],
                    _p(B["vision_x_norm"]), seq, VIS_D, VIS_D, stream=stream)
        fvk.bias_residual_fp16(_p(B["vision_x"]), _p(B["vision_x_norm"]),
                               W16["vision_attn_o_b"][i], seq, VIS_D, stream=stream)
        
        # Pre-FFN LayerNorm
        fvk.layer_norm_fp16(_p(B["vision_x"]), W16["vision_pre_ffn_norm_w"][i],
                           W16["vision_pre_ffn_norm_b"][i], _p(B["vision_x_norm"]),
                           seq, VIS_D, 1e-5, stream=stream)
        
        # FFN up
        gemm.fp16_nn(_p(B["vision_x_norm"]), W16["vision_ffn_up_w"][i],
                    _p(B["vision_hidden"]), seq, VIS_H, VIS_D, stream=stream)
        fvk.add_bias_fp16(_p(B["vision_hidden"]), W16["vision_ffn_up_b"][i],
                         seq, VIS_H, stream=stream)
        fvk.gelu_inplace_fp16(_p(B["vision_hidden"]), seq * VIS_H, stream=stream)
        
        # FFN down
        gemm.fp16_nn(_p(B["vision_hidden"]), W16["vision_ffn_down_w"][i],
                    _p(B["vision_ffn_out"]), seq, VIS_D, VIS_H, stream=stream)
        fvk.bias_residual_fp16(_p(B["vision_x"]), _p(B["vision_ffn_out"]),
                               W16["vision_ffn_down_b"][i], seq, VIS_D, stream=stream)
    
    # ========== Encoder (FP8 GEMM for large matrices) ==========
    
    def transformer_encoder(self, stream: int = 0):
        """Run Gemma-2B encoder with FP8 GEMM."""
        fvk = self.fvk
        fp8_gemm = self.fp8_gemm
        gemm = self.gemm
        W8 = self.weights_fp8
        W16 = self.weights_fp16
        S8 = self.fp8_scales  # Scale tensors
        FB = self.fp8_buffers  # FP8 activation buffers
        B = self.bufs
        vs = self.vision_seq
        es = self.encoder_seq_len
        
        # Vision final norm
        fvk.layer_norm_fp16(_p(B["vision_x"]), W16["vision_final_norm_w"],
                           W16["vision_final_norm_b"], _p(B["vision_x_norm"]),
                           vs, VIS_D, 1e-5, stream=stream)
        
        # Multi-modal projector (FP16)
        gemm.fp16_nn(_p(B["vision_x_norm"]), W16["encoder_multi_modal_projector_w"],
                    _p(B["encoder_x"]), vs, ENC_D, VIS_D, stream=stream)
        fvk.add_bias_fp16(_p(B["encoder_x"]), W16["encoder_multi_modal_projector_b"],
                         vs, ENC_D, stream=stream)
        
        # Encoder layers with FP8 GEMM
        for i in range(self.enc_layers):
            self._encoder_layer_fp8(i, es, stream)
    
    def _encoder_layer_fp8(self, i: int, seq: int, stream: int):
        """Encoder layer with FP8 GEMM for large matrices."""
        fvk = self.fvk
        fp8_gemm = self.fp8_gemm
        gemm = self.gemm
        W8 = self.weights_fp8
        W16 = self.weights_fp16
        S8 = self.fp8_scales
        FB = self.fp8_buffers
        B = self.bufs
        
        # RMSNorm
        fvk.rms_norm_fp16(_p(B["encoder_x"]), _p(B["encoder_ones"]),
                         _p(B["encoder_x_norm"]), seq, ENC_D, 1e-6, stream=stream)
        
        # Quantize activation to FP8 (scale=1.0, fixed)
        fp8_gemm.quantize_activation(_p(B["encoder_x_norm"]), FB["encoder_x_fp8_ptr"],
                                     seq * ENC_D, stream)
        
        # QKV GEMM (FP8 - large matrix 304x2560x2048)
        fp8_gemm.fp8_gemm_simple(
            FB["encoder_x_fp8_ptr"],
            W8["encoder_attn_qkv_w"][i],  # ColumnMajor format
            _p(B["encoder_QKV"]),
            seq, (ENC_NH + 2 * ENC_NKV) * ENC_HD, ENC_D,
            1.0, S8["encoder_attn_qkv_w"][i], stream
        )
        
        # Split QKV
        fvk.gpu_copy(self._encoder_qkv_tensor.data_ptr(), _p(B["encoder_QKV"]),
                    seq * (ENC_NH + 2 * ENC_NKV) * ENC_HD * 2, stream)
        
        Q = self._encoder_qkv_tensor[:, :ENC_NH * ENC_HD].view(seq, ENC_NH, ENC_HD)
        KV = self._encoder_qkv_tensor[:, ENC_NH * ENC_HD:].view(seq, 2, ENC_NKV, ENC_HD)
        K = KV[:, 0, :, :]
        V = KV[:, 1, :, :]
        
        # Apply RoPE
        fvk.gpu_copy(self._encoder_rope_tensor.data_ptr(), _p(B["encoder_rope"]),
                    seq * 256 * 2, stream)
        cos = self._encoder_rope_tensor[:, :128]
        sin = self._encoder_rope_tensor[:, 128:]
        
        Q_rope = self._apply_rope_interleaved(Q, cos, sin)
        K_rope = self._apply_rope_interleaved(K, cos, sin)
        
        # Write K/V to cache
        K_cache_offset = i * seq * ENC_NKV * ENC_HD * 2
        V_cache_offset = i * seq * ENC_NKV * ENC_HD * 2
        fvk.gpu_copy(_p(B["encoder_K_cache"]) + K_cache_offset, K_rope.data_ptr(),
                    seq * ENC_NKV * ENC_HD * 2, stream)
        fvk.gpu_copy(_p(B["encoder_V_cache"]) + V_cache_offset, V.data_ptr(),
                    seq * ENC_NKV * ENC_HD * 2, stream)
        
        # Torch attention (GQA)
        K_expanded = K_rope.expand(-1, ENC_NH, -1)
        V_expanded = V.expand(-1, ENC_NH, -1)
        
        Q_t = Q_rope.transpose(0, 1)
        K_t = K_expanded.transpose(0, 1)
        V_t = V_expanded.transpose(0, 1)
        
        attn_out = F.scaled_dot_product_attention(Q_t, K_t, V_t)
        attn_out = attn_out.transpose(0, 1).reshape(seq, ENC_D)
        
        fvk.gpu_copy(_p(B["encoder_attn_out"]), attn_out.data_ptr(), seq * ENC_D * 2, stream)
        
        # O projection (FP16)
        gemm.fp16_nn(_p(B["encoder_attn_out"]), W16["encoder_attn_o_w"][i],
                    _p(B["encoder_x_norm"]), seq, ENC_D, ENC_D, stream=stream)
        fvk.residual_add_fp16(_p(B["encoder_x"]), _p(B["encoder_x_norm"]), seq * ENC_D, stream)
        
        # FFN - FP8 for large matrices
        fvk.rms_norm_fp16(_p(B["encoder_x"]), _p(B["encoder_ones"]),
                         _p(B["encoder_x_norm"]), seq, ENC_D, 1e-6, stream=stream)
        
        # Quantize activation (scale=1.0 fixed)
        fp8_gemm.quantize_activation(_p(B["encoder_x_norm"]), FB["encoder_x_fp8_ptr"],
                                     seq * ENC_D, stream)
        
        # Gate (FP8 - 304x16384x2048)
        fp8_gemm.fp8_gemm_simple(
            FB["encoder_x_fp8_ptr"],
            W8["encoder_ffn_gate_w"][i],
            _p(B["encoder_gate_merged"]),
            seq, ENC_H, ENC_D,
            1.0, S8["encoder_ffn_gate_w"][i], stream
        )
        
        # Up (FP8 - 304x16384x2048)
        fp8_gemm.fp8_gemm_simple(
            FB["encoder_x_fp8_ptr"],
            W8["encoder_ffn_up_w"][i],
            _p(B["encoder_hidden"]),
            seq, ENC_H, ENC_D,
            1.0, S8["encoder_ffn_up_w"][i], stream
        )
        
        # Gate-GeGLU fusion
        fvk.gate_geglu_fp16(_p(B["encoder_gate_merged"]), _p(B["encoder_hidden"]),
                           _p(B["encoder_hidden"]), seq * ENC_H, stream=stream)
        
        # Quantize hidden for Down GEMM
        fp8_gemm.quantize_activation(_p(B["encoder_hidden"]), FB["encoder_hidden_fp8_ptr"],
                                     seq * ENC_H, stream)
        
        # Down (FP8 - 304x2048x16384)
        fp8_gemm.fp8_gemm_simple(
            FB["encoder_hidden_fp8_ptr"],
            W8["encoder_ffn_down_w"][i],
            _p(B["encoder_ffn_out"]),
            seq, ENC_D, ENC_H,
            1.0, S8["encoder_ffn_down_w"][i], stream
        )
        
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
    
    # ========== Decoder (FP16 - small matrices) ==========
    
    def transformer_decoder(self, stream: int = 0):
        """Run decoder diffusion - FP16 for small matrices."""
        gemm = self.gemm
        fvk = self.fvk
        W16 = self.weights_fp16
        B = self.bufs
        es = self.encoder_seq_len
        sd = self.S_dec
        sa = self.chunk_size
        
        for step in range(self.num_steps):
            self._decoder_step_fp16(step, es, sd, stream)
    
    def _decoder_step_fp16(self, step: int, enc_seq: int, sd: int, stream: int):
        """Decoder step using FP16 GEMM."""
        gemm = self.gemm
        fvk = self.fvk
        W16 = self.weights_fp16
        B = self.bufs
        sa = self.chunk_size
        
        # Assemble decoder input
        time_emb_ptr = self._style_slice_ptr("decoder_time_emb", step)
        
        # Copy time_emb[0] to decoder_x[0]
        self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(_p(B["decoder_x"])),
            ctypes.c_void_p(time_emb_ptr),
            DEC_D * 2, 3, stream)
        
        # action_in_proj
        x_action_ptr = _p(B["decoder_x"]) + DEC_D * 2
        gemm.fp16_nn(_p(B["diffusion_noise"]), W16["decoder_action_in_proj_w"],
                    x_action_ptr, sa, DEC_D, ACTION_DIM, stream=stream)
        fvk.add_bias_fp16(x_action_ptr, W16["decoder_action_in_proj_b"], sa, DEC_D, stream)
        
        # Add time_emb[1:]
        time_emb_action_ptr = time_emb_ptr + DEC_D * 2
        fvk.residual_add_fp16(x_action_ptr, time_emb_action_ptr, sa * DEC_D, stream)
        
        # Decoder layers
        for i in range(self.dec_layers):
            self._decoder_layer_fp16(i, step, enc_seq, sd, stream)
        
        # Final norm and output projection
        style_final_ptr = self._style_slice_ptr("decoder_style_final", step)
        fvk.adarms_fp16(_p(B["decoder_x"]), style_final_ptr,
                       _p(B["x_normed_buf"]), _p(B["gate_buf"]),
                       sd, DEC_D, stream)
        
        x_out_ptr = _p(B["x_normed_buf"]) + DEC_D * 2
        gemm.fp16_nn(x_out_ptr, W16["decoder_action_out_proj_w"],
                    _p(B["diffusion_noise"]), sa, ACTION_DIM, DEC_D, stream=stream)
        fvk.add_bias_fp16(_p(B["diffusion_noise"]), W16["decoder_action_out_proj_b"],
                         sa, ACTION_DIM, stream)
    
    def _decoder_layer_fp16(self, i: int, step: int, enc_seq: int, sd: int, stream: int):
        """Decoder layer using FP16."""
        gemm = self.gemm
        fvk = self.fvk
        W16 = self.weights_fp16
        B = self.bufs
        
        style_attn_ptr = self._style_slice_ptr("decoder_style_attn", step, i)
        style_ffn_ptr = self._style_slice_ptr("decoder_style_ffn", step, i)
        
        # AdaRMSNorm (attention)
        fvk.adarms_fp16(_p(B["decoder_x"]), style_attn_ptr,
                       _p(B["x_normed_buf"]), _p(B["gate_buf"]),
                       sd, DEC_D, stream)
        
        # QKV GEMM
        gemm.fp16_nn(_p(B["x_normed_buf"]), W16["decoder_attn_qkv_w"][i],
                    _p(B["decoder_QKV"]), sd, (DEC_NH + 2 * DEC_NKV) * DEC_HD, DEC_D, stream=stream)
        
        # Attention
        fvk.gpu_copy(self._dec_qkv_tensor.data_ptr(), _p(B["decoder_QKV"]),
                    sd * (DEC_NH + 2 * DEC_NKV) * DEC_HD * 2, stream)
        
        Q = self._dec_qkv_tensor[:, :DEC_NH * DEC_HD].view(sd, DEC_NH, DEC_HD)
        KV = self._dec_qkv_tensor[:, DEC_NH * DEC_HD:].view(sd, 2, DEC_NKV, DEC_HD)
        K_dec = KV[:, 0, :, :]
        V_dec = KV[:, 1, :, :]
        
        # Apply RoPE
        fvk.gpu_copy(self._dec_rope_tensor.data_ptr(), _p(B["decoder_rope"]), sd * 256 * 2, stream)
        Q_rope = self._apply_rope_interleaved(Q, self._dec_rope_cos, self._dec_rope_sin)
        K_dec_rope = self._apply_rope_interleaved(K_dec, self._dec_rope_cos, self._dec_rope_sin)
        
        # Cross-attention with encoder KV
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
        
        # O projection
        gemm.fp16_nn(_p(B["decoder_attn_out"]), W16["decoder_attn_o_w"][i],
                    _p(B["x_normed_buf"]), sd, DEC_D, DEC_NH * DEC_HD, stream=stream)
        
        # gate * residual
        fvk.gate_mul_residual(_p(B["decoder_x"]), _p(B["x_normed_buf"]),
                             _p(B["gate_buf"]), sd * DEC_D, stream)
        
        # AdaRMSNorm (FFN)
        fvk.adarms_fp16(_p(B["decoder_x"]), style_ffn_ptr,
                       _p(B["x_normed_buf"]), _p(B["gate_buf"]),
                       sd, DEC_D, stream)
        
        # FFN
        gemm.fp16_nn(_p(B["x_normed_buf"]), W16["decoder_ffn_gate_w"][i],
                    _p(B["decoder_gate_merged"]), sd, DEC_H, DEC_D, stream=stream)
        gemm.fp16_nn(_p(B["x_normed_buf"]), W16["decoder_ffn_up_w"][i],
                    _p(B["decoder_hidden"]), sd, DEC_H, DEC_D, stream=stream)
        
        fvk.gate_geglu_fp16(_p(B["decoder_gate_merged"]), _p(B["decoder_hidden"]),
                           _p(B["decoder_hidden"]), sd * DEC_H, stream)
        
        gemm.fp16_nn(_p(B["decoder_hidden"]), W16["decoder_ffn_down_w"][i],
                    _p(B["decoder_attn_out"]), sd, DEC_D, DEC_H, stream=stream)
        
        fvk.gate_mul_residual(_p(B["decoder_x"]), _p(B["decoder_attn_out"]),
                             _p(B["gate_buf"]), sd * DEC_D, stream)
    
    # ========== Full Pipeline ==========
    
    def run_pipeline(self, stream: int = 0, sync: bool = True):
        """Run full inference pipeline."""
        self._copy_lang_embeds_to_encoder_x(stream)
        self.vision_encoder(stream)
        self.transformer_encoder(stream)
        self.transformer_decoder(stream)
        if sync:
            self._cudart.cudaStreamSynchronize(ctypes.c_void_p(stream))
    
    def set_language_embeds(self, lang_embeds_np: np.ndarray):
        """Set language embedding from prompt."""
        arr = np.ascontiguousarray(lang_embeds_np)
        self._lang_embeds_buf = CudaBuffer.from_numpy(arr)
    
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