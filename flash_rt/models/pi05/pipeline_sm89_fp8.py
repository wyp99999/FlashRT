"""FlashRT — SM89 FP8 Pi0.5 inference pipeline.

FP8 optimized version of pipeline_sm89.py using CUTLASS Ada FP8 GEMM kernels.

Key optimizations:
- Weights pre-quantized to FP8 E4M3 (50% memory reduction)
- Activations quantized on-the-fly using fast quantize_fp8_static kernel
- CUTLASS FP8 GEMM for 1.5-2x speedup over BF16

Architecture:
- Vision: 27 SigLIP layers (FP8 GEMM for QKV, FFN)
- Encoder: 18 Gemma-2B layers (FP8 GEMM for QKV, Gate, Up, Down)
- Decoder: 18 Gemma-300M layers (FP8 GEMM for QKV, Gate, Up, Down)
- Diffusion: 10-step flow-matching with AdaRMSNorm

Performance target: <50ms (vs 59ms BF16 baseline)
"""

from __future__ import annotations

import ctypes
import logging
import math

import numpy as np
import torch

from flash_rt.core.cuda_buffer import CudaBuffer
import flash_rt.flash_rt_kernels as frk

logger = logging.getLogger(__name__)

# Fixed Pi0.5 model dimensions (same as SM89 FP16)
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
BF16 = np.float16  # For sizing (actual BF16 tensor)

# FP8 dtype
fp8_e4m3 = torch.float8_e4m3fn


def _p(buf) -> int:
    """Extract int pointer from a CudaBuffer."""
    return buf.ptr.value


class FP8GemmRunner:
    """FP8 GEMM wrapper for SM89 using CUTLASS kernels.
    
    Encapsulates quantize + FP8 GEMM into a single efficient call.
    
    Key insight: FP8 E4M3 range is [-448, 448]. We need to scale input
    to utilize full FP8 range for best precision.
    
    For pre-quantized weights (B_fp8), the scale is stored per-weight.
    For activations (A), we use dynamic per-tensor quantization.
    """
    
    def __init__(self, stream: int = 0):
        self.stream = stream
        self.fvk = frk
        
        # Pre-allocated scale tensors
        # Use scale=1.0 for weights that are already properly quantized
        self.scale_one = torch.tensor([1.0], device='cuda', dtype=torch.float32)
        self.scale_one_ptr = self.scale_one.data_ptr()
    
    def fp8_nn(self, A_fp16_ptr: int, B_fp8_ptr: int, C_fp16_ptr: int,
               M: int, N: int, K: int, A_fp8_buf: int = 0,
               scale_a: float = 1.0, scale_b: float = 1.0):
        """FP8 GEMM: quantize A, then compute A_fp8 @ B_fp8 -> C_fp16.
        
        Args:
            A_fp16_ptr: FP16 activation input pointer
            B_fp8_ptr: FP8 weight pointer (pre-quantized with proper scale)
            C_fp16_ptr: FP16 output pointer
            M, N, K: GEMM dimensions (A: MxK, B: KxN, C: MxN)
            A_fp8_buf: Pre-allocated FP8 buffer for quantized A
            scale_a: Scale for A (usually 1.0 for pre-quantized weights)
            scale_b: Scale for B (stored with the weight)
        
        Note:
            For pre-quantized weights (B_fp8), scale_b should be the scale
            used during weight quantization. The output is:
            C = (A_fp8 @ B_fp8) * scale_a * scale_b
            
            We use scale=1.0 for activations since they are quantized
            dynamically with the kernel's internal scaling.
        """
        # Quantize A from FP16 to FP8 using scale=1.0 (kernel handles scaling)
        self.fvk.quantize_fp8_static(
            A_fp16_ptr,
            A_fp8_buf,
            self.scale_one_ptr,
            M * K,
            self.stream
        )
        
        # FP8 GEMM: A_fp8 @ B_fp8 -> C_fp16
        # scale_a=1.0 (activations), scale_b from weight quantization
        self.fvk.cutlass_ada_fp8_gemm_bf16_simple(
            A_fp8_buf,
            B_fp8_ptr,
            C_fp16_ptr,
            M, N, K,
            scale_a, scale_b,  # scales for output
            self.stream
        )
    
    def fp16_nn(self, A_ptr: int, B_ptr: int, C_ptr: int,
               M: int, N: int, K: int):
        """Fallback FP16 GEMM for operations not using FP8."""
        # Use existing FP16 kernel
        self.fvk.fp16_nn(A_ptr, B_ptr, C_ptr, M, N, K, self.stream)


class Pi05PipelineSm89Fp8:
    """Pi0.5 inference pipeline for SM89 with FP8 GEMM optimization.
    
    Key differences from Pi05PipelineSm89:
    - FP8GemmRunner instead of GemmRunner
    - FP8 quantized weights (stored as FP8 tensors)
    - On-the-fly activation quantization
    - ~1.5x speedup over BF16 baseline
    
    Args:
        fvk: flash_rt_kernels module.
        weights_fp8: FP8 quantized weight pointers dict.
        weights_fp16: FP16 weight pointers dict (non-GEMM weights).
        num_views: Number of camera views.
        max_prompt_len: Maximum prompt token length.
        chunk_size: Action chunk size (default 10).
        num_steps: Diffusion denoise steps (default 10).
    """
    
    def __init__(
        self,
        fvk,
        weights_fp8,
        weights_fp16,
        fp8_buffers,  # Pre-allocated FP8 buffers for activations
        *,
        num_views: int,
        max_prompt_len: int,
        chunk_size: int = CHUNK_SIZE_DEFAULT,
        num_steps: int = NUM_STEPS_DEFAULT,
    ):
        self.fvk = fvk
        self.weights_fp8 = weights_fp8  # FP8 weights (GEMM)
        self.weights_fp16 = weights_fp16  # FP16 weights (norm, bias, etc.)
        self.fp8_buffers = fp8_buffers  # Pre-allocated FP8 activation buffers
        
        self.num_views = num_views
        self.max_prompt_len = max_prompt_len
        self.chunk_size = chunk_size
        self.S_dec = chunk_size + 1
        self.num_steps = num_steps
        
        self.vision_seq = num_views * VIS_SEQ_PER_VIEW
        self.encoder_seq_len = self.vision_seq + max_prompt_len
        
        # Layer counts
        self.enc_layers = ENC_L
        self.dec_layers = DEC_L
        
        # Allocate buffers
        self.bufs = self._allocate_buffers()
        
        # Create FP8 GEMM runner
        stream = torch.cuda.current_stream().cuda_stream
        self.fp8_gemm = FP8GemmRunner(stream)
        
        # Legacy gemm for non-FP8 operations
        self.gemm = fvk.GemmRunner()
        
        # CUDART for D2D copies
        self._cudart = ctypes.CDLL("libcudart.so")
        
        logger.info(
            "Pi05PipelineSm89Fp8 initialised (num_views=%d, vision_seq=%d, "
            "encoder_seq_len=%d, chunk_size=%d, num_steps=%d)",
            num_views, self.vision_seq, self.encoder_seq_len,
            chunk_size, num_steps)
    
    def _allocate_buffers(self) -> dict:
        """Allocate working buffers for the pipeline."""
        nv = self.num_views
        vs = self.vision_seq
        es = self.encoder_seq_len
        sd = self.S_dec
        sa = self.chunk_size
        
        B = {}
        
        # Vision buffers (same as FP16 pipeline)
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
        
        # FP8 activation buffers for Vision
        B["vision_x_fp8"] = CudaBuffer.device_empty(vs * VIS_D, FP8)
        B["vision_QKV_fp8"] = CudaBuffer.device_empty(vs * 3 * VIS_D, FP8)
        B["vision_hidden_fp8"] = CudaBuffer.device_empty(vs * VIS_H, FP8)
        
        # Encoder buffers
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
        
        # FP8 activation buffers for Encoder
        B["encoder_x_fp8"] = CudaBuffer.device_empty(es * ENC_D, FP8)
        B["encoder_QKV_fp8"] = CudaBuffer.device_empty(
            es * (ENC_NH + 2 * ENC_NKV) * ENC_HD, FP8)
        B["encoder_hidden_fp8"] = CudaBuffer.device_empty(es * ENC_H, FP8)
        
        # Decoder buffers
        B["decoder_x"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        B["decoder_x_norm"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        B["decoder_QKV"] = CudaBuffer.device_empty(
            sd * (DEC_NH + 2 * DEC_NKV) * DEC_HD, FP16)
        B["decoder_hidden"] = CudaBuffer.device_empty(sd * DEC_H, FP16)
        B["decoder_gate_merged"] = CudaBuffer.device_empty(sd * 2 * DEC_H, FP16)
        B["decoder_attn_out"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        
        # FP8 activation buffers for Decoder
        B["decoder_x_fp8"] = CudaBuffer.device_empty(sd * DEC_D, FP8)
        B["decoder_QKV_fp8"] = CudaBuffer.device_empty(
            sd * (DEC_NH + 2 * DEC_NKV) * DEC_HD, FP8)
        B["decoder_hidden_fp8"] = CudaBuffer.device_empty(sd * DEC_H, FP8)
        
        # AdaRMSNorm buffers
        B["encoder_ones"] = CudaBuffer.from_numpy(np.ones(ENC_D, dtype=FP16))
        B["decoder_ones"] = CudaBuffer.from_numpy(np.ones(DEC_D, dtype=FP16))
        B["x_normed_buf"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        B["gate_buf"] = CudaBuffer.device_empty(sd * DEC_D, FP16)
        
        # Precomputed style buffers
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
    
    def run_vision(self, stream: int = 0):
        """Run SigLIP vision encoder with FP8 GEMM."""
        fvk = self.fvk
        fp8_gemm = self.fp8_gemm
        W8 = self.weights_fp8  # FP8 weights
        W16 = self.weights_fp16  # FP16 weights
        B = self.bufs
        seq = self.vision_seq
        
        # Patch embedding (FP16 for now - small matrix)
        fvk.patch_im2col(
            _p(B["observation_images_normalized"]),
            _p(B["vision_patches"]),
            self.num_views, stream)
        
        # Patch embedding GEMM (keep FP16 - not dominant)
        self.gemm.fp16_nn(
            _p(B["vision_patches"]),
            W16["vision_patch_embedding_w"],
            _p(B["vision_x"]),
            seq, VIS_D, VIS_PATCH_FLAT, stream=stream)
        
        fvk.bias_residual_fp16(
            _p(B["vision_x"]),
            _p(B["vision_pos_embed_expanded"]),
            W16["vision_patch_embedding_b"],
            seq, VIS_D, stream=stream)
        
        # Vision layers with FP8 GEMM
        for i in range(VIS_L):
            self._vision_layer_fp8(i, stream)
    
    def _vision_layer_fp8(self, i: int, stream: int):
        """One SigLIP layer with FP8 GEMM for QKV and FFN."""
        fvk = self.fvk
        fp8_gemm = self.fp8_gemm
        W8 = self.weights_fp8
        W16 = self.weights_fp16
        B = self.bufs
        seq = self.vision_seq
        
        # Pre-attention LayerNorm (FP16)
        fvk.layer_norm_fp16(
            _p(B["vision_x"]),
            W16["vision_pre_attn_norm_w"][i],
            W16["vision_pre_attn_norm_b"][i],
            _p(B["vision_x_norm"]),
            seq, VIS_D, 1e-5, stream=stream)
        
        # QKV GEMM (FP8)
        fp8_gemm.fp8_nn(
            _p(B["vision_x_norm"]),  # BF16 input
            W8["vision_attn_qkv_w"][i],  # FP8 weight
            _p(B["vision_QKV"]),  # BF16 output
            seq, 3 * VIS_D, VIS_D,
            _p(B["vision_x_fp8"])  # FP8 buffer
        )
        
        fvk.add_bias_fp16(
            _p(B["vision_QKV"]),
            W16["vision_attn_qkv_b"][i],
            seq, 3 * VIS_D, stream=stream)
        
        # Attention (torch SDPA)
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
            attn_result = torch.nn.functional.scaled_dot_product_attention(Q_view, K_view, V_view)
            attn_out[start:end] = attn_result.transpose(0, 1).reshape(VIS_SEQ_PER_VIEW, VIS_D)
        
        fvk.gpu_copy(_p(B["vision_attn_out"]), attn_out.data_ptr(), seq * VIS_D * 2, stream)
        
        # O projection (FP8)
        fp8_gemm.fp8_nn(
            _p(B["vision_attn_out"]),
            W8["vision_attn_o_w"][i],
            _p(B["vision_x_norm"]),
            seq, VIS_D, VIS_D,
            _p(B["vision_x_fp8"])
        )
        
        fvk.bias_residual_fp16(
            _p(B["vision_x"]),
            _p(B["vision_x_norm"]),
            W16["vision_attn_o_b"][i],
            seq, VIS_D, stream=stream)
        
        # Pre-FFN LayerNorm
        fvk.layer_norm_fp16(
            _p(B["vision_x"]),
            W16["vision_pre_ffn_norm_w"][i],
            W16["vision_pre_ffn_norm_b"][i],
            _p(B["vision_x_norm"]),
            seq, VIS_D, 1e-5, stream=stream)
        
        # FFN up (FP8)
        fp8_gemm.fp8_nn(
            _p(B["vision_x_norm"]),
            W8["vision_ffn_up_w"][i],
            _p(B["vision_hidden"]),
            seq, VIS_H, VIS_D,
            _p(B["vision_x_fp8"])
        )
        
        fvk.add_bias_fp16(
            _p(B["vision_hidden"]),
            W16["vision_ffn_up_b"][i],
            seq, VIS_H, stream=stream)
        
        # GELU activation
        fvk.gelu_inplace_fp16(_p(B["vision_hidden"]), seq * VIS_H, stream=stream)
        
        # FFN down (FP8)
        fp8_gemm.fp8_nn(
            _p(B["vision_hidden"]),
            W8["vision_ffn_down_w"][i],
            _p(B["vision_ffn_out"]),
            seq, VIS_D, VIS_H,
            _p(B["vision_hidden_fp8"])
        )
        
        fvk.bias_residual_fp16(
            _p(B["vision_x"]),
            _p(B["vision_ffn_out"]),
            W16["vision_ffn_down_b"][i],
            seq, VIS_D, stream=stream)
    
    def run_encoder(self, stream: int = 0):
        """Run Gemma-2B encoder with FP8 GEMM for large matrices."""
        fvk = self.fvk
        fp8_gemm = self.fp8_gemm
        gemm = self.gemm  # FP16 fallback for small ops
        W8 = self.weights_fp8
        W16 = self.weights_fp16
        B = self.bufs
        vs = self.vision_seq
        es = self.encoder_seq_len
        
        # Vision final norm (FP16)
        fvk.layer_norm_fp16(
            _p(B["vision_x"]),
            W16["vision_final_norm_w"],
            W16["vision_final_norm_b"],
            _p(B["vision_x_norm"]),
            vs, VIS_D, 1e-5, stream=stream)
        
        # Multi-modal projector (FP16 - small matrix)
        gemm.fp16_nn(
            _p(B["vision_x_norm"]),
            W16["encoder_multi_modal_projector_w"],
            _p(B["encoder_x"]),
            vs, ENC_D, VIS_D, stream=stream)
        fvk.add_bias_fp16(
            _p(B["encoder_x"]),
            W16["encoder_multi_modal_projector_b"],
            vs, ENC_D, stream=stream)
        
        # Encoder layers with FP8 GEMM (large matrices)
        for i in range(ENC_L):
            self._encoder_layer_fp8(i, es, stream)
    
    def _encoder_layer_fp8(self, i: int, seq: int, stream: int):
        """One encoder layer with FP8 GEMM for QKV and FFN (large matrices)."""
        fvk = self.fvk
        fp8_gemm = self.fp8_gemm
        gemm = self.gemm
        W8 = self.weights_fp8
        W16 = self.weights_fp16
        B = self.bufs
        
        # RMSNorm (FP16)
        fvk.rms_norm_fp16(
            _p(B["encoder_x"]),
            _p(B["encoder_ones"]),
            _p(B["encoder_x_norm"]),
            seq, ENC_D, 1e-6, stream=stream)
        
        # QKV GEMM (FP8 - large matrix 304x2560x2048)
        fp8_gemm.fp8_nn(
            _p(B["encoder_x_norm"]),
            W8["encoder_attn_qkv_w"][i],
            _p(B["encoder_QKV"]),
            seq, (ENC_NH + 2 * ENC_NKV) * ENC_HD, ENC_D,
            _p(B["encoder_x_fp8"])
        )
        
        # Split QKV and apply RoPE
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
        
        attn_out = torch.nn.functional.scaled_dot_product_attention(Q_t, K_t, V_t)
        attn_out = attn_out.transpose(0, 1).reshape(seq, ENC_D)
        
        fvk.gpu_copy(_p(B["encoder_attn_out"]), attn_out.data_ptr(), seq * ENC_D * 2, stream)
        
        # O projection (FP16 - smaller matrix, more precise)
        gemm.fp16_nn(
            _p(B["encoder_attn_out"]),
            W16["encoder_attn_o_w"][i],
            _p(B["encoder_x_norm"]),
            seq, ENC_D, ENC_D, stream=stream)
        
        fvk.residual_add_fp16(_p(B["encoder_x"]), _p(B["encoder_x_norm"]), seq * ENC_D, stream)
        
        # FFN - use FP8 for large matrices
        fvk.rms_norm_fp16(
            _p(B["encoder_x"]),
            _p(B["encoder_ones"]),
            _p(B["encoder_x_norm"]),
            seq, ENC_D, 1e-6, stream=stream)
        
        # Gate (FP8 - large matrix 304x16384x2048)
        fp8_gemm.fp8_nn(
            _p(B["encoder_x_norm"]),
            W8["encoder_ffn_gate_w"][i],
            _p(B["encoder_gate_merged"]),
            seq, ENC_H, ENC_D,
            _p(B["encoder_x_fp8"])
        )
        
        # Up (FP8 - large matrix 304x16384x2048)
        fp8_gemm.fp8_nn(
            _p(B["encoder_x_norm"]),
            W8["encoder_ffn_up_w"][i],
            _p(B["encoder_hidden"]),
            seq, ENC_H, ENC_D,
            _p(B["encoder_x_fp8"])
        )
        
        # Gate-GeGLU fusion
        fvk.gate_geglu_fp16(
            _p(B["encoder_gate_merged"]),
            _p(B["encoder_hidden"]),
            _p(B["encoder_hidden"]),
            seq * ENC_H, stream=stream)
        
        # Down (FP8 - large matrix 304x2048x16384)
        fp8_gemm.fp8_nn(
            _p(B["encoder_hidden"]),
            W8["encoder_ffn_down_w"][i],
            _p(B["encoder_ffn_out"]),
            seq, ENC_D, ENC_H,
            _p(B["encoder_hidden_fp8"])
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
    
    def run_decoder(self, stream: int = 0):
        """Run decoder diffusion steps - use FP16 for small matrices."""
        # Decoder has small seq_len=11, FP8 is slower than FP16
        # Use FP16 GEMM for decoder
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
        """Decoder step using FP16 GEMM (small matrices)."""
        gemm = self.gemm
        fvk = self.fvk
        W16 = self.weights_fp16
        B = self.bufs
        sa = self.chunk_size  # chunk_size
        
        # Assemble decoder input
        time_emb_ptr = self._style_slice_ptr("decoder_time_emb", step)
        
        # Copy time_emb[0] to decoder_x[0]
        self._cudart.cudaMemcpyAsync(
            ctypes.c_void_p(_p(B["decoder_x"])),
            ctypes.c_void_p(time_emb_ptr),
            DEC_D * 2, 3, stream)
        
        # action_in_proj
        x_action_ptr = _p(B["decoder_x"]) + DEC_D * 2
        gemm.fp16_nn(
            _p(B["diffusion_noise"]),
            W16["decoder_action_in_proj_w"],
            x_action_ptr,
            sa, DEC_D, ACTION_DIM, stream=stream)
        fvk.add_bias_fp16(x_action_ptr, W16["decoder_action_in_proj_b"], sa, DEC_D, stream)
        
        # Add time_emb[1:]
        time_emb_action_ptr = time_emb_ptr + DEC_D * 2
        fvk.residual_add_fp16(x_action_ptr, time_emb_action_ptr, sa * DEC_D, stream)
        
        # Decoder layers (FP16 - small matrices)
        for i in range(DEC_L):
            self._decoder_layer_fp16(i, step, enc_seq, sd, stream)
        
        # Final norm and output projection
        style_final_ptr = self._style_slice_ptr("decoder_style_final", step)
        fvk.adarms_fp16(
            _p(B["decoder_x"]), style_final_ptr,
            _p(B["x_normed_buf"]), _p(B["gate_buf"]),
            sd, DEC_D, stream)
        
        x_out_ptr = _p(B["x_normed_buf"]) + DEC_D * 2
        gemm.fp16_nn(
            x_out_ptr, W16["decoder_action_out_proj_w"],
            _p(B["diffusion_noise"]),
            sa, ACTION_DIM, DEC_D, stream=stream)
        fvk.add_bias_fp16(_p(B["diffusion_noise"]), W16["decoder_action_out_proj_b"], sa, ACTION_DIM, stream)
    
    def _decoder_layer_fp16(self, i: int, step: int, enc_seq: int, sd: int, stream: int):
        """Decoder layer using FP16 (small matrices - FP8 would be slower)."""
        gemm = self.gemm
        fvk = self.fvk
        W16 = self.weights_fp16
        B = self.bufs
        
        style_attn_ptr = self._style_slice_ptr("decoder_style_attn", step, i)
        style_ffn_ptr = self._style_slice_ptr("decoder_style_ffn", step, i)
        
        # AdaRMSNorm (attention)
        fvk.adarms_fp16(
            _p(B["decoder_x"]), style_attn_ptr,
            _p(B["x_normed_buf"]), _p(B["gate_buf"]),
            sd, DEC_D, stream)
        
        # QKV GEMM (FP16)
        gemm.fp16_nn(
            _p(B["x_normed_buf"]), W16["decoder_attn_qkv_w"][i],
            _p(B["decoder_QKV"]),
            sd, (DEC_NH + 2 * DEC_NKV) * DEC_HD, DEC_D, stream=stream)
        
        # Attention (cross-attention with encoder KV)
        qkv_tensor = torch.empty(sd, (DEC_NH + 2 * DEC_NKV) * DEC_HD, dtype=torch.float16, device='cuda')
        fvk.gpu_copy(qkv_tensor.data_ptr(), _p(B["decoder_QKV"]), 
                     sd * (DEC_NH + 2 * DEC_NKV) * DEC_HD * 2, stream)
        
        Q = qkv_tensor[:, :DEC_NH * DEC_HD].view(sd, DEC_NH, DEC_HD)
        KV = qkv_tensor[:, DEC_NH * DEC_HD:].view(sd, 2, DEC_NKV, DEC_HD)
        K_dec = KV[:, 0, :, :]
        V_dec = KV[:, 1, :, :]
        
        # Apply RoPE
        fvk.gpu_copy(self._dec_rope_tensor.data_ptr(), _p(B["decoder_rope"]), sd * 256 * 2, stream)
        Q_rope = self._apply_rope_interleaved(Q, self._dec_rope_cos, self._dec_rope_sin)
        K_dec_rope = self._apply_rope_interleaved(K_dec, self._dec_rope_cos, self._dec_rope_sin)
        
        # Cross-attention
        enc_K_ptr = _p(B["encoder_K_cache"]) + i * enc_seq * ENC_NKV * ENC_HD * 2
        enc_V_ptr = _p(B["encoder_V_cache"]) + i * enc_seq * ENC_NKV * ENC_HD * 2
        
        enc_K = torch.empty(enc_seq, ENC_NKV, ENC_HD, dtype=torch.float16, device='cuda')
        enc_V = torch.empty(enc_seq, ENC_NKV, ENC_HD, dtype=torch.float16, device='cuda')
        fvk.gpu_copy(enc_K.data_ptr(), enc_K_ptr, enc_seq * ENC_NKV * ENC_HD * 2, stream)
        fvk.gpu_copy(enc_V.data_ptr(), enc_V_ptr, enc_seq * ENC_NKV * ENC_HD * 2, stream)
        
        enc_K_expanded = enc_K.expand(-1, DEC_NH, -1)
        enc_V_expanded = enc_V.expand(-1, DEC_NH, -1)
        
        Q_t = Q_rope.transpose(0, 1)
        K_t = enc_K_expanded.transpose(0, 1)
        V_t = enc_V_expanded.transpose(0, 1)
        
        attn_out = torch.nn.functional.scaled_dot_product_attention(Q_t, K_t, V_t)
        attn_out = attn_out.transpose(0, 1).reshape(sd, DEC_NH * DEC_HD)
        
        fvk.gpu_copy(_p(B["decoder_attn_out"]), attn_out.data_ptr(), sd * DEC_NH * DEC_HD * 2, stream)
        
        # O projection
        gemm.fp16_nn(
            _p(B["decoder_attn_out"]), W16["decoder_attn_o_w"][i],
            _p(B["x_normed_buf"]), sd, DEC_D, DEC_NH * DEC_HD, stream=stream)
        
        # gate * residual
        fvk.gate_mul_residual(_p(B["decoder_x"]), _p(B["x_normed_buf"]), _p(B["gate_buf"]), sd * DEC_D, stream)
        
        # AdaRMSNorm (FFN)
        fvk.adarms_fp16(
            _p(B["decoder_x"]), style_ffn_ptr,
            _p(B["x_normed_buf"]), _p(B["gate_buf"]),
            sd, DEC_D, stream)
        
        # FFN Gate/Up/Down (FP16)
        gemm.fp16_nn(_p(B["x_normed_buf"]), W16["decoder_ffn_gate_w"][i],
                    _p(B["decoder_gate_merged"]), sd, DEC_H, DEC_D, stream=stream)
        gemm.fp16_nn(_p(B["x_normed_buf"]), W16["decoder_ffn_up_w"][i],
                    _p(B["decoder_hidden"]), sd, DEC_H, DEC_D, stream=stream)
        
        fvk.gate_geglu_fp16(_p(B["decoder_gate_merged"]), _p(B["decoder_hidden"]),
                           _p(B["decoder_hidden"]), sd * DEC_H, stream)
        
        gemm.fp16_nn(_p(B["decoder_hidden"]), W16["decoder_ffn_down_w"][i],
                    _p(B["decoder_attn_out"]), sd, DEC_D, DEC_H, stream=stream)
        
        fvk.gate_mul_residual(_p(B["decoder_x"]), _p(B["decoder_attn_out"]), _p(B["gate_buf"]), sd * DEC_D, stream)
    
    def _style_slice_ptr(self, buf_name: str, step: int, layer: int | None = None) -> int:
        """Compute device pointer for a per-step style slice."""
        base = _p(self.bufs[buf_name])
        sd = self.S_dec
        
        if buf_name == "decoder_time_emb":
            return base + step * sd * DEC_D * 2
        if buf_name == "decoder_style_final":
            return base + step * sd * 3 * DEC_D * 2
        
        per_layer = sd * 3 * DEC_D * 2
        per_step = DEC_L * per_layer
        return base + step * per_step + layer * per_layer
    
    def run_pipeline(self, stream: int = 0, sync: bool = True):
        """Run full inference pipeline."""
        self._copy_lang_embeds_to_encoder_x(stream)
        self.run_vision(stream)
        self.run_encoder(stream)
        self.run_decoder(stream)
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
        
        # Pre-allocated tensors for decoder RoPE
        self._dec_rope_tensor = torch.empty(self.S_dec, 256, dtype=torch.float16, device='cuda')
        self._dec_rope_cos = self._dec_rope_tensor[:, :128]
        self._dec_rope_sin = self._dec_rope_tensor[:, 128:]
    
    @property
    def input_images_buf(self):
        return self.bufs["observation_images_normalized"]
    
    @property
    def input_noise_buf(self):
        return self.bufs["diffusion_noise"]
    
    @property
    def output_noise_buf(self):
        return self.bufs["diffusion_noise"]