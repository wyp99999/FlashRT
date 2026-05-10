"""FlashRT — Optimized Pi0.5 inference with SINGLE CUDA Graph (static unrolled decoder).

Key innovation:
- Unroll 10 decoder steps as static code (no loop variables)
- Single CUDA Graph captures entire pipeline (Vision + Encoder + Decoder 10 steps)
- Eliminates 12 replay calls → 1 replay call

Expected improvement:
- Current: 12 graphs, ~21ms, overhead 0.8ms
- This:    1 graph,  ~20ms, overhead ~0.3ms
- Gain:    ~0.5ms (2.4% improvement)
"""

from __future__ import annotations

import ctypes
import gc
import logging
import time
from pathlib import Path

import numpy as np
import torch

from flash_rt.models.pi05.pipeline_sm89_fp8_ffn import (
    DEC_D, ACTION_DIM, _p
)

logger = logging.getLogger(__name__)


class Pi05TorchFrontendSm89SingleGraph:
    """Pi0.5 inference with SINGLE CUDA Graph using static unrolled decoder.
    
    Key optimization:
    - Unroll decoder 10 steps as static code (no for loop)
    - All step values are constants (0, 1, 2, ..., 9)
    - Single CUDA Graph captures entire pipeline
    - One replay() call completes all computation
    
    Usage:
        pipe = Pi05TorchFrontendSm89SingleGraph('/data/models/pi05_base', num_views=2, num_steps=10)
        pipe.set_prompt('pick up')
        pipe.build_pipeline()
        result = pipe.infer(obs)
    """
    
    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        num_views: int = 2,
        max_prompt_len: int = 50,
        chunk_size: int = 10,
        num_steps: int = 10,
        use_fp8_ffn: bool = True,
    ):
        """Initialize single graph frontend."""
        from flash_rt.frontends.torch.pi05_sm89_fp8_ffn import Pi05TorchFrontendSm89Fp8Ffn
        
        # Create base frontend
        self._base_frontend = Pi05TorchFrontendSm89Fp8Ffn(
            checkpoint_dir, num_views=num_views, max_prompt_len=max_prompt_len,
            chunk_size=chunk_size, num_steps=num_steps, use_fp8_ffn=use_fp8_ffn)
        
        # Copy attributes
        self.num_views = num_views
        self.num_steps = num_steps  # Must be 10 for static unroll
        self.chunk_size = chunk_size
        self.max_prompt_len = max_prompt_len
        self.use_fp8_ffn = use_fp8_ffn
        self.checkpoint_dir = Path(checkpoint_dir)
        
        self.latency_records = []
        self._graphs_built = False
        self._single_graph = None
        
        logger.info("Pi05TorchFrontendSm89SingleGraph initialised (num_steps=%d)", num_steps)
    
    def set_prompt(self, prompt_text: str):
        """Set text prompt."""
        self._base_frontend.set_prompt(prompt_text)
        logger.info("Set prompt: %d tokens", self._base_frontend._prompt_len)
    
    def build_pipeline(self):
        """Build pipeline with SINGLE CUDA Graph (static unrolled decoder)."""
        if self._graphs_built:
            return
        
        # Build base frontend pipeline first
        self._base_frontend.build_pipeline()
        
        # Get pipeline and components
        self._pipeline = self._base_frontend._pipeline
        self._fvk = self._base_frontend._fvk
        self._gemm = self._base_frontend._gemm
        self._cudart = self._base_frontend._cudart
        self._img_buf = self._base_frontend._img_buf
        self._noise_buf = self._base_frontend._noise_buf
        self._noise_out = self._base_frontend._noise_out
        
        # Capture SINGLE CUDA Graph with static unrolled decoder
        self._capture_single_graph_static_unrolled()
        
        self._graphs_built = True
        logger.info("Pipeline built with SINGLE CUDA Graph (static unrolled %d steps)", self.num_steps)
    
    def _run_decoder_step_static(self, step: int, stream: int):
        """Execute single decoder step with fixed step value.
        
        This is called during graph capture with step as a CONSTANT.
        """
        p = self._pipeline
        
        # Assemble decoder input for this step (step is constant)
        p._assemble_decoder_x(step, stream)
        
        # Run all decoder layers (layer loop is OK - doesn't affect pointers)
        for layer in range(p.dec_layers):
            p._decoder_layer(layer, step, p.encoder_seq_len, p.S_dec, stream)
        
        # Final AdaRMSNorm + output projection
        style_final_ptr = p._style_slice_ptr("decoder_style_final", step)
        
        p._ada_rms_norm_fp16(
            _p(p.bufs["decoder_x"]),
            style_final_ptr,
            _p(p.bufs["x_normed_buf"]),
            _p(p.bufs["gate_buf"]),
            p.S_dec, DEC_D, stream)
        
        x_out_action_ptr = _p(p.bufs["x_normed_buf"]) + DEC_D * 2
        self._gemm.fp16_nn(
            x_out_action_ptr,
            p.weights["decoder_action_out_proj_w"],
            _p(p.bufs["diffusion_noise"]),
            p.S_dec - 1, ACTION_DIM, DEC_D, stream=stream)
        self._fvk.add_bias_fp16(
            _p(p.bufs["diffusion_noise"]),
            p.weights["decoder_action_out_proj_b"],
            p.S_dec - 1, ACTION_DIM, stream=stream)
    
    def _capture_single_graph_static_unrolled(self):
        """Capture SINGLE CUDA Graph with static unrolled decoder.
        
        Key: Unroll the 10 decoder steps as static code.
        Each step uses a constant (0, 1, 2, ..., 9), not a loop variable.
        This allows CUDA Graph to capture the entire pipeline correctly.
        """
        logger.info("Capturing SINGLE CUDA Graph with static unrolled decoder...")
        
        # Warmup
        stream = 0
        for _ in range(3):
            self._pipeline.run_pipeline(stream, sync=True)
        torch.cuda.synchronize()
        
        # Single Graph capture
        self._single_graph = torch.cuda.CUDAGraph()
        
        with torch.cuda.graph(self._single_graph):
            # ========== Phase A: Vision Encoder (1 graph) ==========
            self._pipeline.vision_encoder(stream)
            
            # ========== Phase B: Transformer Encoder (1 graph) ==========
            self._pipeline.transformer_encoder(stream)
            
            # ========== Phase C: Decoder 10 steps (STATIC UNROLLED) ⭐⭐⭐ ==========
            # Step 0 - step is CONSTANT 0
            self._run_decoder_step_static(0, stream)
            
            # Step 1 - step is CONSTANT 1
            self._run_decoder_step_static(1, stream)
            
            # Step 2 - step is CONSTANT 2
            self._run_decoder_step_static(2, stream)
            
            # Step 3 - step is CONSTANT 3
            self._run_decoder_step_static(3, stream)
            
            # Step 4 - step is CONSTANT 4
            self._run_decoder_step_static(4, stream)
            
            # Step 5 - step is CONSTANT 5
            self._run_decoder_step_static(5, stream)
            
            # Step 6 - step is CONSTANT 6
            self._run_decoder_step_static(6, stream)
            
            # Step 7 - step is CONSTANT 7
            self._run_decoder_step_static(7, stream)
            
            # Step 8 - step is CONSTANT 8
            self._run_decoder_step_static(8, stream)
            
            # Step 9 - step is CONSTANT 9
            self._run_decoder_step_static(9, stream)
        
        logger.info("SINGLE CUDA Graph capture complete (Vision + Encoder + Decoder %d steps)", self.num_steps)
    
    def infer(self, observation, debug=False, reset_noise=True):
        """Run inference with SINGLE CUDA Graph replay.
        
        One replay() call completes the entire pipeline:
        Vision → Encoder → Decoder 10 steps
        """
        if not self._graphs_built:
            self.build_pipeline()
        
        t0 = time.perf_counter()
        
        # Prepare images
        if "images" in observation:
            img_list = observation["images"]
        elif self.num_views == 1:
            img_list = [observation["image"]]
        else:
            img_list = [observation["image"], observation.get("wrist_image", observation["image"])]
        
        for v, im in enumerate(img_list[:self.num_views]):
            norm = torch.from_numpy(im.astype(np.float32) / 127.5 - 1.0)
            self._img_buf[v].copy_(norm.to(torch.float16))
        
        if reset_noise:
            self._noise_buf.normal_()
        
        stream = 0
        
        # Copy inputs to pipeline buffers
        self._fvk.gpu_copy(self._pipeline.input_images_buf.ptr.value,
                           self._img_buf.data_ptr(), self._img_buf.numel() * 2, stream)
        self._fvk.gpu_copy(self._pipeline.input_noise_buf.ptr.value,
                           self._noise_buf.data_ptr(), self._noise_buf.numel() * 2, stream)
        
        # Copy language embeddings
        self._pipeline._copy_lang_embeds_to_encoder_x(stream)
        
        # ⭐⭐⭐ SINGLE Graph replay - completes all computation in one call ⭐⭐⭐
        self._single_graph.replay()
        
        self._cudart.cudaStreamSynchronize(ctypes.c_void_p(stream))
        
        # Copy output
        self._fvk.gpu_copy(self._noise_out.data_ptr(),
                           self._pipeline.output_noise_buf.ptr.value, 
                           self._noise_out.numel() * 2, stream)
        
        raw_actions = self._noise_out.float().cpu().numpy()
        
        # Unnormalize
        norm_stats = self._base_frontend.norm_stats
        if 'action' in norm_stats:
            mean = norm_stats['action']['mean']
            std = norm_stats['action']['std']
            if len(mean) < ACTION_DIM:
                mean = np.concatenate([mean, np.zeros(ACTION_DIM - len(mean))])
                std = np.concatenate([std, np.ones(ACTION_DIM - len(std))])
            unnorm = raw_actions * std + mean
            robot_actions = unnorm[:, :7]
        else:
            robot_actions = raw_actions[:, :7]
        
        latency_ms = (time.perf_counter() - t0) * 1000
        self.latency_records.append(latency_ms)
        
        if debug:
            logger.info("Latency: %.1f ms (Single Graph)", latency_ms)
        
        return {"actions": robot_actions}


# Convenience alias
Pi05SingleGraph = Pi05TorchFrontendSm89SingleGraph