"""FlashRT -- SM89 FP16 Pi0 torch frontend.

SM89 (RTX 4060 Ti/4090) has limited FP8 GEMM support in cuBLAS.
This frontend uses FP16 weights throughout, avoiding FP8 quantization.

Key differences from pi0_rtx.py:
- Memory-optimized loading (device='cpu' + batch GPU transfer)
- No FP8 quantization of weights
- No calibration needed (FP16 doesn't require dynamic scaling)
- Uses torch.nn.functional.scaled_dot_product_attention for stability
- FP16 throughout the pipeline

Usage::

    from flash_rt.frontends.torch.pi0_sm89 import Pi0TorchFrontendSm89
    pipe = Pi0TorchFrontendSm89("/path/to/pi0_libero_pytorch", num_views=2)
    pipe.set_prompt("pick up the red block")
    out = pipe.infer({"image": img, "wrist_image": wrist, "state": state})
    actions = out["actions"]  # shape: (10, 7)
"""

from __future__ import annotations

import ctypes
import gc
import json
import logging
import math
import os
import pathlib
import time
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from flash_rt.models.pi0.pipeline_sm89 import (
    Pi0PipelineSm89,
    VIS_L, VIS_D, VIS_H, VIS_NH, VIS_HD, VIS_SEQ_PER_VIEW, VIS_PATCH_FLAT,
    ENC_L, ENC_D, ENC_H, ENC_NH, ENC_NKV, ENC_HD,
    DEC_L, DEC_D, DEC_H, DEC_NH, DEC_NKV, DEC_HD,
    ACTION_DIM, CHUNK_SIZE_DEFAULT, NUM_STEPS_DEFAULT,
)

logger = logging.getLogger(__name__)

fp16 = torch.float16

IMG_HW = 224
MAX_PROMPT_LEN_DEFAULT = 48


class Pi0TorchFrontendSm89:
    """SM89 FP16 Pi0 torch frontend.
    
    Uses memory-optimized loading and FP16 weights throughout.
    No calibration needed since FP16 doesn't require dynamic scaling.
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, pathlib.Path],
        num_views: int = 2,
        chunk_size: int = CHUNK_SIZE_DEFAULT,
        max_prompt_len: int = MAX_PROMPT_LEN_DEFAULT,
        num_steps: int = NUM_STEPS_DEFAULT,
    ):
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.num_views = int(num_views)
        self.chunk_size = int(chunk_size)
        self.S_dec = self.chunk_size + 1
        self.max_prompt_len = int(max_prompt_len)
        self.num_steps = int(num_steps)
        
        # Fixed layer configuration (18 encoder + 18 decoder)
        # NOTE: Reducing layers causes severe quality loss in diffusion models
        self.enc_layers = ENC_L
        self.dec_layers = DEC_L
        
        # Validate num_steps range (flow matching quality vs performance tradeoff)
        if not (1 <= self.num_steps <= 20):
            raise ValueError(f"num_steps must be in [1, 20], got {num_steps}")
        
        self.latency_records: list[float] = []
        self._calibrated = True  # FP16 doesn't need calibration
        self._graphs_built = False
        self._weight_store = []
        self._ckpt_fp16 = {}
        
        # Load norm stats
        self._load_norm_stats(checkpoint_dir)
        
        # Memory-optimized checkpoint loading
        safetensors_path = checkpoint_dir / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(
                f"safetensors not found at {safetensors_path}")
        self._checkpoint_path = str(safetensors_path)
        
        # Memory-optimized load
        self._load_checkpoint_memory_optimized(safetensors_path)
        
        # Build weights dict
        self._build_weights()
        
        # Init kernels
        from flash_rt import flash_rt_kernels as fvk
        self._fvk = fvk
        self._gemm = fvk.GemmRunner()
        self._cudart = ctypes.CDLL("libcudart.so")
        
        logger.info(
            "Pi0TorchFrontendSm89 initialised (num_views=%d, chunk_size=%d, num_steps=%d)",
            self.num_views, self.chunk_size, self.num_steps)
    
    def _load_norm_stats(self, checkpoint_dir: pathlib.Path) -> None:
        """Load normalization statistics."""
        from flash_rt.core.utils.norm_stats import (
            load_norm_stats, lerobot_candidates,
        )
        candidates = [
            checkpoint_dir / "assets" / "physical-intelligence" / "libero"
            / "norm_stats.json",
            checkpoint_dir / "norm_stats.json",
            *lerobot_candidates(checkpoint_dir),
        ]
        try:
            self.norm_stats = load_norm_stats(
                candidates, checkpoint_dir=checkpoint_dir)
        except FileNotFoundError as e:
            # Use dummy norm stats if not found
            logger.warning(f"norm_stats not found: {e}, using identity")
            self.norm_stats = {
                "action": {"mean": np.zeros(7), "std": np.ones(7)},
            }
    
    def _load_checkpoint_memory_optimized(self, safetensors_path) -> None:
        """Load safetensors with memory optimization."""
        from safetensors import safe_open
        
        logger.info("Memory-optimized loading: %s", safetensors_path)
        
        gc.collect()
        torch.cuda.empty_cache()
        
        fp16_weights = {}
        with safe_open(str(safetensors_path), framework='pt', device='cpu') as f:
            keys = list(f.keys())
            logger.info("Total tensors: %d", len(keys))
            
            # Batch processing
            batch_size = 100
            for i in range(0, len(keys), batch_size):
                batch_keys = keys[i:i+batch_size]
                
                for k in batch_keys:
                    t_cpu = f.get_tensor(k)
                    t_gpu = t_cpu.to(fp16).cuda().contiguous()
                    fp16_weights[k] = t_gpu
                    del t_cpu
                
                gc.collect()
                
                if (i // batch_size + 1) % 3 == 0:
                    logger.info(
                        "  Loaded %d/%d tensors, GPU: %.1f MB",
                        i + len(batch_keys), len(keys),
                        torch.cuda.memory_allocated() / 1024**2)
        
        # Auto-strip prefix
        from flash_rt.executors.torch_weights import _autodetect_strip_prefix
        self._strip_prefix = _autodetect_strip_prefix(set(fp16_weights.keys()))
        
        # Store weights with stripped keys
        if self._strip_prefix:
            self._sd = {}
            for k, v in fp16_weights.items():
                new_key = k[len(self._strip_prefix):] if k.startswith(self._strip_prefix) else k
                self._sd[new_key] = v
        else:
            self._sd = fp16_weights
        
        self.embedding_weight = self._sd.get("paligemma_with_expert.paligemma.lm_head.weight")
        if self.embedding_weight is None:
            # Try alternative keys
            for key in ["paligemma.lm_head.weight", "paligemma_with_expert.paligemma.lm_head.weight"]:
                if key in self._sd:
                    self.embedding_weight = self._sd[key]
                    break
            # Also check with model. prefix
            if self.embedding_weight is None:
                full_key = f"model.paligemma_with_expert.paligemma.lm_head.weight"
                if full_key in fp16_weights:
                    self.embedding_weight = fp16_weights[full_key]
        
        logger.info(
            "Loaded %d tensors, GPU memory: %.1f MB",
            len(self._sd), torch.cuda.memory_allocated() / 1024**2)
    
    def _build_weights(self) -> None:
        """Build processed weights dict for pipeline."""
        W = self._sd
        store = self._weight_store
        
        def _stash(t):
            store.append(t)
            return t
        
        def g(key: str) -> torch.Tensor:
            return W.get(key, W.get(f"model.{key}"))
        
        # Vision encoder (27 SigLIP layers)
        vp = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
        
        # Patch embedding
        pe_w = g(f"{vp}.embeddings.patch_embedding.weight")
        if pe_w is not None:
            self._ckpt_fp16["vision_patch_embedding_w"] = _stash(pe_w.permute(2, 3, 1, 0).contiguous())
        pe_b = g(f"{vp}.embeddings.patch_embedding.bias")
        if pe_b is not None:
            self._ckpt_fp16["vision_patch_embedding_b"] = _stash(pe_b)
        
        pos_emb = g(f"{vp}.embeddings.position_embedding.weight")
        if pos_emb is not None:
            self._ckpt_fp16["vision_position_embedding"] = _stash(pos_emb)
        
        # Vision layers
        for i in range(VIS_L):
            lp = f"{vp}.encoder.layers.{i}"
            
            # QKV
            q_w = g(f"{lp}.self_attn.q_proj.weight")
            k_w = g(f"{lp}.self_attn.k_proj.weight")
            v_w = g(f"{lp}.self_attn.v_proj.weight")
            if q_w is not None and k_w is not None and v_w is not None:
                qkv_w = torch.cat([q_w, k_w, v_w], dim=0).t()
                _stash(qkv_w)
                if "vision_attn_qkv_w" not in self._ckpt_fp16:
                    self._ckpt_fp16["vision_attn_qkv_w"] = []
                self._ckpt_fp16["vision_attn_qkv_w"].append(qkv_w)
            
            # QKV bias
            q_b = g(f"{lp}.self_attn.q_proj.bias")
            k_b = g(f"{lp}.self_attn.k_proj.bias")
            v_b = g(f"{lp}.self_attn.v_proj.bias")
            if q_b is not None and k_b is not None and v_b is not None:
                qkv_b = torch.cat([q_b, k_b, v_b])
                _stash(qkv_b)
                if "vision_attn_qkv_b" not in self._ckpt_fp16:
                    self._ckpt_fp16["vision_attn_qkv_b"] = []
                self._ckpt_fp16["vision_attn_qkv_b"].append(qkv_b)
            
            # O proj
            o_w = g(f"{lp}.self_attn.out_proj.weight")
            if o_w is not None:
                o_w_t = _stash(o_w.t())
                if "vision_attn_o_w" not in self._ckpt_fp16:
                    self._ckpt_fp16["vision_attn_o_w"] = []
                self._ckpt_fp16["vision_attn_o_w"].append(o_w_t)
            
            o_b = g(f"{lp}.self_attn.out_proj.bias")
            if o_b is not None:
                o_b_s = _stash(o_b)
                if "vision_attn_o_b" not in self._ckpt_fp16:
                    self._ckpt_fp16["vision_attn_o_b"] = []
                self._ckpt_fp16["vision_attn_o_b"].append(o_b_s)
            
            # FFN
            up_w = g(f"{lp}.mlp.fc1.weight")
            if up_w is not None:
                up_w_t = _stash(up_w.t())
                if "vision_ffn_up_w" not in self._ckpt_fp16:
                    self._ckpt_fp16["vision_ffn_up_w"] = []
                self._ckpt_fp16["vision_ffn_up_w"].append(up_w_t)
            
            up_b = g(f"{lp}.mlp.fc1.bias")
            if up_b is not None:
                up_b_s = _stash(up_b)
                if "vision_ffn_up_b" not in self._ckpt_fp16:
                    self._ckpt_fp16["vision_ffn_up_b"] = []
                self._ckpt_fp16["vision_ffn_up_b"].append(up_b_s)
            
            down_w = g(f"{lp}.mlp.fc2.weight")
            if down_w is not None:
                down_w_t = _stash(down_w.t())
                if "vision_ffn_down_w" not in self._ckpt_fp16:
                    self._ckpt_fp16["vision_ffn_down_w"] = []
                self._ckpt_fp16["vision_ffn_down_w"].append(down_w_t)
            
            down_b = g(f"{lp}.mlp.fc2.bias")
            if down_b is not None:
                down_b_s = _stash(down_b)
                if "vision_ffn_down_b" not in self._ckpt_fp16:
                    self._ckpt_fp16["vision_ffn_down_b"] = []
                self._ckpt_fp16["vision_ffn_down_b"].append(down_b_s)
            
            # LayerNorm
            ln1_w = g(f"{lp}.layer_norm1.weight")
            ln1_b = g(f"{lp}.layer_norm1.bias")
            ln2_w = g(f"{lp}.layer_norm2.weight")
            ln2_b = g(f"{lp}.layer_norm2.bias")
            
            for key, val in [("vision_pre_attn_norm_w", ln1_w), 
                            ("vision_pre_attn_norm_b", ln1_b),
                            ("vision_pre_ffn_norm_w", ln2_w),
                            ("vision_pre_ffn_norm_b", ln2_b)]:
                if val is not None:
                    v_s = _stash(val)
                    if key not in self._ckpt_fp16:
                        self._ckpt_fp16[key] = []
                    self._ckpt_fp16[key].append(v_s)
        
        # Stack vision lists
        for key in ["vision_attn_qkv_w", "vision_attn_qkv_b", "vision_attn_o_w",
                    "vision_attn_o_b", "vision_ffn_up_w", "vision_ffn_up_b",
                    "vision_ffn_down_w", "vision_ffn_down_b",
                    "vision_pre_attn_norm_w", "vision_pre_attn_norm_b",
                    "vision_pre_ffn_norm_w", "vision_pre_ffn_norm_b"]:
            if key in self._ckpt_fp16 and isinstance(self._ckpt_fp16[key], list):
                self._ckpt_fp16[key] = torch.stack(self._ckpt_fp16[key])
        
        # Final vision norm
        fn_w = g(f"{vp}.post_layernorm.weight")
        fn_b = g(f"{vp}.post_layernorm.bias")
        if fn_w is not None:
            self._ckpt_fp16["vision_final_norm_w"] = _stash(fn_w)
        if fn_b is not None:
            self._ckpt_fp16["vision_final_norm_b"] = _stash(fn_b)
        
        # Multi-modal projector
        mp = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
        proj_w = g(f"{mp}.weight")
        proj_b = g(f"{mp}.bias")
        if proj_w is not None:
            self._ckpt_fp16["encoder_multi_modal_projector_w"] = _stash(proj_w.t())
        if proj_b is not None:
            self._ckpt_fp16["encoder_multi_modal_projector_b"] = _stash(proj_b)
        
        # Encoder (Gemma-2B)
        ep = "paligemma_with_expert.paligemma.model.language_model.layers"
        enc_qkv_list, enc_o_list = [], []
        enc_gate_list, enc_up_list, enc_down_list = [], [], []
        
        for i in range(ENC_L):
            # Encoder QKV (需要 interleave Q/K for RoPE，简化处理直接使用)
            q_w = g(f"{ep}.{i}.self_attn.q_proj.weight")
            k_w = g(f"{ep}.{i}.self_attn.k_proj.weight")
            v_w = g(f"{ep}.{i}.self_attn.v_proj.weight")
            
            if q_w is not None and k_w is not None and v_w is not None:
                # Interleave Q/K for fused RoPE
                def _interleave_qk(w, num_heads):
                    out_dim, in_dim = w.shape
                    head_dim = out_dim // num_heads
                    return w.reshape(num_heads, head_dim, in_dim).reshape(
                        num_heads, 2, head_dim // 2, in_dim).permute(
                        0, 2, 1, 3).reshape(out_dim, in_dim)
                
                q_w_il = _interleave_qk(q_w, ENC_NH)
                k_w_il = _interleave_qk(k_w, ENC_NKV)
                qkv_enc = torch.cat([q_w_il, k_w_il, v_w], dim=0).t()
                enc_qkv_list.append(_stash(qkv_enc))
            
            o_w = g(f"{ep}.{i}.self_attn.o_proj.weight")
            if o_w is not None:
                enc_o_list.append(_stash(o_w.t()))
            
            gate_w = g(f"{ep}.{i}.mlp.gate_proj.weight")
            up_w = g(f"{ep}.{i}.mlp.up_proj.weight")
            down_w = g(f"{ep}.{i}.mlp.down_proj.weight")
            
            if gate_w is not None:
                enc_gate_list.append(_stash(gate_w.t()))
            if up_w is not None:
                enc_up_list.append(_stash(up_w.t()))
            if down_w is not None:
                enc_down_list.append(_stash(down_w.t()))
        
        if enc_qkv_list:
            self._ckpt_fp16["encoder_attn_qkv_w"] = torch.stack(enc_qkv_list)
        if enc_o_list:
            self._ckpt_fp16["encoder_attn_o_w"] = torch.stack(enc_o_list)
        if enc_gate_list:
            self._ckpt_fp16["encoder_ffn_gate_w"] = torch.stack(enc_gate_list)
        if enc_up_list:
            self._ckpt_fp16["encoder_ffn_up_w"] = torch.stack(enc_up_list)
        if enc_down_list:
            self._ckpt_fp16["encoder_ffn_down_w"] = torch.stack(enc_down_list)
        
        # Decoder (Gemma-300M)
        dp = "paligemma_with_expert.gemma_expert.model.layers"
        dec_qkv_list, dec_o_list = [], []
        dec_gate_list, dec_up_list, dec_down_list = [], [], []
        
        for i in range(DEC_L):
            q_w = g(f"{dp}.{i}.self_attn.q_proj.weight")
            k_w = g(f"{dp}.{i}.self_attn.k_proj.weight")
            v_w = g(f"{dp}.{i}.self_attn.v_proj.weight")
            
            if q_w is not None and k_w is not None and v_w is not None:
                def _interleave_qk(w, num_heads):
                    out_dim, in_dim = w.shape
                    head_dim = out_dim // num_heads
                    return w.reshape(num_heads, head_dim, in_dim).reshape(
                        num_heads, 2, head_dim // 2, in_dim).permute(
                        0, 2, 1, 3).reshape(out_dim, in_dim)
                
                q_w_il = _interleave_qk(q_w, DEC_NH)
                k_w_il = _interleave_qk(k_w, DEC_NKV)
                qkv_dec = torch.cat([q_w_il, k_w_il, v_w], dim=0).t()
                dec_qkv_list.append(_stash(qkv_dec))
            
            o_w = g(f"{dp}.{i}.self_attn.o_proj.weight")
            if o_w is not None:
                dec_o_list.append(_stash(o_w.t()))
            
            gate_w = g(f"{dp}.{i}.mlp.gate_proj.weight")
            up_w = g(f"{dp}.{i}.mlp.up_proj.weight")
            down_w = g(f"{dp}.{i}.mlp.down_proj.weight")
            
            if gate_w is not None:
                dec_gate_list.append(_stash(gate_w.t()))
            if up_w is not None:
                dec_up_list.append(_stash(up_w.t()))
            if down_w is not None:
                dec_down_list.append(_stash(down_w.t()))
        
        if dec_qkv_list:
            self._ckpt_fp16["decoder_attn_qkv_w"] = torch.stack(dec_qkv_list)
        if dec_o_list:
            self._ckpt_fp16["decoder_attn_o_w"] = torch.stack(dec_o_list)
        if dec_gate_list:
            self._ckpt_fp16["decoder_ffn_gate_w"] = torch.stack(dec_gate_list)
        if dec_up_list:
            self._ckpt_fp16["decoder_ffn_up_w"] = torch.stack(dec_up_list)
        if dec_down_list:
            self._ckpt_fp16["decoder_ffn_down_w"] = torch.stack(dec_down_list)
        
        # Decoder final norm
        norm_w = g("paligemma_with_expert.gemma_expert.model.norm.weight")
        if norm_w is not None:
            # Fold (1 + w) inline
            self._ckpt_fp16["decoder_final_norm_w"] = _stash((1.0 + norm_w).contiguous())
        
        # Pi0 action/state projections
        state_proj_w = g("state_proj.weight")
        state_proj_b = g("state_proj.bias")
        if state_proj_w is not None:
            self._ckpt_fp16["state_proj_w"] = _stash(state_proj_w.t().contiguous())
        if state_proj_b is not None:
            self._ckpt_fp16["state_proj_b"] = _stash(state_proj_b)
        
        action_in_proj_w = g("action_in_proj.weight")
        action_in_proj_b = g("action_in_proj.bias")
        if action_in_proj_w is not None:
            self._ckpt_fp16["decoder_action_in_proj_w"] = _stash(action_in_proj_w.t().contiguous())
        if action_in_proj_b is not None:
            self._ckpt_fp16["decoder_action_in_proj_b"] = _stash(action_in_proj_b)
        
        # action_time_mlp
        atmlp_in_w = g("action_time_mlp_in.weight")
        atmlp_in_b = g("action_time_mlp_in.bias")
        atmlp_out_w = g("action_time_mlp_out.weight")
        atmlp_out_b = g("action_time_mlp_out.bias")
        
        if atmlp_in_w is not None:
            # Split action/time halves: (Da, 2*Da) → (Da, Da) each
            self._ckpt_fp16["action_time_mlp_in_wa_w"] = _stash(
                atmlp_in_w[:, :DEC_D].t().contiguous())
            self._time_proj_wt = atmlp_in_w[:, DEC_D:].contiguous()  # For precompute
        
        if atmlp_in_b is not None:
            self._time_proj_bias = atmlp_in_b
        
        if atmlp_out_w is not None:
            self._ckpt_fp16["action_time_mlp_out_w"] = _stash(atmlp_out_w.t().contiguous())
        if atmlp_out_b is not None:
            self._ckpt_fp16["action_time_mlp_out_b"] = _stash(atmlp_out_b)
        
        # action_out_proj (pre-scaled by -1/num_steps)
        action_out_proj_w = g("action_out_proj.weight")
        action_out_proj_b = g("action_out_proj.bias")
        
        if action_out_proj_w is not None:
            dt_scale = -1.0 / self.num_steps  # Use dynamic num_steps
            self._ckpt_fp16["decoder_action_out_proj_w"] = _stash(
                (action_out_proj_w.t() * dt_scale).contiguous())
        if action_out_proj_b is not None:
            dt_scale = -1.0 / self.num_steps  # Use dynamic num_steps
            self._ckpt_fp16["decoder_action_out_proj_b"] = _stash(
                (action_out_proj_b * dt_scale).contiguous())
        
        # Pre-compute time_proj_all (if weights available)
        if hasattr(self, '_time_proj_wt'):
            self._precompute_time_proj_all()
        
        logger.info("Built %d weight entries", len(self._ckpt_fp16))
    
    def _precompute_time_proj_all(self) -> None:
        """Pre-compute time embeddings for all diffusion steps."""
        if not hasattr(self, '_time_proj_wt') or not hasattr(self, '_time_proj_bias'):
            logger.warning("time_proj weights not available, skipping precompute")
            return
        
        W_t = self._time_proj_wt.to(torch.float32)
        b = self._time_proj_bias.to(torch.float32)
        
        # Compute time embeddings
        fraction = torch.linspace(0.0, 1.0, DEC_D // 2, dtype=torch.float64, device="cuda")
        period = 4e-3 * (4.0 / 4e-3) ** fraction
        scaling = 1.0 / period * 2 * math.pi
        
        out = torch.empty(self.num_steps, self.chunk_size, DEC_D, 
                          dtype=torch.float16, device="cuda")
        
        for step in range(self.num_steps):  # Use dynamic num_steps
            t_val = 1.0 - step / self.num_steps
            sin_input = scaling * t_val
            time_emb_f32 = torch.cat([
                torch.sin(sin_input),
                torch.cos(sin_input)
            ], dim=-1).to(torch.float32)
            
            # Linear projection: time_emb @ W_t.T + b
            tp = (time_emb_f32.unsqueeze(0) @ W_t.t() + b.unsqueeze(0)).to(torch.float16)
            out[step] = tp.expand(self.chunk_size, -1)
        
        self._time_proj_all = out.reshape(self.num_steps * self.chunk_size, DEC_D).contiguous()
        logger.info("Precomputed time_proj_all for %d steps", self.num_steps)
    
    def set_prompt(self, prompt_text) -> None:
        """Set the text prompt for inference."""
        if isinstance(prompt_text, str):
            # Simple tokenization: use character-level encoding as fallback
            # PaliGemma uses Gemma tokenizer, but we use a simple fallback
            
            # Try to find tokenizer
            tokenizer_found = False
            sp = None
            
            # Check for tokenizer files
            tokenizer_paths = [
                "/data/models/paligemma_tokenizer/tokenizer.model",
                "/data/models/pi0/tokenizer.model",
                "/root/.cache/openpi/big_vision/paligemma_tokenizer.model",
                "/workspace/paligemma_tokenizer.model",
            ]
            
            for sp_path in tokenizer_paths:
                if os.path.exists(sp_path):
                    try:
                        import sentencepiece as spm
                        sp = spm.SentencePieceProcessor()
                        sp.Load(sp_path)
                        tokenizer_found = True
                        logger.info("Loaded tokenizer from: %s", sp_path)
                        break
                    except Exception:
                        pass
            
            if tokenizer_found and sp:
                # Use sentencepiece tokenizer
                try:
                    bos_id = sp.bos_id() if hasattr(sp, 'bos_id') else 2
                    tokens = [bos_id] + sp.Encode(prompt_text) + [108]
                    token_ids = torch.tensor(tokens, dtype=torch.long, device="cuda")
                    prompt_len = len(token_ids)
                except Exception as e:
                    logger.warning("Tokenizer encode failed: %s, using fallback", e)
                    tokenizer_found = False
            
            if not tokenizer_found:
                # Fallback: simple character encoding
                # Use ASCII values as token ids (limited but functional)
                prompt_len = min(len(prompt_text), self.max_prompt_len)
                # Use a simple hash-based encoding
                token_ids = torch.zeros(prompt_len + 2, dtype=torch.long, device="cuda")
                token_ids[0] = 2  # BOS-like
                for i, c in enumerate(prompt_text[:prompt_len]):
                    token_ids[i + 1] = ord(c) % 257152  # vocab size
                token_ids[prompt_len + 1] = 108  # EOS-like
                prompt_len = prompt_len + 2
                logger.warning("No tokenizer found, using character fallback")
            
            # Embed tokens
            if self.embedding_weight is not None:
                embeds = F.embedding(token_ids, self.embedding_weight.to("cuda"))
                embeds = embeds * float(embeds.shape[-1] ** 0.5)
                self._prompt_embeds = embeds
                self._prompt_len = prompt_len
            else:
                logger.warning("No embedding weight, using zeros")
                self._prompt_embeds = torch.zeros(
                    prompt_len, ENC_D, dtype=fp16, device="cuda")
                self._prompt_len = prompt_len
        else:
            # Assume token ids
            token_ids = torch.tensor(prompt_text, dtype=torch.long, device="cuda")
            prompt_len = token_ids.numel()
            if self.embedding_weight is not None:
                embeds = F.embedding(token_ids, self.embedding_weight.to("cuda"))
                embeds = embeds * float(embeds.shape[-1] ** 0.5)
                self._prompt_embeds = embeds
            self._prompt_len = prompt_len
        
        logger.info("Set prompt: %d tokens", self._prompt_len)
    
    def build_pipeline(self) -> None:
        """Build inference pipeline buffers and create Pi0PipelineSm89 instance."""
        if self._graphs_built:
            return
        
        # Build position embedding expanded
        from flash_rt.models.pi0.pipeline_sm89 import Pi0PipelineSm89
        
        # Allocate buffers for frontend
        self._img_buf = torch.empty(
            self.num_views, IMG_HW, IMG_HW, 3, dtype=fp16, device="cuda")
        self._noise_buf = torch.empty(
            self.chunk_size, ACTION_DIM, dtype=fp16, device="cuda")
        self._noise_out = torch.empty(
            self.chunk_size, ACTION_DIM, dtype=fp16, device="cuda")
        self._state_buf = torch.zeros(
            1, ACTION_DIM, dtype=fp16, device="cuda")
        
        # Build weights dict for pipeline (using data_ptr)
        pipeline_weights = self._build_pipeline_weights()
        
        # Create pipeline instance
        if hasattr(self, '_prompt_len') and self._prompt_len > 0:
            max_prompt_len = self._prompt_len
        else:
            max_prompt_len = self.max_prompt_len
        
        self._pipeline = Pi0PipelineSm89(
            gemm=self._gemm,
            fvk=self._fvk,
            weights=pipeline_weights,
            num_views=self.num_views,
            max_prompt_len=max_prompt_len,
            chunk_size=self.chunk_size,
            num_steps=self.num_steps,  # Use dynamic num_steps
        )
        
        # Expand position embeddings
        self._pipeline._build_pos_embed_expanded()
        
        # Set language embeddings
        if hasattr(self, '_prompt_embeds'):
            self._pipeline.set_language_embeds(self._prompt_embeds.cpu().numpy())
        
        # Warmup + Capture CUDA Graph
        self._warmup_for_cuda_graph()
        self._capture_cuda_graph()
        
        self._graphs_built = True
        logger.info("Pipeline built with Pi0PipelineSm89 + CUDA Graph")
    
    def _warmup_for_cuda_graph(self) -> None:
        """Warmup runs before CUDA Graph capture."""
        dummy_img = torch.zeros(self.num_views, IMG_HW, IMG_HW, 3, dtype=fp16, device="cuda")
        dummy_noise = torch.zeros(self.chunk_size, ACTION_DIM, dtype=fp16, device="cuda")
        dummy_state = torch.zeros(1, ACTION_DIM, dtype=fp16, device="cuda")
        
        stream = 0
        for _ in range(3):
            self._fvk.gpu_copy(
                self._pipeline.input_images_buf.ptr.value,
                dummy_img.data_ptr(), dummy_img.numel() * 2, stream)
            self._fvk.gpu_copy(
                self._pipeline.input_noise_buf.ptr.value,
                dummy_noise.data_ptr(), dummy_noise.numel() * 2, stream)
            self._fvk.gpu_copy(
                self._pipeline.input_state_buf.ptr.value,
                dummy_state.data_ptr(), dummy_state.numel() * 2, stream)
            self._pipeline.run_pipeline(stream, sync=True)  # warmup需要sync
        
        torch.cuda.synchronize()
        logger.info("CUDA Graph warmup completed")
    
    def _capture_cuda_graph(self) -> None:
        """Capture the pipeline into a CUDA Graph."""
        self._cuda_graph = torch.cuda.CUDAGraph()
        self._graph_stream = torch.cuda.Stream()
        
        with torch.cuda.stream(self._graph_stream):
            stream_int = self._graph_stream.cuda_stream
            self._cuda_graph.capture_begin()
            self._pipeline.run_pipeline(stream_int, sync=False)  # capture不能sync
            self._cuda_graph.capture_end()
        
        logger.info("CUDA Graph captured successfully")
    
    def _build_pipeline_weights(self) -> dict:
        """Build weights dict with data_ptr for pipeline."""
        W = self._ckpt_fp16
        
        def p(key: str) -> int:
            """Get data_ptr for a weight."""
            if key not in W:
                logger.warning(f"Weight {key} not found in checkpoint")
                return 0
            return W[key].data_ptr()
        
        def p_list(key: str) -> list:
            """Get data_ptr list for stacked weights."""
            if key not in W:
                logger.warning(f"Weight list {key} not found")
                return [0] * (VIS_L if "vision" in key else (ENC_L if "encoder" in key else DEC_L))
            t = W[key]
            stride = t.stride(0) * t.element_size() if len(t.shape) > 0 else 0
            base = t.data_ptr()
            return [base + i * stride for i in range(t.shape[0])]
        
        weights = {
            # Vision
            "vision_patch_embedding_w": p("vision_patch_embedding_w"),
            "vision_patch_embedding_b": p("vision_patch_embedding_b"),
            "vision_position_embedding": p("vision_position_embedding"),
            "vision_pre_attn_norm_w": p_list("vision_pre_attn_norm_w"),
            "vision_pre_attn_norm_b": p_list("vision_pre_attn_norm_b"),
            "vision_pre_ffn_norm_w": p_list("vision_pre_ffn_norm_w"),
            "vision_pre_ffn_norm_b": p_list("vision_pre_ffn_norm_b"),
            "vision_attn_qkv_w": p_list("vision_attn_qkv_w"),
            "vision_attn_qkv_b": p_list("vision_attn_qkv_b"),
            "vision_attn_o_w": p_list("vision_attn_o_w"),
            "vision_attn_o_b": p_list("vision_attn_o_b"),
            "vision_ffn_up_w": p_list("vision_ffn_up_w"),
            "vision_ffn_up_b": p_list("vision_ffn_up_b"),
            "vision_ffn_down_w": p_list("vision_ffn_down_w"),
            "vision_ffn_down_b": p_list("vision_ffn_down_b"),
            "vision_final_norm_w": p("vision_final_norm_w"),
            "vision_final_norm_b": p("vision_final_norm_b"),
            # Encoder
            "encoder_multi_modal_projector_w": p("encoder_multi_modal_projector_w"),
            "encoder_multi_modal_projector_b": p("encoder_multi_modal_projector_b"),
            "encoder_attn_qkv_w": p_list("encoder_attn_qkv_w"),
            "encoder_attn_o_w": p_list("encoder_attn_o_w"),
            "encoder_ffn_gate_w": p_list("encoder_ffn_gate_w"),
            "encoder_ffn_up_w": p_list("encoder_ffn_up_w"),
            "encoder_ffn_down_w": p_list("encoder_ffn_down_w"),
            # Decoder
            "decoder_attn_qkv_w": p_list("decoder_attn_qkv_w"),
            "decoder_attn_o_w": p_list("decoder_attn_o_w"),
            "decoder_ffn_gate_w": p_list("decoder_ffn_gate_w"),
            "decoder_ffn_up_w": p_list("decoder_ffn_up_w"),
            "decoder_ffn_down_w": p_list("decoder_ffn_down_w"),
            "decoder_final_norm_w": p("decoder_final_norm_w"),
            # Pi0 specifics
            "state_proj_w": p("state_proj_w"),
            "state_proj_b": p("state_proj_b"),
            "decoder_action_in_proj_w": p("decoder_action_in_proj_w"),
            "decoder_action_in_proj_b": p("decoder_action_in_proj_b"),
            "action_time_mlp_in_wa_w": p("action_time_mlp_in_wa_w"),
            "action_time_mlp_out_w": p("action_time_mlp_out_w"),
            "action_time_mlp_out_b": p("action_time_mlp_out_b"),
            "decoder_action_out_proj_w": p("decoder_action_out_proj_w"),
            "decoder_action_out_proj_b": p("decoder_action_out_proj_b"),
            "time_proj_all": self._time_proj_all.data_ptr() if hasattr(self, '_time_proj_all') else 0,
        }
        
        return weights
    
    def infer(self, observation: dict, debug: bool = False, reset_noise: bool = True) -> dict:
        """Run inference using Pi0PipelineSm89.
        
        Args:
            observation: dict with 'image' and 'wrist_image'
            debug: enable debug logging
            reset_noise: whether to reset noise buffer (set False for reproducibility testing)
        """
        if not self._graphs_built:
            self.build_pipeline()
        
        t0 = time.perf_counter()
        
        # Fill buffers
        self._fill_img_buf(observation)
        if reset_noise:
            self._noise_buf.normal_()
        self._fill_state_buf(observation.get("state"))
        
        # Copy inputs to pipeline buffers (outside of graph)
        stream = 0
        fvk = self._fvk
        
        # Copy images
        fvk.gpu_copy(
            self._pipeline.input_images_buf.ptr.value,
            self._img_buf.data_ptr(),
            self._img_buf.numel() * 2, stream)
        
        # Copy noise
        fvk.gpu_copy(
            self._pipeline.input_noise_buf.ptr.value,
            self._noise_buf.data_ptr(),
            self._noise_buf.numel() * 2, stream)
        
        # Copy state
        fvk.gpu_copy(
            self._pipeline.input_state_buf.ptr.value,
            self._state_buf.data_ptr(),
            self._state_buf.numel() * 2, stream)
        
        # Replay CUDA Graph (zero kernel launch overhead)
        if hasattr(self, '_cuda_graph'):
            self._cuda_graph.replay()
        else:
            # Fallback if graph not captured
            self._pipeline.run_pipeline(stream, sync=True)
        
        # Copy output (outside of graph)
        fvk.gpu_copy(
            self._noise_out.data_ptr(),
            self._pipeline.output_noise_buf.ptr.value,
            self._noise_out.numel() * 2, stream)
        
        raw_actions = self._noise_out.float().cpu().numpy()
        
        # Unnormalize actions
        if hasattr(self, 'norm_stats') and 'action' in self.norm_stats:
            mean = self.norm_stats['action']['mean']
            std = self.norm_stats['action']['std']
            unnorm = raw_actions * std + mean
            robot_actions = unnorm[:, :7]  # LIBERO has 7-dim actions
        else:
            robot_actions = raw_actions[:, :7]
        
        latency_ms = (time.perf_counter() - t0) * 1000
        self.latency_records.append(latency_ms)
        
        if debug:
            logger.info("Latency: %.1f ms", latency_ms)
            logger.info("Raw actions[0,:5]: %s", raw_actions[0, :5])
        
        return {"actions": robot_actions}
    
    def _fill_img_buf(self, observation: dict) -> None:
        """Fill image buffer."""
        if "images" in observation:
            img_list = observation["images"]
        else:
            img_list = [observation["image"], observation["wrist_image"]]
        
        for v, im in enumerate(img_list[:self.num_views]):
            norm = torch.from_numpy(im.astype(np.float32) / 127.5 - 1.0)
            self._img_buf[v].copy_(norm.to(fp16))
    
    def _fill_state_buf(self, state) -> None:
        """Fill state buffer."""
        self._state_buf.zero_()
        if state is None:
            return
        if isinstance(state, torch.Tensor):
            s = state.to(dtype=fp16, device="cuda").reshape(-1)
        else:
            s_np = np.asarray(state, dtype=np.float32).reshape(-1)
            s = torch.from_numpy(s_np).to("cuda", fp16)
        n = min(s.numel(), ACTION_DIM)
        self._state_buf[0, :n].copy_(s[:n])
    
    def get_latency_stats(self) -> dict:
        """Get latency statistics."""
        if not self.latency_records:
            return {}
        lat = np.array(self.latency_records)
        return {
            "count": len(lat),
            "mean_ms": float(np.mean(lat)),
            "std_ms": float(np.std(lat)),
            "min_ms": float(np.min(lat)),
            "max_ms": float(np.max(lat)),
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
            "hz": float(1000 / np.mean(lat)),
        }