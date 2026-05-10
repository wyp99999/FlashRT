"""FlashRT -- SM89 FP16 GROOT N1.7 torch frontend.

SM89 (RTX 4060 Ti/4090) does not fully support FP8 E4M3 GEMM in cuBLAS.
This frontend uses FP16 weights throughout, avoiding FP8 quantization.

Key differences from groot_n17_thor.py:
- No FP8 quantization of weights
- No calibration needed (FP16 doesn't require dynamic scaling)
- Uses fp16_nn GEMM throughout
- Uses torch.nn.functional.scaled_dot_product_attention for stability
- State/Action dim: 132 (vs 128 in N1.6)
- Action Horizon: 40 (vs 50 in N1.6)
- New VL Self Attention module (4 layers)

Usage::

    from flash_rt.frontends.torch.groot_n17_sm89 import GrootN17TorchFrontendSm89
    pipe = GrootN17TorchFrontendSm89(
        "/path/to/GR00T-N1.7-3B",
        num_views=2,
        embodiment_tag="gr1_unified",
    )
    pipe.set_prompt("pick up the red block")
    out = pipe.infer({
        "image": img1,
        "wrist_image": img2,
        "state": np.zeros(state_dim, dtype=np.float32),
    })
    actions = out["actions"]  # shape: (40, 132)
"""

from __future__ import annotations

import ctypes
import json
import logging
import math
import pathlib
import time
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from flash_rt.models.groot_n17.pipeline_sm89 import (
    GrootQwen3VLViTFP16,
    GrootCosmosLLMFP16,
    GrootVLSelfAttnFP16,
    GrootDiTN17FP16,
    VIT_D, VIT_L, VIT_NH, VIT_HD, VIT_SPV, VIT_SPV_RAW,
    LLM_D, LLM_L, LLM_NHQ, LLM_NHKV, LLM_HD,
    VLSA_D, VLSA_L, VLSA_NH, VLSA_HD,
    DIT_D, DIT_L, DIT_NH, DIT_HD, DIT_OUTPUT_DIM,
    ACTION_DIM, STATE_DIM, ACTION_HORIZON_MAX, NUM_FLOW_STEPS,
)

logger = logging.getLogger(__name__)

fp16 = torch.float16

# ── GROOT N1.7 checkpoint key prefixes ──
VIT_PREFIX = "backbone.model.model.visual"
LLM_PREFIX = "backbone.model.model.language_model"
DIT_PREFIX = "action_head.model"
AH_PREFIX = "action_head"

# Embodiment mapping for N1.7
EMBODIMENT_TAG_TO_INDEX = {
    "gr1_unified": 20,
    "gr1": 20,
    "new_embodiment": 0,
}


class GrootN17TorchFrontendSm89:
    """SM89 FP16 GROOT N1.7 torch frontend.
    
    Uses FP16 weights throughout (no FP8 quantization).
    No calibration needed since FP16 doesn't require dynamic scaling.
    CUDA Graph optimized for minimal kernel launch overhead.
    """
    
    def __init__(
        self,
        checkpoint_dir: Union[str, pathlib.Path],
        num_views: int = 2,
        embodiment_tag: str = "gr1_unified",
        action_horizon: int = ACTION_HORIZON_MAX,
        num_flow_steps: int = NUM_FLOW_STEPS,
    ):
        if not (1 <= int(action_horizon) <= ACTION_HORIZON_MAX):
            raise ValueError(
                f"action_horizon must be in [1, {ACTION_HORIZON_MAX}], "
                f"got {action_horizon}")
        
        if not (1 <= int(num_flow_steps) <= 10):
            raise ValueError(
                f"num_flow_steps must be in [1, 10], "
                f"got {num_flow_steps}")
        
        self._checkpoint_dir = pathlib.Path(checkpoint_dir)
        self._num_views = int(num_views)
        self._embodiment_tag = embodiment_tag
        self._num_flow_steps = int(num_flow_steps)
        
        # Load embodiment_id from config
        self._load_embodiment_id()
        
        self._calibrated = True  # FP16 doesn't need calibration
        self._action_horizon = int(action_horizon)
        
        self.latency_records: list[float] = []
        self._graphs_built = False
        self._weight_store = []
        self._cuda_graph = None  # CUDA Graph support
        self._graph_stream = None
        
        # Init kernels
        from flash_rt import flash_rt_kernels as fvk
        self._fvk = fvk
        self._gemm = fvk.GemmRunner()
        self._cudart = ctypes.CDLL("libcudart.so")
        
        # Load checkpoint
        self._load_checkpoint()
        
        logger.info(
            "GrootN17TorchFrontendSm89 initialised (num_views=%d, embodiment=%s id=%d, action_horizon=%d, num_flow_steps=%d)",
            self._num_views, embodiment_tag, self._embodiment_id, self._action_horizon, self._num_flow_steps,
        )
    
    def _load_embodiment_id(self):
        """Load embodiment_id mapping from embodiment_id.json."""
        emb_file = self._checkpoint_dir / "embodiment_id.json"
        if emb_file.exists():
            with open(emb_file) as f:
                emb_map = json.load(f)
            self._embodiment_id = emb_map.get(self._embodiment_tag, 0)
        else:
            self._embodiment_id = EMBODIMENT_TAG_TO_INDEX.get(self._embodiment_tag, 0)
    
    def _load_checkpoint(self) -> None:
        """Load safetensors into GPU memory."""
        from safetensors import safe_open
        
        st_files = sorted(self._checkpoint_dir.glob("*.safetensors"))
        if not st_files:
            raise FileNotFoundError(
                f"No safetensors found in {self._checkpoint_dir}")
        logger.info("Loading %d safetensors files...", len(st_files))
        
        sd = {}
        for f in st_files:
            with safe_open(str(f), framework="pt", device="cuda") as sf:
                for k in sf.keys():
                    sd[k] = sf.get_tensor(k)
        logger.info("Loaded %d tensors", len(sd))
        self._sd = sd
        
        # Load embed_tokens for text embedding
        self._embed_tokens = sd[f"{LLM_PREFIX}.embed_tokens.weight"]
    
    def _build_vit_weights(self) -> dict:
        """Build ViT weights dict (FP16, no FP8 quantization)."""
        sd = self._sd
        store = self._weight_store
        logger.info("Building Qwen3-VL ViT weights (FP16 for SM89)...")
        prefix = f"{VIT_PREFIX}.blocks"
        
        ln1_w_list, ln1_b_list = [], []
        ln2_w_list, ln2_b_list = [], []
        qkv_w_list, qkv_b_list = [], []
        o_w_list, o_b_list = [], []
        fc1_w_list, fc1_b_list = [], []
        fc2_w_list, fc2_b_list = [], []
        
        def _stash(t):
            store.append(t)
            return t
        
        for i in range(VIT_L):
            lp = f"{prefix}.{i}"
            
            # LN weights
            ln1_w_list.append(_stash(sd[f"{lp}.norm1.weight"].to(fp16).contiguous()))
            ln1_b_list.append(_stash(sd[f"{lp}.norm1.bias"].to(fp16).contiguous()))
            ln2_w_list.append(_stash(sd[f"{lp}.norm2.weight"].to(fp16).contiguous()))
            ln2_b_list.append(_stash(sd[f"{lp}.norm2.bias"].to(fp16).contiguous()))
            
            # QKV (FP16)
            qkv_w = sd[f"{lp}.attn.qkv.weight"].T.contiguous().to(fp16)
            _stash(qkv_w)
            qkv_w_list.append(qkv_w.data_ptr())
            
            qkv_b = _stash(sd[f"{lp}.attn.qkv.bias"].to(fp16).contiguous())
            qkv_b_list.append(qkv_b.data_ptr())
            
            # O proj (FP16)
            o_w = sd[f"{lp}.attn.proj.weight"].T.contiguous().to(fp16)
            _stash(o_w)
            o_w_list.append(o_w.data_ptr())
            
            o_b = _stash(sd[f"{lp}.attn.proj.bias"].to(fp16).contiguous())
            o_b_list.append(o_b.data_ptr())
            
            # FFN (FP16)
            fc1_w = sd[f"{lp}.mlp.linear_fc1.weight"].T.contiguous().to(fp16)
            _stash(fc1_w)
            fc1_w_list.append(fc1_w.data_ptr())
            
            fc1_b = _stash(sd[f"{lp}.mlp.linear_fc1.bias"].to(fp16).contiguous())
            fc1_b_list.append(fc1_b.data_ptr())
            
            fc2_w = sd[f"{lp}.mlp.linear_fc2.weight"].T.contiguous().to(fp16)
            _stash(fc2_w)
            fc2_w_list.append(fc2_w.data_ptr())
            
            fc2_b = _stash(sd[f"{lp}.mlp.linear_fc2.bias"].to(fp16).contiguous())
            fc2_b_list.append(fc2_b.data_ptr())
        
        # Patch embedding
        pe_w = _stash(sd[f"{VIT_PREFIX}.patch_embed.proj.weight"].to(fp16).contiguous())
        pe_b = _stash(sd[f"{VIT_PREFIX}.patch_embed.proj.bias"].to(fp16).contiguous())
        pos_emb = _stash(sd[f"{VIT_PREFIX}.pos_embed.weight"].to(fp16).contiguous())
        
        return {
            "patch_embed_w": pe_w.data_ptr(),
            "patch_embed_b": pe_b.data_ptr(),
            "vit_pos_embed": pos_emb.data_ptr(),
            "ln1_w": [w.data_ptr() for w in ln1_w_list],
            "ln1_b": [w.data_ptr() for w in ln1_b_list],
            "ln2_w": [w.data_ptr() for w in ln2_w_list],
            "ln2_b": [w.data_ptr() for w in ln2_b_list],
            "qkv_w": qkv_w_list,
            "qkv_b": qkv_b_list,
            "o_w": o_w_list,
            "o_b": o_b_list,
            "fc1_w": fc1_w_list,
            "fc1_b": fc1_b_list,
            "fc2_w": fc2_w_list,
            "fc2_b": fc2_b_list,
        }
    
    def _build_llm_weights(self) -> dict:
        """Build LLM weights dict (FP16, no FP8 quantization)."""
        sd = self._sd
        store = self._weight_store
        logger.info("Building Qwen3 LLM weights (FP16 for SM89)...")
        prefix = f"{LLM_PREFIX}.layers"
        
        input_ln_w_list = []
        post_ln_w_list = []
        q_norm_w_list = []
        k_norm_w_list = []
        qkv_w_list = []
        o_w_list = []
        gate_w_list = []
        up_w_list = []
        down_w_list = []
        
        def _stash(t):
            store.append(t)
            return t
        
        for i in range(LLM_L):
            lp = f"{prefix}.{i}"
            
            input_ln_w_list.append(_stash(sd[f"{lp}.input_layernorm.weight"].to(fp16).contiguous()).data_ptr())
            post_ln_w_list.append(_stash(sd[f"{lp}.post_attention_layernorm.weight"].to(fp16).contiguous()).data_ptr())
            
            q_norm_w_list.append(_stash(sd[f"{lp}.self_attn.q_norm.weight"].to(fp16).contiguous()).data_ptr())
            k_norm_w_list.append(_stash(sd[f"{lp}.self_attn.k_norm.weight"].to(fp16).contiguous()).data_ptr())
            
            # QKV fused (FP16)
            qkv_T = torch.cat([
                sd[f"{lp}.self_attn.q_proj.weight"],
                sd[f"{lp}.self_attn.k_proj.weight"],
                sd[f"{lp}.self_attn.v_proj.weight"],
            ], dim=0).T.contiguous().to(fp16)
            _stash(qkv_T)
            qkv_w_list.append(qkv_T.data_ptr())
            
            # O proj (FP16)
            o_w = sd[f"{lp}.self_attn.o_proj.weight"].T.contiguous().to(fp16)
            _stash(o_w)
            o_w_list.append(o_w.data_ptr())
            
            # FFN (FP16)
            gate_w = sd[f"{lp}.mlp.gate_proj.weight"].T.contiguous().to(fp16)
            _stash(gate_w)
            gate_w_list.append(gate_w.data_ptr())
            
            up_w = sd[f"{lp}.mlp.up_proj.weight"].T.contiguous().to(fp16)
            _stash(up_w)
            up_w_list.append(up_w.data_ptr())
            
            down_w = sd[f"{lp}.mlp.down_proj.weight"].T.contiguous().to(fp16)
            _stash(down_w)
            down_w_list.append(down_w.data_ptr())
        
        # Final norm
        final_norm_w = _stash(sd[f"{LLM_PREFIX}.norm.weight"].to(fp16).contiguous())
        vlln_w = _stash(sd[f"{AH_PREFIX}.vlln.weight"].to(fp16).contiguous())
        vlln_b = _stash(sd[f"{AH_PREFIX}.vlln.bias"].to(fp16).contiguous())
        
        return {
            "input_ln_w": input_ln_w_list,
            "post_attn_ln_w": post_ln_w_list,
            "q_norm_w": q_norm_w_list,
            "k_norm_w": k_norm_w_list,
            "qkv_w": qkv_w_list,
            "o_w": o_w_list,
            "gate_w": gate_w_list,
            "up_w": up_w_list,
            "down_w": down_w_list,
            "llm_norm_w": final_norm_w.data_ptr(),
            "vlln_w": vlln_w.data_ptr(),
            "vlln_b": vlln_b.data_ptr(),
        }
    
    def _build_vlsa_weights(self) -> dict:
        """Build VL Self Attention weights dict (FP16)."""
        sd = self._sd
        store = self._weight_store
        logger.info("Building VL Self Attention weights (FP16 for SM89)...")
        prefix = f"{AH_PREFIX}.vl_self_attention.transformer_blocks"
        
        norm1_w_list, norm1_b_list = [], []
        norm3_w_list, norm3_b_list = [], []
        q_w_list, q_b_list = [], []
        k_w_list, k_b_list = [], []
        v_w_list, v_b_list = [], []
        o_w_list, o_b_list = [], []
        fc1_w_list, fc1_b_list = [], []
        fc2_w_list, fc2_b_list = [], []
        
        def _stash(t):
            store.append(t)
            return t
        
        for i in range(VLSA_L):
            lp = f"{prefix}.{i}"
            
            norm1_w_list.append(_stash(sd[f"{lp}.norm1.weight"].to(fp16).contiguous()).data_ptr())
            norm1_b_list.append(_stash(sd[f"{lp}.norm1.bias"].to(fp16).contiguous()).data_ptr())
            norm3_w_list.append(_stash(sd[f"{lp}.norm3.weight"].to(fp16).contiguous()).data_ptr())
            norm3_b_list.append(_stash(sd[f"{lp}.norm3.bias"].to(fp16).contiguous()).data_ptr())
            
            # Q, K, V (FP16)
            q_w = sd[f"{lp}.attn1.to_q.weight"].T.contiguous().to(fp16)
            _stash(q_w)
            q_w_list.append(q_w.data_ptr())
            q_b_list.append(_stash(sd[f"{lp}.attn1.to_q.bias"].to(fp16).contiguous()).data_ptr())
            
            k_w = sd[f"{lp}.attn1.to_k.weight"].T.contiguous().to(fp16)
            _stash(k_w)
            k_w_list.append(k_w.data_ptr())
            k_b_list.append(_stash(sd[f"{lp}.attn1.to_k.bias"].to(fp16).contiguous()).data_ptr())
            
            v_w = sd[f"{lp}.attn1.to_v.weight"].T.contiguous().to(fp16)
            _stash(v_w)
            v_w_list.append(v_w.data_ptr())
            v_b_list.append(_stash(sd[f"{lp}.attn1.to_v.bias"].to(fp16).contiguous()).data_ptr())
            
            o_w = sd[f"{lp}.attn1.to_out.0.weight"].T.contiguous().to(fp16)
            _stash(o_w)
            o_w_list.append(o_w.data_ptr())
            o_b_list.append(_stash(sd[f"{lp}.attn1.to_out.0.bias"].to(fp16).contiguous()).data_ptr())
            
            fc1_w = sd[f"{lp}.ff.net.0.proj.weight"].T.contiguous().to(fp16)
            _stash(fc1_w)
            fc1_w_list.append(fc1_w.data_ptr())
            fc1_b_list.append(_stash(sd[f"{lp}.ff.net.0.proj.bias"].to(fp16).contiguous()).data_ptr())
            
            fc2_w = sd[f"{lp}.ff.net.2.weight"].T.contiguous().to(fp16)
            _stash(fc2_w)
            fc2_w_list.append(fc2_w.data_ptr())
            fc2_b_list.append(_stash(sd[f"{lp}.ff.net.2.bias"].to(fp16).contiguous()).data_ptr())
        
        return {
            "norm1_w": norm1_w_list,
            "norm1_b": norm1_b_list,
            "norm3_w": norm3_w_list,
            "norm3_b": norm3_b_list,
            "q_w": q_w_list,
            "q_b": q_b_list,
            "k_w": k_w_list,
            "k_b": k_b_list,
            "v_w": v_w_list,
            "v_b": v_b_list,
            "o_w": o_w_list,
            "o_b": o_b_list,
            "fc1_w": fc1_w_list,
            "fc1_b": fc1_b_list,
            "fc2_w": fc2_w_list,
            "fc2_b": fc2_b_list,
        }
    
    def _build_dit_weights(self) -> dict:
        """Build DiT weights dict (FP16)."""
        sd = self._sd
        store = self._weight_store
        logger.info("Building DiT weights (FP16 for SM89)...")
        prefix = f"{DIT_PREFIX}.transformer_blocks"
        
        q_w_list, q_b_list = [], []
        k_w_list, k_b_list = [], []
        v_w_list, v_b_list = [], []
        o_w_list, o_b_list = [], []
        ff_proj_w_list, ff_proj_b_list = [], []
        ff_down_w_list, ff_down_b_list = [], []
        
        def _stash(t):
            store.append(t)
            return t
        
        for i in range(DIT_L):
            lp = f"{prefix}.{i}"
            
            # Q (FP16)
            q_w = sd[f"{lp}.attn1.to_q.weight"].T.contiguous().to(fp16)
            _stash(q_w)
            q_w_list.append(q_w.data_ptr())
            q_b_list.append(_stash(sd[f"{lp}.attn1.to_q.bias"].to(fp16).contiguous()).data_ptr())
            
            # K (FP16)
            k_w = sd[f"{lp}.attn1.to_k.weight"].T.contiguous().to(fp16)
            _stash(k_w)
            k_w_list.append(k_w.data_ptr())
            k_b_list.append(_stash(sd[f"{lp}.attn1.to_k.bias"].to(fp16).contiguous()).data_ptr())
            
            # V (FP16)
            v_w = sd[f"{lp}.attn1.to_v.weight"].T.contiguous().to(fp16)
            _stash(v_w)
            v_w_list.append(v_w.data_ptr())
            v_b_list.append(_stash(sd[f"{lp}.attn1.to_v.bias"].to(fp16).contiguous()).data_ptr())
            
            # O (FP16)
            o_w = sd[f"{lp}.attn1.to_out.0.weight"].T.contiguous().to(fp16)
            _stash(o_w)
            o_w_list.append(o_w.data_ptr())
            o_b_list.append(_stash(sd[f"{lp}.attn1.to_out.0.bias"].to(fp16).contiguous()).data_ptr())
            
            # FF (FP16)
            ff_proj_w = sd[f"{lp}.ff.net.0.proj.weight"].T.contiguous().to(fp16)
            _stash(ff_proj_w)
            ff_proj_w_list.append(ff_proj_w.data_ptr())
            ff_proj_b_list.append(_stash(sd[f"{lp}.ff.net.0.proj.bias"].to(fp16).contiguous()).data_ptr())
            
            ff_down_w = sd[f"{lp}.ff.net.2.weight"].T.contiguous().to(fp16)
            _stash(ff_down_w)
            ff_down_w_list.append(ff_down_w.data_ptr())
            ff_down_b_list.append(_stash(sd[f"{lp}.ff.net.2.bias"].to(fp16).contiguous()).data_ptr())
        
        # Output projections
        proj_out_2_w = _stash(sd[f"{DIT_PREFIX}.proj_out_2.weight"].T.contiguous().to(fp16))
        proj_out_2_b = _stash(sd[f"{DIT_PREFIX}.proj_out_2.bias"].to(fp16).contiguous())
        
        # Per-embodiment action encoder/decoder weights
        eid = self._embodiment_id
        
        def _emb(name):
            t = sd[f"{AH_PREFIX}.{name}"][eid].contiguous().to(fp16)
            _stash(t)
            return t
        
        def _emb_b(name):
            t = sd[f"{AH_PREFIX}.{name}"][eid].to(fp16).contiguous()
            _stash(t)
            return t
        
        ac_enc_W1 = _emb("action_encoder.W1.W")
        ac_enc_W1_b = _emb_b("action_encoder.W1.b")
        
        ac_dec_l1_W = _emb("action_decoder.layer1.W")
        ac_dec_l1_b = _emb_b("action_decoder.layer1.b")
        ac_dec_l2_W = _emb("action_decoder.layer2.W")
        ac_dec_l2_b = _emb_b("action_decoder.layer2.b")
        
        # Position embedding
        pos_emb = _stash(sd[f"{AH_PREFIX}.position_embedding.weight"].to(fp16).contiguous())
        
        # State encoder
        st_enc_l1_W = _emb("state_encoder.layer1.W")
        st_enc_l1_b = _emb_b("state_encoder.layer1.b")
        st_enc_l2_W = _emb("state_encoder.layer2.W")
        st_enc_l2_b = _emb_b("state_encoder.layer2.b")
        
        self._state_enc_l1_W = st_enc_l1_W
        self._state_enc_l1_b = st_enc_l1_b
        self._state_enc_l2_W = st_enc_l2_W
        self._state_enc_l2_b = st_enc_l2_b
        
        return {
            "q_w": q_w_list,
            "q_b": q_b_list,
            "k_w": k_w_list,
            "k_b": k_b_list,
            "v_w": v_w_list,
            "v_b": v_b_list,
            "o_w": o_w_list,
            "o_b": o_b_list,
            "ff_proj_w": ff_proj_w_list,
            "ff_proj_b": ff_proj_b_list,
            "ff_down_w": ff_down_w_list,
            "ff_down_b": ff_down_b_list,
            "proj_out_2_w": proj_out_2_w.data_ptr(),
            "proj_out_2_b": proj_out_2_b.data_ptr(),
            "ac_enc_W1": ac_enc_W1.data_ptr(),
            "ac_enc_W1_b": ac_enc_W1_b.data_ptr(),
            "ac_dec_l1_W": ac_dec_l1_W.data_ptr(),
            "ac_dec_l1_b": ac_dec_l1_b.data_ptr(),
            "ac_dec_l2_W": ac_dec_l2_W.data_ptr(),
            "ac_dec_l2_b": ac_dec_l2_b.data_ptr(),
            "pos_emb": pos_emb.data_ptr(),
        }
    
    def set_prompt(self, prompt: str) -> None:
        """Tokenize and prepare text embeddings."""
        from transformers import AutoTokenizer
        
        if not hasattr(self, "_tokenizer"):
            tokenizer_candidates = [
                str(self._checkpoint_dir),
                str(self._checkpoint_dir / "tokenizer"),
                "Qwen/Qwen3-1.7B",
            ]
            for tok_path in tokenizer_candidates:
                try:
                    # Use local_files_only to avoid network timeout
                    self._tokenizer = AutoTokenizer.from_pretrained(
                        tok_path, trust_remote_code=True, local_files_only=True)
                    logger.info("Loaded Qwen3 tokenizer from %s", tok_path)
                    break
                except Exception:
                    continue
            if not hasattr(self, "_tokenizer"):
                raise RuntimeError(
                    "Cannot load Qwen3 tokenizer. Pre-download via "
                    "`hf download Qwen/Qwen3-1.7B --include 'tokenizer*'`")
            
            # Qwen3-VL image tokens
            self._img_token_id = 151669
            self._img_start_id = 151670
            self._img_end_id = 151671
        
        S_img = self._num_views * VIT_SPV
        text_ids = self._tokenizer.encode(prompt, add_special_tokens=False)
        full_ids = (text_ids + [self._img_start_id] +
                    [self._img_token_id] * S_img + [self._img_end_id])
        
        new_Se = len(full_ids)
        
        # Check if new prompt length exceeds Se_max when pipeline is already built
        if hasattr(self, '_llm') and new_Se > self._llm.Se_max:
            # Truncate prompt to fit within Se_max
            max_text_len = self._llm.Se_max - S_img - 2  # -2 for start/end tokens
            if max_text_len > 0:
                # Truncate by removing tokens from the end
                text_ids = text_ids[:max_text_len]
                full_ids = (text_ids + [self._img_start_id] +
                            [self._img_token_id] * S_img + [self._img_end_id])
                logger.warning(
                    "Prompt truncated from %d to %d text tokens (Se_max=%d)",
                    len(self._tokenizer.encode(prompt, add_special_tokens=False)),
                    len(text_ids), self._llm.Se_max)
            else:
                raise ValueError(f"Prompt too long: Se={new_Se} exceeds Se_max={self._llm.Se_max}")
        
        self._input_ids = torch.tensor([full_ids], dtype=torch.long, device="cuda")
        self._text_len = len(text_ids)
        self._Se = len(full_ids)
        self._prompt_text = prompt
        
        self._text_embeds = F.embedding(self._input_ids, self._embed_tokens)
        self._image_mask = (self._input_ids == self._img_token_id)
        
        logger.info(
            "Prompt set: '%s' (%d text + %d img = %d tokens)",
            prompt[:50], self._text_len, S_img, self._Se,
        )
    
    def build_pipeline(self) -> None:
        """Build the inference pipeline with CUDA Graph capture."""
        if self._graphs_built:
            return
        
        # Set default prompt if not already set
        # Use longer prompt to ensure larger Se_max for API usage
        if not hasattr(self, '_Se'):
            self.set_prompt("pick up the object from the table and place it")
        
        logger.info("Building N1.7 SM89 FP16 pipeline...")
        
        # Build weights
        vit_weights = self._build_vit_weights()
        llm_weights = self._build_llm_weights()
        vlsa_weights = self._build_vlsa_weights()
        dit_weights = self._build_dit_weights()
        
        Se = self._Se
        T = self._action_horizon
        
        # Create pipeline modules
        self._vit = GrootQwen3VLViTFP16(
            self._gemm, self._fvk, vit_weights, self._num_views)
        
        self._llm = GrootCosmosLLMFP16(
            self._gemm, self._fvk, llm_weights, encoder_seq_max=Se)
        
        self._vlsa = GrootVLSelfAttnFP16(
            self._gemm, self._fvk, vlsa_weights, seq_len=Se)
        
        self._dit = GrootDiTN17FP16(
            self._gemm, self._fvk, dit_weights,
            action_horizon=T, encoder_seq=Se,
            num_flow_steps=self._num_flow_steps)
        
        # Pre-allocate input/output buffers
        self._img_buf = torch.empty(self._num_views, 224, 224, 3, dtype=torch.float16, device="cuda")
        self._state_buf = torch.empty(STATE_DIM, dtype=torch.float16, device="cuda")
        self._noise_buf = torch.empty(self._action_horizon, ACTION_DIM, dtype=torch.float32, device="cuda")
        self._actions_out = torch.empty(self._action_horizon, ACTION_DIM, dtype=torch.float32, device="cuda")
        
        # Warmup for CUDA Graph capture
        self._warmup_for_cuda_graph()
        
        # Capture CUDA Graph
        self._capture_cuda_graph()
        
        self._graphs_built = True
        logger.info("N1.7 Pipeline built with CUDA Graph")
    
    def _warmup_for_cuda_graph(self) -> None:
        """Warmup runs before CUDA Graph capture."""
        # Prepare dummy input data
        dummy_img = torch.zeros(self._num_views, 224, 224, 3, dtype=torch.float16, device="cuda")
        dummy_state = torch.zeros(STATE_DIM, dtype=torch.float16, device="cuda")
        dummy_noise = torch.zeros(self._action_horizon, ACTION_DIM, dtype=torch.float32, device="cuda")
        
        stream = 0
        for _ in range(3):
            # Copy dummy data to pipeline buffers
            self._fvk.gpu_copy(self._vit.bufs["input_images"].ptr.value,
                              dummy_img.data_ptr(), dummy_img.numel() * 2, stream)
            
            # State encoder
            state_feat = F.relu(dummy_state @ self._state_enc_l1_W + self._state_enc_l1_b) @ self._state_enc_l2_W + self._state_enc_l2_b
            self._fvk.gpu_copy(self._dit.bufs["state_feat"].ptr.value, state_feat.data_ptr(), DIT_D * 2, stream)
            
            # Text embeddings
            self._llm.set_seq_len(self._Se)
            self._fvk.gpu_copy(self._llm.bufs["h"].ptr.value, self._text_embeds.data_ptr(), self._Se * LLM_D * 2, stream)
            
            # Noise
            self._fvk.gpu_copy(self._dit.bufs["actions"].ptr.value, dummy_noise.data_ptr(), self._action_horizon * ACTION_DIM * 4, stream)
            
            # Run pipeline with sync for warmup validation
            self._run_pipeline(stream, sync=True)
        
        torch.cuda.synchronize()
        logger.info("CUDA Graph warmup complete")
    
    def _capture_cuda_graph(self) -> None:
        """Capture the pipeline into a CUDA Graph."""
        self._cuda_graph = torch.cuda.CUDAGraph()
        
        # Use a dedicated stream for capture
        self._graph_stream = torch.cuda.Stream()
        
        with torch.cuda.stream(self._graph_stream):
            stream_int = self._graph_stream.cuda_stream
            
            # Begin capture
            self._cuda_graph.capture_begin()
            
            # Run pipeline without sync (sync not allowed during capture)
            self._run_pipeline(stream_int, sync=False)
            
            # End capture
            self._cuda_graph.capture_end()
        
        logger.info("CUDA Graph captured successfully")
    
    def _run_pipeline(self, stream: int, sync: bool = True) -> None:
        """Run full inference pipeline (used for warmup and CUDA Graph capture)."""
        # ViT forward
        self._vit.forward(stream, sync=False)
        
        # Get vision features
        vision_feat_ptr = self._vit.bufs["vision_features"].ptr.value
        
        # LLM forward (with vision features injected)
        self._llm.forward(stream, sync=False)
        
        # Get backbone features
        backbone_feat_ptr = self._llm.bufs["backbone_features"].ptr.value
        
        # VL Self Attention forward
        self._fvk.gpu_copy(
            self._vlsa.bufs["h"].ptr.value, backbone_feat_ptr,
            self._Se * VLSA_D * 2, stream)
        self._vlsa.forward(stream, sync=False)
        
        # Get VL Self Attention output
        vlsa_out_ptr = self._vlsa.bufs["h"].ptr.value
        
        # DiT forward
        self._dit.precompute_cross_kv(vlsa_out_ptr, stream, sync=False)
        self._dit.run_steps(stream, sync=False)
        
        if sync:
            self._cudart.cudaStreamSynchronize(ctypes.c_void_p(stream))
    
    def infer(self, observation: dict) -> dict:
        """Run inference on a single observation using CUDA Graph replay."""
        # Set default prompt if not already set
        if not hasattr(self, '_Se'):
            self.set_prompt("pick up")
        
        if not self._graphs_built:
            self.build_pipeline()
        
        t0 = time.perf_counter()
        
        # Get images
        img1 = observation.get("image")
        img2 = observation.get("wrist_image")
        state = observation.get("state", np.zeros(STATE_DIM, dtype=np.float32))
        
        # Prepare images (FP16, normalized)
        def _prep_img(img):
            if isinstance(img, np.ndarray):
                t = torch.from_numpy(img).to(fp16).cuda()
            else:
                t = img.to(fp16).cuda()
            if t.dtype == torch.uint8:
                t = t.float() / 255.0
            return t
        
        img1_t = _prep_img(img1).contiguous()
        img2_t = _prep_img(img2).contiguous() if img2 is not None else img1_t
        
        # Copy images to pre-allocated buffer
        self._img_buf[0].copy_(img1_t)
        if self._num_views >= 2:
            self._img_buf[1].copy_(img2_t)
        
        # Prepare state (padding to STATE_DIM=132)
        state_np = np.asarray(state, dtype=np.float32)
        if state_np.shape[0] < STATE_DIM:
            state_padded = np.zeros(STATE_DIM, dtype=np.float32)
            state_padded[:state_np.shape[0]] = state_np
        else:
            state_padded = state_np[:STATE_DIM]
        self._state_buf.copy_(torch.from_numpy(state_padded).to(fp16))
        
        # State encoder (FP16)
        state_feat = F.relu(self._state_buf @ self._state_enc_l1_W + self._state_enc_l1_b) @ self._state_enc_l2_W + self._state_enc_l2_b
        
        # Generate noise for diffusion
        self._noise_buf.normal_()
        
        stream = 0
        
        # Copy inputs to pipeline buffers (outside of graph)
        self._fvk.gpu_copy(
            self._vit.bufs["input_images"].ptr.value,
            self._img_buf.data_ptr(), self._img_buf.numel() * 2, stream)
        
        self._fvk.gpu_copy(
            self._dit.bufs["state_feat"].ptr.value,
            state_feat.data_ptr(), DIT_D * 2, stream)
        
        self._llm.set_seq_len(self._Se)
        self._fvk.gpu_copy(
            self._llm.bufs["h"].ptr.value,
            self._text_embeds.data_ptr(), self._Se * LLM_D * 2, stream)
        
        self._fvk.gpu_copy(
            self._dit.bufs["actions"].ptr.value,
            self._noise_buf.data_ptr(), self._action_horizon * ACTION_DIM * 4, stream)
        
        # Replay CUDA Graph (zero kernel launch overhead)
        if self._cuda_graph is not None:
            self._cuda_graph.replay()
        else:
            # Fallback if graph not captured
            self._run_pipeline(stream, sync=True)
        
        # Copy outputs from pipeline (outside of graph)
        self._fvk.gpu_copy(
            self._actions_out.data_ptr(),
            self._dit.bufs["actions"].ptr.value,
            self._action_horizon * ACTION_DIM * 4, stream)
        
        torch.cuda.synchronize()
        
        latency = time.perf_counter() - t0
        self.latency_records.append(latency)
        logger.info("Inference latency: %.3f s", latency)
        
        return {"actions": self._actions_out.cpu().numpy()}


__all__ = ["GrootN17TorchFrontendSm89"]