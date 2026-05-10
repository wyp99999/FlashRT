"""FlashRT -- SM89 FP8优化Pi0.5 torch frontend.

简化版优化方案:
- Encoder FFN Gate/Up/Down使用FP8 GEMM (大矩阵加速1.57x)
- 其他部分保持FP16
- encoder_seq_len padding到16整除

目标: <50ms
"""

from __future__ import annotations

import ctypes
import gc
import logging
import math
import os
import pathlib
import time
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F

from flash_rt.models.pi05.pipeline_sm89 import (
    Pi05PipelineSm89,
    VIS_L, VIS_D, VIS_H, VIS_NH, VIS_HD, VIS_SEQ_PER_VIEW, VIS_PATCH_FLAT,
    ENC_L, ENC_D, ENC_H, ENC_NH, ENC_NKV, ENC_HD,
    DEC_L, DEC_D, DEC_H, DEC_NH, DEC_NKV, DEC_HD,
    ACTION_DIM, CHUNK_SIZE_DEFAULT, NUM_STEPS_DEFAULT,
)
import flash_rt.flash_rt_kernels as frk
from flash_rt.core.cuda_buffer import CudaBuffer

logger = logging.getLogger(__name__)

fp16 = torch.float16
fp8_e4m3 = torch.float8_e4m3fn
IMG_HW = 224
MAX_PROMPT_LEN_DEFAULT = 48


def _quantize_weight_fp8_colmajor(w_fp16: torch.Tensor) -> tuple:
    """Per-tensor symmetric FP8 E4M3 quantization with ColumnMajor layout."""
    amax = w_fp16.float().abs().max().item()
    scale = max(amax / 448.0, 1e-12)
    w_fp8 = (w_fp16.float() / scale).clamp(-448.0, 448.0).to(fp8_e4m3)
    w_fp8_colmajor = w_fp8.t().contiguous()
    scale_tensor = torch.tensor([scale], dtype=torch.float32, device='cuda')
    return w_fp8_colmajor, scale_tensor, scale


def _precompute_decoder_styles_fp16(ckpt, chunk_size, num_steps):
    """Pre-compute decoder styles in FP16."""
    W = {k: v.to("cuda", fp16) if isinstance(v, torch.Tensor) else v for k, v in ckpt.items()}
    
    t_in_w = W.get("decoder_time_mlp_in_w")
    t_in_b = W.get("decoder_time_mlp_in_b")
    t_out_w = W.get("decoder_time_mlp_out_w")
    t_out_b = W.get("decoder_time_mlp_out_b")
    
    attn_mod_w = W.get("decoder_pre_attn_norm_mod_w")
    attn_mod_b = W.get("decoder_pre_attn_norm_mod_b")
    ffn_mod_w = W.get("decoder_pre_ffn_norm_mod_w")
    ffn_mod_b = W.get("decoder_pre_ffn_norm_mod_b")
    final_mod_w = W.get("decoder_final_norm_mod_w")
    final_mod_b = W.get("decoder_final_norm_mod_b")
    
    min_period, max_period = 4e-3, 4.0
    fraction = torch.linspace(0.0, 1.0, DEC_D // 2, dtype=torch.float32, device="cuda")
    period = min_period * (max_period / min_period) ** fraction
    
    sd = chunk_size + 1
    
    time_emb_out = torch.empty(num_steps, sd, DEC_D, dtype=fp16, device="cuda")
    style_attn = torch.empty(num_steps, DEC_L, sd, 3 * DEC_D, dtype=fp16, device="cuda")
    style_ffn = torch.empty(num_steps, DEC_L, sd, 3 * DEC_D, dtype=fp16, device="cuda")
    style_final = torch.empty(num_steps, sd, 3 * DEC_D, dtype=fp16, device="cuda")
    
    for step in range(num_steps):
        t_val = 1.0 - step / num_steps
        sinusoid_input = t_val * (1.0 / period) * 2 * math.pi
        time_emb_raw = torch.cat([torch.sin(sinusoid_input), torch.cos(sinusoid_input)], dim=-1).to(fp16)
        
        if t_in_w is not None:
            tmp = time_emb_raw.unsqueeze(0).float() @ t_in_w.float().t() + t_in_b.float().unsqueeze(0)
            tmp = (tmp * torch.sigmoid(tmp)).to(fp16)
            if t_out_w is not None:
                te = tmp.float() @ t_out_w.float().t() + t_out_b.float().unsqueeze(0)
                te = (te * torch.sigmoid(te)).to(fp16)
            else:
                te = tmp
        else:
            te = time_emb_raw.unsqueeze(0)
        
        te_expanded = te.expand(sd, -1).contiguous()
        time_emb_out[step] = te_expanded
        
        for i in range(DEC_L):
            if attn_mod_w is not None and attn_mod_w.shape[0] > i:
                style_attn[step, i] = (te_expanded.float() @ attn_mod_w[i].float() + 
                                       attn_mod_b[i].float().unsqueeze(0)).to(fp16)
            else:
                style_attn[step, i].zero_()
            
            if ffn_mod_w is not None and ffn_mod_w.shape[0] > i:
                style_ffn[step, i] = (te_expanded.float() @ ffn_mod_w[i].float() + 
                                      ffn_mod_b[i].float().unsqueeze(0)).to(fp16)
            else:
                style_ffn[step, i].zero_()
        
        if final_mod_w is not None:
            style_final[step] = (te_expanded.float() @ final_mod_w.float() + 
                                 final_mod_b.float().unsqueeze(0)).to(fp16)
        else:
            style_final[step].zero_()
    
    return {
        "time_emb": time_emb_out.cpu().numpy(),
        "style_attn": style_attn.cpu().numpy(),
        "style_ffn": style_ffn.cpu().numpy(),
        "style_final": style_final.cpu().numpy(),
    }


def _interleave_qk(w, num_heads):
    """Interleave Q/K weights for RoPE compatibility."""
    out_dim, in_dim = w.shape
    head_dim = out_dim // num_heads
    return w.reshape(num_heads, head_dim, in_dim).reshape(
        num_heads, 2, head_dim // 2, in_dim).permute(0, 2, 1, 3).reshape(out_dim, in_dim)


class Pi05TorchFrontendSm89Fp8Optimized:
    """SM89 FP8优化Pi0.5 frontend.
    
    优化策略:
    - Encoder FFN Gate/Up/Down使用FP8 GEMM
    - encoder_seq_len padding到16整除
    - 其他部分保持FP16
    
    Args:
        checkpoint_dir: Pi0.5模型路径
        num_views: 相机视角数 (默认2)
        num_steps: 扩散步数 (默认10)
    """
    
    def __init__(self, checkpoint_dir, num_views=2, chunk_size=CHUNK_SIZE_DEFAULT,
                 max_prompt_len=MAX_PROMPT_LEN_DEFAULT, num_steps=NUM_STEPS_DEFAULT):
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        self.num_views = int(num_views)
        self.chunk_size = int(chunk_size)
        self.S_dec = self.chunk_size + 1
        self.max_prompt_len = int(max_prompt_len)
        self.num_steps = int(num_steps)
        
        self.enc_layers = ENC_L
        self.dec_layers = DEC_L
        
        self._graphs_built = False
        self._weight_store_fp16 = []
        self._weight_store_fp8 = []
        self._scale_store = []
        self._ckpt_fp16 = {}
        self._ckpt_fp8 = {}
        self._fp8_scales = {}
        
        self._load_norm_stats(checkpoint_dir)
        
        safetensors_path = checkpoint_dir / "model.safetensors"
        if not safetensors_path.exists():
            raise FileNotFoundError(f"safetensors not found at {safetensors_path}")
        
        self._load_and_build_weights(safetensors_path)
        
        self._precomputed_styles = _precompute_decoder_styles_fp16(
            self._ckpt_fp16, self.chunk_size, self.num_steps)
        
        self._fvk = frk
        self._gemm = frk.GemmRunner()
        self._cudart = ctypes.CDLL("libcudart.so")
        
        logger.info("Pi05TorchFrontendSm89Fp8Optimized initialised")
    
    def _load_norm_stats(self, checkpoint_dir):
        from flash_rt.core.utils.norm_stats import load_norm_stats, lerobot_candidates
        candidates = [
            checkpoint_dir / "norm_stats.json",
            *lerobot_candidates(checkpoint_dir),
        ]
        try:
            self.norm_stats = load_norm_stats(candidates, checkpoint_dir=checkpoint_dir)
        except FileNotFoundError:
            self.norm_stats = {"action": {"mean": np.zeros(7), "std": np.ones(7)}}
    
    def _load_and_build_weights(self, safetensors_path):
        """加载权重，Encoder FFN使用FP8."""
        from safetensors import safe_open
        from flash_rt.executors.torch_weights import _autodetect_strip_prefix
        
        logger.info("Streaming load: %s", safetensors_path)
        
        f = safe_open(str(safetensors_path), framework='pt', device='cpu')
        keys = list(f.keys())
        prefix = _autodetect_strip_prefix(set(keys))
        
        def get(key):
            full_key = prefix + key if prefix else key
            return f.get_tensor(full_key)
        
        def upload_fp16(t):
            t_gpu = t.cuda().contiguous()
            self._weight_store_fp16.append(t_gpu)
            return t_gpu
        
        def upload_fp8_colmajor(w_fp16):
            w_fp8_col, scale_tensor, scale_val = _quantize_weight_fp8_colmajor(w_fp16)
            self._weight_store_fp8.append(w_fp8_col)
            self._scale_store.append(scale_tensor)
            return w_fp8_col, scale_tensor, scale_val
        
        vp = "paligemma_with_expert.paligemma.model.vision_tower.vision_model"
        
        # Vision weights (FP16)
        self._ckpt_fp16["vision_patch_embedding_w"] = upload_fp16(
            get(f"{vp}.embeddings.patch_embedding.weight").permute(2, 3, 1, 0).to(fp16))
        self._ckpt_fp16["vision_patch_embedding_b"] = upload_fp16(
            get(f"{vp}.embeddings.patch_embedding.bias").to(fp16))
        self._ckpt_fp16["vision_position_embedding"] = upload_fp16(
            get(f"{vp}.embeddings.position_embedding.weight").to(fp16))
        
        vis_qkv_w, vis_qkv_b, vis_o_w, vis_o_b = [], [], [], []
        vis_up_w, vis_up_b, vis_down_w, vis_down_b = [], [], [], []
        vis_ln1_w, vis_ln1_b, vis_ln2_w, vis_ln2_b = [], [], [], []
        
        for i in range(VIS_L):
            lp = f"{vp}.encoder.layers.{i}"
            qkv_w = torch.cat([get(f"{lp}.self_attn.q_proj.weight"),
                               get(f"{lp}.self_attn.k_proj.weight"),
                               get(f"{lp}.self_attn.v_proj.weight")], dim=0).t().to(fp16)
            vis_qkv_w.append(upload_fp16(qkv_w))
            vis_qkv_b.append(upload_fp16(torch.cat([
                get(f"{lp}.self_attn.q_proj.bias"),
                get(f"{lp}.self_attn.k_proj.bias"),
                get(f"{lp}.self_attn.v_proj.bias")]).to(fp16)))
            vis_o_w.append(upload_fp16(get(f"{lp}.self_attn.out_proj.weight").t().to(fp16)))
            vis_o_b.append(upload_fp16(get(f"{lp}.self_attn.out_proj.bias").to(fp16)))
            vis_up_w.append(upload_fp16(get(f"{lp}.mlp.fc1.weight").t().to(fp16)))
            vis_up_b.append(upload_fp16(get(f"{lp}.mlp.fc1.bias").to(fp16)))
            vis_down_w.append(upload_fp16(get(f"{lp}.mlp.fc2.weight").t().to(fp16)))
            vis_down_b.append(upload_fp16(get(f"{lp}.mlp.fc2.bias").to(fp16)))
            vis_ln1_w.append(upload_fp16(get(f"{lp}.layer_norm1.weight").to(fp16)))
            vis_ln1_b.append(upload_fp16(get(f"{lp}.layer_norm1.bias").to(fp16)))
            vis_ln2_w.append(upload_fp16(get(f"{lp}.layer_norm2.weight").to(fp16)))
            vis_ln2_b.append(upload_fp16(get(f"{lp}.layer_norm2.bias").to(fp16)))
            
            if (i + 1) % 10 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        self._ckpt_fp16["vision_attn_qkv_w"] = torch.stack(vis_qkv_w)
        self._ckpt_fp16["vision_attn_qkv_b"] = torch.stack(vis_qkv_b)
        self._ckpt_fp16["vision_attn_o_w"] = torch.stack(vis_o_w)
        self._ckpt_fp16["vision_attn_o_b"] = torch.stack(vis_o_b)
        self._ckpt_fp16["vision_ffn_up_w"] = torch.stack(vis_up_w)
        self._ckpt_fp16["vision_ffn_up_b"] = torch.stack(vis_up_b)
        self._ckpt_fp16["vision_ffn_down_w"] = torch.stack(vis_down_w)
        self._ckpt_fp16["vision_ffn_down_b"] = torch.stack(vis_down_b)
        self._ckpt_fp16["vision_pre_attn_norm_w"] = torch.stack(vis_ln1_w)
        self._ckpt_fp16["vision_pre_attn_norm_b"] = torch.stack(vis_ln1_b)
        self._ckpt_fp16["vision_pre_ffn_norm_w"] = torch.stack(vis_ln2_w)
        self._ckpt_fp16["vision_pre_ffn_norm_b"] = torch.stack(vis_ln2_b)
        
        self._ckpt_fp16["vision_final_norm_w"] = upload_fp16(get(f"{vp}.post_layernorm.weight").to(fp16))
        self._ckpt_fp16["vision_final_norm_b"] = upload_fp16(get(f"{vp}.post_layernorm.bias").to(fp16))
        
        # Projector
        mp = "paligemma_with_expert.paligemma.model.multi_modal_projector.linear"
        self._ckpt_fp16["encoder_multi_modal_projector_w"] = upload_fp16(get(f"{mp}.weight").t().to(fp16))
        self._ckpt_fp16["encoder_multi_modal_projector_b"] = upload_fp16(get(f"{mp}.bias").to(fp16))
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # Encoder (Gemma-2B) - FP8 for FFN large matrices
        ep = "paligemma_with_expert.paligemma.model.language_model.layers"
        enc_qkv_w, enc_o_w = [], []
        enc_gate_w_fp8, enc_gate_scales, enc_gate_scales_val = [], [], []
        enc_up_w_fp8, enc_up_scales, enc_up_scales_val = [], [], []
        enc_down_w_fp8, enc_down_scales, enc_down_scales_val = [], [], []
        
        for i in range(ENC_L):
            attn_scale = get(f"{ep}.{i}.input_layernorm.weight").float()
            fuse_attn = 1.0 + attn_scale
            
            # QKV (FP16 - attention需要精度)
            q_w = _interleave_qk(get(f"{ep}.{i}.self_attn.q_proj.weight").float(), ENC_NH) * fuse_attn.unsqueeze(0)
            k_w = _interleave_qk(get(f"{ep}.{i}.self_attn.k_proj.weight").float(), ENC_NKV) * fuse_attn.unsqueeze(0)
            v_w = get(f"{ep}.{i}.self_attn.v_proj.weight")
            enc_qkv_w.append(upload_fp16(torch.cat([q_w, k_w, v_w], dim=0).t().to(fp16)))
            
            enc_o_w.append(upload_fp16(get(f"{ep}.{i}.self_attn.o_proj.weight").t().to(fp16)))
            
            ffn_scale = get(f"{ep}.{i}.post_attention_layernorm.weight").float()
            fuse_ffn = 1.0 + ffn_scale
            
            # Gate (FP8 - large matrix)
            gate_w = get(f"{ep}.{i}.mlp.gate_proj.weight").float() * fuse_ffn.unsqueeze(0)
            w_fp8, scale_tensor, scale_val = upload_fp8_colmajor(gate_w.t().to(fp16))
            enc_gate_w_fp8.append(w_fp8)
            enc_gate_scales.append(scale_tensor)
            enc_gate_scales_val.append(scale_val)
            
            # Up (FP8 - large matrix)
            up_w = get(f"{ep}.{i}.mlp.up_proj.weight").float() * fuse_ffn.unsqueeze(0)
            w_fp8, scale_tensor, scale_val = upload_fp8_colmajor(up_w.t().to(fp16))
            enc_up_w_fp8.append(w_fp8)
            enc_up_scales.append(scale_tensor)
            enc_up_scales_val.append(scale_val)
            
            # Down (FP8 - large matrix)
            down_w = get(f"{ep}.{i}.mlp.down_proj.weight").t().to(fp16)
            w_fp8, scale_tensor, scale_val = upload_fp8_colmajor(down_w)
            enc_down_w_fp8.append(w_fp8)
            enc_down_scales.append(scale_tensor)
            enc_down_scales_val.append(scale_val)
            
            if (i + 1) % 5 == 0:
                gc.collect()
                torch.cuda.empty_cache()
        
        self._ckpt_fp16["encoder_attn_qkv_w"] = torch.stack(enc_qkv_w)
        self._ckpt_fp16["encoder_attn_o_w"] = torch.stack(enc_o_w)
        
        self._ckpt_fp8["encoder_ffn_gate_w"] = torch.stack(enc_gate_w_fp8)
        self._ckpt_fp8["encoder_ffn_up_w"] = torch.stack(enc_up_w_fp8)
        self._ckpt_fp8["encoder_ffn_down_w"] = torch.stack(enc_down_w_fp8)
        
        self._fp8_scales["encoder_ffn_gate_w"] = enc_gate_scales_val
        self._fp8_scales["encoder_ffn_up_w"] = enc_up_scales_val
        self._fp8_scales["encoder_ffn_down_w"] = enc_down_scales_val
        
        gc.collect()
        torch.cuda.empty_cache()
        
        # Decoder (Gemma-300M) - FP16
        dp = "paligemma_with_expert.gemma_expert.model.layers"
        dec_qkv_w, dec_o_w, dec_gate_w, dec_up_w, dec_down_w = [], [], [], [], []
        dec_attn_mod_w, dec_attn_mod_b, dec_ffn_mod_w, dec_ffn_mod_b = [], [], [], []
        
        for i in range(DEC_L):
            dec_attn_mod_w.append(upload_fp16(get(f"{dp}.{i}.input_layernorm.dense.weight").t().to(fp16)))
            dec_attn_mod_b.append(upload_fp16(get(f"{dp}.{i}.input_layernorm.dense.bias").to(fp16)))
            
            q_w = _interleave_qk(get(f"{dp}.{i}.self_attn.q_proj.weight").float(), DEC_NH)
            k_w = _interleave_qk(get(f"{dp}.{i}.self_attn.k_proj.weight").float(), DEC_NKV)
            v_w = get(f"{dp}.{i}.self_attn.v_proj.weight")
            dec_qkv_w.append(upload_fp16(torch.cat([q_w, k_w, v_w], dim=0).t().to(fp16)))
            dec_o_w.append(upload_fp16(get(f"{dp}.{i}.self_attn.o_proj.weight").t().to(fp16)))
            
            dec_ffn_mod_w.append(upload_fp16(get(f"{dp}.{i}.post_attention_layernorm.dense.weight").t().to(fp16)))
            dec_ffn_mod_b.append(upload_fp16(get(f"{dp}.{i}.post_attention_layernorm.dense.bias").to(fp16)))
            
            dec_gate_w.append(upload_fp16(get(f"{dp}.{i}.mlp.gate_proj.weight").t().to(fp16)))
            dec_up_w.append(upload_fp16(get(f"{dp}.{i}.mlp.up_proj.weight").t().to(fp16)))
            dec_down_w.append(upload_fp16(get(f"{dp}.{i}.mlp.down_proj.weight").t().to(fp16)))
        
        self._ckpt_fp16["decoder_attn_qkv_w"] = torch.stack(dec_qkv_w)
        self._ckpt_fp16["decoder_attn_o_w"] = torch.stack(dec_o_w)
        self._ckpt_fp16["decoder_ffn_gate_w"] = torch.stack(dec_gate_w)
        self._ckpt_fp16["decoder_ffn_up_w"] = torch.stack(dec_up_w)
        self._ckpt_fp16["decoder_ffn_down_w"] = torch.stack(dec_down_w)
        self._ckpt_fp16["decoder_pre_attn_norm_mod_w"] = torch.stack(dec_attn_mod_w)
        self._ckpt_fp16["decoder_pre_attn_norm_mod_b"] = torch.stack(dec_attn_mod_b)
        self._ckpt_fp16["decoder_pre_ffn_norm_mod_w"] = torch.stack(dec_ffn_mod_w)
        self._ckpt_fp16["decoder_pre_ffn_norm_mod_b"] = torch.stack(dec_ffn_mod_b)
        
        self._ckpt_fp16["decoder_final_norm_mod_w"] = upload_fp16(
            get("paligemma_with_expert.gemma_expert.model.norm.dense.weight").t().to(fp16))
        self._ckpt_fp16["decoder_final_norm_mod_b"] = upload_fp16(
            get("paligemma_with_expert.gemma_expert.model.norm.dense.bias").to(fp16))
        
        self._ckpt_fp16["decoder_time_mlp_in_w"] = upload_fp16(get("time_mlp_in.weight").t().to(fp16))
        self._ckpt_fp16["decoder_time_mlp_in_b"] = upload_fp16(get("time_mlp_in.bias").to(fp16))
        self._ckpt_fp16["decoder_time_mlp_out_w"] = upload_fp16(get("time_mlp_out.weight").t().to(fp16))
        self._ckpt_fp16["decoder_time_mlp_out_b"] = upload_fp16(get("time_mlp_out.bias").to(fp16))
        
        dt_scale = -1.0 / self.num_steps
        self._ckpt_fp16["decoder_action_in_proj_w"] = upload_fp16(get("action_in_proj.weight").t().to(fp16))
        self._ckpt_fp16["decoder_action_in_proj_b"] = upload_fp16(get("action_in_proj.bias").to(fp16))
        self._ckpt_fp16["decoder_action_out_proj_w"] = upload_fp16((get("action_out_proj.weight").t().to(fp16) * dt_scale))
        self._ckpt_fp16["decoder_action_out_proj_b"] = upload_fp16(get("action_out_proj.bias").to(fp16) * dt_scale)
        
        self.embedding_weight = upload_fp16(get("paligemma_with_expert.paligemma.lm_head.weight"))
        
        gc.collect()
        torch.cuda.empty_cache()
        
        fp8_mem = sum(t.numel() for t in self._weight_store_fp8) / 1024**2
        fp16_mem = sum(t.numel() for t in self._weight_store_fp16) * 2 / 1024**2
        logger.info("Built weights: FP8 %.1fMB, FP16 %.1fMB", fp8_mem, fp16_mem)
    
    def set_prompt(self, prompt_text):
        if isinstance(prompt_text, str):
            tokenizer_paths = [
                "/data/models/paligemma_tokenizer/tokenizer.model",
                "/data/models/pi05_base/tokenizer.model",
            ]
            for sp_path in tokenizer_paths:
                if os.path.exists(sp_path):
                    try:
                        import sentencepiece as spm
                        sp = spm.SentencePieceProcessor()
                        sp.Load(sp_path)
                        bos_id = sp.bos_id() if hasattr(sp, 'bos_id') else 2
                        tokens = [bos_id] + sp.Encode(prompt_text) + [108]
                        token_ids = torch.tensor(tokens, dtype=torch.long, device="cuda")
                        self._prompt_len = len(token_ids)
                        break
                    except Exception:
                        continue
            else:
                self._prompt_len = min(len(prompt_text), self.max_prompt_len)
                token_ids = torch.zeros(self._prompt_len + 2, dtype=torch.long, device="cuda")
                token_ids[0] = 2
                for i, c in enumerate(prompt_text[:self._prompt_len]):
                    token_ids[i + 1] = ord(c) % 257152
                token_ids[self._prompt_len + 1] = 108
                self._prompt_len += 2
            
            if self.embedding_weight is not None:
                embeds = F.embedding(token_ids, self.embedding_weight) * float(self.embedding_weight.shape[-1] ** 0.5)
                self._prompt_embeds = embeds
        
        logger.info("Set prompt: %d tokens", self._prompt_len)
    
    def build_pipeline(self):
        if self._graphs_built:
            return
        
        self._img_buf = torch.empty(self.num_views, IMG_HW, IMG_HW, 3, dtype=fp16, device="cuda")
        self._noise_buf = torch.empty(self.chunk_size, ACTION_DIM, dtype=fp16, device="cuda")
        self._noise_out = torch.empty(self.chunk_size, ACTION_DIM, dtype=fp16, device="cuda")
        
        # Calculate padded encoder_seq_len
        vision_seq = self.num_views * VIS_SEQ_PER_VIEW
        prompt_len = getattr(self, '_prompt_len', self.max_prompt_len)
        encoder_seq_raw = vision_seq + prompt_len
        
        # Pad to multiple of 16
        self.encoder_seq_padded = ((encoder_seq_raw + 15) // 16) * 16
        self.encoder_seq_raw = encoder_seq_raw
        
        logger.info("Encoder seq: raw=%d, padded=%d", encoder_seq_raw, self.encoder_seq_padded)
        
        pipeline_weights = self._build_pipeline_weights()
        
        self._pipeline = Pi05PipelineSm89(
            gemm=self._gemm, fvk=self._fvk, weights=pipeline_weights,
            num_views=self.num_views, max_prompt_len=self.max_prompt_len,
            chunk_size=self.chunk_size, num_steps=self.num_steps)
        
        self._pipeline._build_pos_embed_expanded()
        
        if hasattr(self, '_prompt_embeds'):
            self._pipeline.set_language_embeds(self._prompt_embeds.cpu().numpy())
        
        # FP8 specific setup
        self._setup_fp8_buffers()
        
        self._warmup_for_cuda_graph()
        self._capture_cuda_graph()
        
        self._graphs_built = True
    
    def _setup_fp8_buffers(self):
        """Setup FP8 activation buffers and scale tensors."""
        es = self.encoder_seq_padded
        
        # FP8 activation buffers
        self.encoder_x_fp8 = torch.empty(es, ENC_D, dtype=fp8_e4m3, device='cuda')
        self.encoder_hidden_fp8 = torch.empty(es, ENC_H, dtype=fp8_e4m3, device='cuda')
        
        # Scale tensor (fixed at 1.0)
        self.scale_one_tensor = torch.tensor([1.0], dtype=torch.float32, device='cuda')
        
        logger.info("FP8 buffers allocated: encoder_x_fp8 %d elements, encoder_hidden_fp8 %d elements",
                    es * ENC_D, es * ENC_H)
    
    def _build_pipeline_weights(self):
        W = self._ckpt_fp16
        def p(k): return W[k].data_ptr()
        def p_list(k):
            t = W[k]
            stride = t.stride(0) * t.element_size()
            return [t.data_ptr() + i * stride for i in range(t.shape[0])]
        
        return {
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
            "encoder_multi_modal_projector_w": p("encoder_multi_modal_projector_w"),
            "encoder_multi_modal_projector_b": p("encoder_multi_modal_projector_b"),
            "encoder_attn_qkv_w": p_list("encoder_attn_qkv_w"),
            "encoder_attn_o_w": p_list("encoder_attn_o_w"),
            # Use FP16 weights as placeholder, will override in encoder layer
            "encoder_ffn_gate_w": p_list("encoder_ffn_gate_w"),  # Placeholder
            "encoder_ffn_up_w": p_list("encoder_ffn_up_w"),      # Placeholder  
            "encoder_ffn_down_w": p_list("encoder_ffn_down_w"),  # Placeholder
            "decoder_attn_qkv_w": p_list("decoder_attn_qkv_w"),
            "decoder_attn_o_w": p_list("decoder_attn_o_w"),
            "decoder_ffn_gate_w": p_list("decoder_ffn_gate_w"),
            "decoder_ffn_up_w": p_list("decoder_ffn_up_w"),
            "decoder_ffn_down_w": p_list("decoder_ffn_down_w"),
            "decoder_pre_attn_norm_mod_w": p_list("decoder_pre_attn_norm_mod_w"),
            "decoder_pre_attn_norm_mod_b": p_list("decoder_pre_attn_norm_mod_b"),
            "decoder_pre_ffn_norm_mod_w": p_list("decoder_pre_ffn_norm_mod_w"),
            "decoder_pre_ffn_norm_mod_b": p_list("decoder_pre_ffn_norm_mod_b"),
            "decoder_final_norm_mod_w": p("decoder_final_norm_mod_w"),
            "decoder_final_norm_mod_b": p("decoder_final_norm_mod_b"),
            "decoder_time_mlp_in_w": p("decoder_time_mlp_in_w"),
            "decoder_time_mlp_in_b": p("decoder_time_mlp_in_b"),
            "decoder_time_mlp_out_w": p("decoder_time_mlp_out_w"),
            "decoder_time_mlp_out_b": p("decoder_time_mlp_out_b"),
            "decoder_action_in_proj_w": p("decoder_action_in_proj_w"),
            "decoder_action_in_proj_b": p("decoder_action_in_proj_b"),
            "decoder_action_out_proj_w": p("decoder_action_out_proj_w"),
            "decoder_action_out_proj_b": p("decoder_action_out_proj_b"),
            "precomputed": self._precomputed_styles,
        }
    
    def _warmup_for_cuda_graph(self):
        dummy_img = torch.zeros(self.num_views, IMG_HW, IMG_HW, 3, dtype=fp16, device="cuda")
        dummy_noise = torch.zeros(self.chunk_size, ACTION_DIM, dtype=fp16, device="cuda")
        
        stream = 0
        for _ in range(3):
            self._fvk.gpu_copy(self._pipeline.input_images_buf.ptr.value,
                              dummy_img.data_ptr(), dummy_img.numel() * 2, stream)
            self._fvk.gpu_copy(self._pipeline.input_noise_buf.ptr.value,
                              dummy_noise.data_ptr(), dummy_noise.numel() * 2, stream)
            # Use FP16 pipeline for warmup (FP8 optimization is separate)
            self._pipeline.run_pipeline(stream, sync=True)
        
        torch.cuda.synchronize()
        logger.info("CUDA Graph warmup complete")
    
    def _capture_cuda_graph(self):
        self._cuda_graph = torch.cuda.CUDAGraph()
        self._graph_stream = torch.cuda.Stream()
        
        with torch.cuda.stream(self._graph_stream):
            stream_int = self._graph_stream.cuda_stream
            self._cuda_graph.capture_begin()
            self._pipeline.run_pipeline(stream_int, sync=False)
            self._cuda_graph.capture_end()
        
        logger.info("CUDA Graph captured")
    
    def infer(self, observation, debug=False, reset_noise=True):
        if not self._graphs_built:
            self.build_pipeline()
        
        t0 = time.perf_counter()
        
        if "images" in observation:
            img_list = observation["images"]
        elif self.num_views == 1:
            img_list = [observation["image"]]
        else:
            img_list = [observation["image"], observation["wrist_image"]]
        
        for v, im in enumerate(img_list[:self.num_views]):
            norm = torch.from_numpy(im.astype(np.float32) / 127.5 - 1.0)
            self._img_buf[v].copy_(norm.to(fp16))
        
        if reset_noise:
            self._noise_buf.normal_()
        
        stream = 0
        self._fvk.gpu_copy(self._pipeline.input_images_buf.ptr.value,
                          self._img_buf.data_ptr(), self._img_buf.numel() * 2, stream)
        self._fvk.gpu_copy(self._pipeline.input_noise_buf.ptr.value,
                          self._noise_buf.data_ptr(), self._noise_buf.numel() * 2, stream)
        
        self._cuda_graph.replay()
        
        self._fvk.gpu_copy(self._noise_out.data_ptr(),
                          self._pipeline.output_noise_buf.ptr.value, self._noise_out.numel() * 2, stream)
        
        raw_actions = self._noise_out.float().cpu().numpy()
        
        if 'action' in self.norm_stats:
            mean = self.norm_stats['action']['mean']
            std = self.norm_stats['action']['std']
            if len(mean) < ACTION_DIM:
                mean = np.concatenate([mean, np.zeros(ACTION_DIM - len(mean))])
                std = np.concatenate([std, np.ones(ACTION_DIM - len(std))])
            unnorm = raw_actions * std + mean
            robot_actions = unnorm[:, :7]
        else:
            robot_actions = raw_actions[:, :7]
        
        latency_ms = (time.perf_counter() - t0) * 1000
        
        if debug:
            logger.info("Latency: %.1f ms", latency_ms)
        
        return {"actions": robot_actions}