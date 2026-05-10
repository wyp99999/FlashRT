"""Declarative weight spec for ``GrootN17TorchFrontendSm89``.

SM89 FP16 version - no FP8 quantization, no dynamic scaling.

Covers the full GR00T-N1.7-3B safetensors checkpoint across:
* ViT block       — 24 layers (Qwen3-VL ViT, FP16)
* LLM block       — 16 layers (FP16 GEMM with q_norm/k_norm + M-RoPE)
* vl_self_attn    — 4 layers (BasicTransformerBlock from diffusers, FP16)
* DiT block       — 32 layers (AlternateVLDiT, FP16)
* Singletons      — patch_embed, pos_embed, deepstack mergers, embed_tokens, 
                    vlln, action_encoder/decoder, timestep encoder, etc.

Key differences from _groot_n17_thor_spec.py:
  - No Quant() operations (FP16 only)
  - No scale_into parameters (no dynamic scaling needed)
  - Retains ToFp16() and T() transpose operations
"""

from __future__ import annotations

from flash_rt.executors.weight_loader import Item, LayerBlock, ModelWeightSpec
from flash_rt.executors.torch_weights import (
    Attr,
    Cat,
    T,
    TensorList,
    ToFp16,
)


# ════════════════════════════════════════════════════════════════════
#  Block builders - FP16 only (no Quant)
# ════════════════════════════════════════════════════════════════════

_LP = "backbone.model.model.language_model.layers.{i}"
_VP = "backbone.model.model.visual.blocks.{i}"
_VLSA = "action_head.vl_self_attention.transformer_blocks.{i}"
_DIT = "action_head.model.transformer_blocks.{i}"
_DSM = "backbone.model.model.visual.deepstack_merger_list.{i}"


def _vit_block() -> LayerBlock:
    """Qwen3-VL ViT block: 24 layers × (LN1, QKV+bias, O+bias, LN2, FC1+bias, FC2+bias).
    
    FP16 only for SM89 - no quantization.
    """
    items = [
        Item("ln1_w", f"{_VP}.norm1.weight", [ToFp16()], TensorList("_vit_ln1_w")),
        Item("ln1_b", f"{_VP}.norm1.bias",   [ToFp16()], TensorList("_vit_ln1_b")),

        # Pre-fused QKV [3072, 1024] - FP16 only
        Item("qkv_w", f"{_VP}.attn.qkv.weight",
             [ToFp16(), T()],
             TensorList("_vit_qkv_w")),
        Item("qkv_b", f"{_VP}.attn.qkv.bias",
             [ToFp16()], TensorList("_vit_qkv_b")),

        Item("o_w", f"{_VP}.attn.proj.weight",
             [ToFp16(), T()],
             TensorList("_vit_o_w")),
        Item("o_b", f"{_VP}.attn.proj.bias",
             [ToFp16()], TensorList("_vit_o_b")),

        Item("ln2_w", f"{_VP}.norm2.weight", [ToFp16()], TensorList("_vit_ln2_w")),
        Item("ln2_b", f"{_VP}.norm2.bias",   [ToFp16()], TensorList("_vit_ln2_b")),

        Item("fc1_w", f"{_VP}.mlp.linear_fc1.weight",
             [ToFp16(), T()],
             TensorList("_vit_fc1_w")),
        Item("fc1_b", f"{_VP}.mlp.linear_fc1.bias",
             [ToFp16()], TensorList("_vit_fc1_b")),

        Item("fc2_w", f"{_VP}.mlp.linear_fc2.weight",
             [ToFp16(), T()],
             TensorList("_vit_fc2_w")),
        Item("fc2_b", f"{_VP}.mlp.linear_fc2.bias",
             [ToFp16()], TensorList("_vit_fc2_b")),
    ]
    return LayerBlock(prefix_fmt="", num_layers=24, items=items, name="qwen3vl_vit_fp16")


def _llm_block() -> LayerBlock:
    """Qwen3-VL LLM block (truncated to 16 layers): standard Qwen3 attention + SwiGLU FFN.
    
    FP16 only for SM89.
    GQA: 16Q heads / 8KV heads / head_dim=128.
    """
    items = [
        Item("input_ln_w", f"{_LP}.input_layernorm.weight",
             [ToFp16()], TensorList("_llm_input_ln_w")),

        # Fused QKV: cat(q, k, v) then transpose (FP16)
        Item("qkv_w",
             Cat([f"{_LP}.self_attn.q_proj.weight",
                  f"{_LP}.self_attn.k_proj.weight",
                  f"{_LP}.self_attn.v_proj.weight"], dim=0),
             [ToFp16(), T()],
             TensorList("_llm_qkv_w")),

        # Per-head q_norm / k_norm weights (RMSNorm, shape [128])
        Item("q_norm_w", f"{_LP}.self_attn.q_norm.weight",
             [ToFp16()], TensorList("_llm_q_norm_w")),
        Item("k_norm_w", f"{_LP}.self_attn.k_norm.weight",
             [ToFp16()], TensorList("_llm_k_norm_w")),

        Item("o_w", f"{_LP}.self_attn.o_proj.weight",
             [ToFp16(), T()],
             TensorList("_llm_o_w")),

        Item("post_attn_ln_w", f"{_LP}.post_attention_layernorm.weight",
             [ToFp16()], TensorList("_llm_post_ln_w")),

        # SwiGLU: gate, up, down (FP16)
        Item("gate_w", f"{_LP}.mlp.gate_proj.weight",
             [ToFp16(), T()],
             TensorList("_llm_gate_w")),
        Item("up_w", f"{_LP}.mlp.up_proj.weight",
             [ToFp16(), T()],
             TensorList("_llm_up_w")),
        Item("down_w", f"{_LP}.mlp.down_proj.weight",
             [ToFp16(), T()],
             TensorList("_llm_down_w")),
    ]
    return LayerBlock(prefix_fmt="", num_layers=16, items=items, name="qwen3vl_llm_fp16")


def _vl_self_attn_block() -> LayerBlock:
    """4-layer vl_self_attention: BasicTransformerBlock from diffusers.
    
    FP16 only for SM89.
    Each block: norm1 → self-attn (Q,K,V,O all 2048→2048, no GQA) →
    norm3 → ff (2048→8192→2048, GELU activation, with biases).
    Inner dim = 32 heads × 64 head_dim = 2048.
    """
    items = [
        Item("norm1_w", f"{_VLSA}.norm1.weight", [ToFp16()], TensorList("_vlsa_norm1_w")),
        Item("norm1_b", f"{_VLSA}.norm1.bias",   [ToFp16()], TensorList("_vlsa_norm1_b")),

        # Q, K, V kept SEPARATE (no fusion - diffusers stores them split)
        Item("q_w", f"{_VLSA}.attn1.to_q.weight",
             [ToFp16(), T()],
             TensorList("_vlsa_q_w")),
        Item("q_b", f"{_VLSA}.attn1.to_q.bias",
             [ToFp16()], TensorList("_vlsa_q_b")),
        Item("k_w", f"{_VLSA}.attn1.to_k.weight",
             [ToFp16(), T()],
             TensorList("_vlsa_k_w")),
        Item("k_b", f"{_VLSA}.attn1.to_k.bias",
             [ToFp16()], TensorList("_vlsa_k_b")),
        Item("v_w", f"{_VLSA}.attn1.to_v.weight",
             [ToFp16(), T()],
             TensorList("_vlsa_v_w")),
        Item("v_b", f"{_VLSA}.attn1.to_v.bias",
             [ToFp16()], TensorList("_vlsa_v_b")),

        # Output projection (.to_out.0 in diffusers naming)
        Item("o_w", f"{_VLSA}.attn1.to_out.0.weight",
             [ToFp16(), T()],
             TensorList("_vlsa_o_w")),
        Item("o_b", f"{_VLSA}.attn1.to_out.0.bias",
             [ToFp16()], TensorList("_vlsa_o_b")),

        Item("norm3_w", f"{_VLSA}.norm3.weight", [ToFp16()], TensorList("_vlsa_norm3_w")),
        Item("norm3_b", f"{_VLSA}.norm3.bias",   [ToFp16()], TensorList("_vlsa_norm3_b")),

        # FF: ff.net is [GeGLU/Linear, Dropout, Linear]
        Item("fc1_w", f"{_VLSA}.ff.net.0.proj.weight",
             [ToFp16(), T()],
             TensorList("_vlsa_fc1_w")),
        Item("fc1_b", f"{_VLSA}.ff.net.0.proj.bias",
             [ToFp16()], TensorList("_vlsa_fc1_b")),
        Item("fc2_w", f"{_VLSA}.ff.net.2.weight",
             [ToFp16(), T()],
             TensorList("_vlsa_fc2_w")),
        Item("fc2_b", f"{_VLSA}.ff.net.2.bias",
             [ToFp16()], TensorList("_vlsa_fc2_b")),
    ]
    return LayerBlock(prefix_fmt="", num_layers=4, items=items, name="vl_self_attn_fp16")


def _dit_block() -> LayerBlock:
    """32-layer AlternateVLDiT (interleave_self_attention=True).
    
    FP16 only for SM89.
    Each block has Q (always 1536→1536), K/V whose shape varies per index:
      * Self-attn blocks (odd): K/V = (1536, 1536).
      * Cross-attn blocks (even): K/V = (1536, 2048) — KV from backbone.
    """
    items = [
        # Q always 1536→1536
        Item("q_w", f"{_DIT}.attn1.to_q.weight",
             [ToFp16(), T()],
             TensorList("_dit_q_w")),
        Item("q_b", f"{_DIT}.attn1.to_q.bias",
             [ToFp16()], TensorList("_dit_q_b")),

        # K/V: per-layer shape variance handled implicitly
        Item("k_w", f"{_DIT}.attn1.to_k.weight",
             [ToFp16(), T()],
             TensorList("_dit_k_w")),
        Item("k_b", f"{_DIT}.attn1.to_k.bias",
             [ToFp16()], TensorList("_dit_k_b")),
        Item("v_w", f"{_DIT}.attn1.to_v.weight",
             [ToFp16(), T()],
             TensorList("_dit_v_w")),
        Item("v_b", f"{_DIT}.attn1.to_v.bias",
             [ToFp16()], TensorList("_dit_v_b")),

        Item("o_w", f"{_DIT}.attn1.to_out.0.weight",
             [ToFp16(), T()],
             TensorList("_dit_o_w")),
        Item("o_b", f"{_DIT}.attn1.to_out.0.bias",
             [ToFp16()], TensorList("_dit_o_b")),

        # AdaLN: norm1.linear projects (1536) → 6× shift/scale modulators
        Item("ada_w", f"{_DIT}.norm1.linear.weight",
             [ToFp16(), T()],
             TensorList("_dit_ada_w")),
        Item("ada_b", f"{_DIT}.norm1.linear.bias",
             [ToFp16()], TensorList("_dit_ada_b")),

        # FF.net.0.proj: GeGLU-style projection 1536→6144
        Item("ff_proj_w", f"{_DIT}.ff.net.0.proj.weight",
             [ToFp16(), T()],
             TensorList("_dit_ff_proj_w")),
        Item("ff_proj_b", f"{_DIT}.ff.net.0.proj.bias",
             [ToFp16()], TensorList("_dit_ff_proj_b")),

        # FF.net.2: down 6144→1536
        Item("ff_down_w", f"{_DIT}.ff.net.2.weight",
             [ToFp16(), T()],
             TensorList("_dit_ff_down_w")),
        Item("ff_down_b", f"{_DIT}.ff.net.2.bias",
             [ToFp16()], TensorList("_dit_ff_down_b")),
    ]
    return LayerBlock(prefix_fmt="", num_layers=32, items=items, name="dit_fp16")


def _deepstack_singletons() -> list[Item]:
    """3 DeepStack mergers (tied to ViT layers [5, 11, 17]).
    
    FP16 only for SM89.
    Each merger: norm + linear_fc1 (4096→4096) + linear_fc2 (4096→2048).
    """
    items: list[Item] = []
    for k in range(3):
        p = f"backbone.model.model.visual.deepstack_merger_list.{k}"
        items.extend([
            Item(f"dsm{k}_norm_w", f"{p}.norm.weight", [ToFp16()], Attr(f"_dsm{k}_norm_w")),
            Item(f"dsm{k}_norm_b", f"{p}.norm.bias",   [ToFp16()], Attr(f"_dsm{k}_norm_b")),
            Item(f"dsm{k}_fc1_w",  f"{p}.linear_fc1.weight",
                 [ToFp16(), T()], Attr(f"_dsm{k}_fc1_w")),
            Item(f"dsm{k}_fc1_b",  f"{p}.linear_fc1.bias", [ToFp16()], Attr(f"_dsm{k}_fc1_b")),
            Item(f"dsm{k}_fc2_w",  f"{p}.linear_fc2.weight",
                 [ToFp16(), T()], Attr(f"_dsm{k}_fc2_w")),
            Item(f"dsm{k}_fc2_b",  f"{p}.linear_fc2.bias", [ToFp16()], Attr(f"_dsm{k}_fc2_b")),
        ])
    return items


def _other_singletons() -> list[Item]:
    """ViT patch_embed + pos_embed + final merger + LLM final norm + embed_tokens
    + vlln + DiT proj_out / timestep_encoder + action_head.position_embedding +
    per-embodiment dense matrices.
    
    FP16 only for SM89.
    """
    return [
        # ── ViT input head ──────────────────────────────────────────────
        Item("patch_embed_w", "backbone.model.model.visual.patch_embed.proj.weight",
             [ToFp16()], Attr("_patch_embed_w")),
        Item("patch_embed_b", "backbone.model.model.visual.patch_embed.proj.bias",
             [ToFp16()], Attr("_patch_embed_b")),
        Item("vit_pos_embed", "backbone.model.model.visual.pos_embed.weight",
             [ToFp16()], Attr("_vit_pos_embed")),

        # ── ViT final merger (4096→4096→2048) ───────────────────────────
        Item("merger_norm_w", "backbone.model.model.visual.merger.norm.weight",
             [ToFp16()], Attr("_merger_norm_w")),
        Item("merger_norm_b", "backbone.model.model.visual.merger.norm.bias",
             [ToFp16()], Attr("_merger_norm_b")),
        Item("merger_fc1_w", "backbone.model.model.visual.merger.linear_fc1.weight",
             [ToFp16(), T()], Attr("_merger_fc1_w")),
        Item("merger_fc1_b", "backbone.model.model.visual.merger.linear_fc1.bias",
             [ToFp16()], Attr("_merger_fc1_b")),
        Item("merger_fc2_w", "backbone.model.model.visual.merger.linear_fc2.weight",
             [ToFp16(), T()], Attr("_merger_fc2_w")),
        Item("merger_fc2_b", "backbone.model.model.visual.merger.linear_fc2.bias",
             [ToFp16()], Attr("_merger_fc2_b")),

        # ── LLM final norm + embed_tokens ───────────────────────────────
        Item("llm_norm_w", "backbone.model.model.language_model.norm.weight",
             [ToFp16()], Attr("_llm_norm_w")),
        Item("embed_tokens_w", "backbone.model.model.language_model.embed_tokens.weight",
             [ToFp16()], Attr("_embed_tokens_w")),

        # ── Action-head singletons ──────────────────────────────────────
        Item("vlln_w", "action_head.vlln.weight", [ToFp16()], Attr("_vlln_w")),
        Item("vlln_b", "action_head.vlln.bias",   [ToFp16()], Attr("_vlln_b")),
        Item("ah_pos_embed", "action_head.position_embedding.weight",
             [ToFp16()], Attr("_ah_pos_embed_w")),

        # DiT timestep embedder + final shift/scale + output projector
        Item("ts_lin1_w", "action_head.model.timestep_encoder.timestep_embedder.linear_1.weight",
             [ToFp16(), T()], Attr("_ts_lin1_w")),
        Item("ts_lin1_b", "action_head.model.timestep_encoder.timestep_embedder.linear_1.bias",
             [ToFp16()], Attr("_ts_lin1_b")),
        Item("ts_lin2_w", "action_head.model.timestep_encoder.timestep_embedder.linear_2.weight",
             [ToFp16(), T()], Attr("_ts_lin2_w")),
        Item("ts_lin2_b", "action_head.model.timestep_encoder.timestep_embedder.linear_2.bias",
             [ToFp16()], Attr("_ts_lin2_b")),
        Item("proj_out_1_w", "action_head.model.proj_out_1.weight",
             [ToFp16(), T()], Attr("_proj_out_1_w")),
        Item("proj_out_1_b", "action_head.model.proj_out_1.bias",
             [ToFp16()], Attr("_proj_out_1_b")),
        Item("proj_out_2_w", "action_head.model.proj_out_2.weight",
             [ToFp16(), T()], Attr("_proj_out_2_w")),
        Item("proj_out_2_b", "action_head.model.proj_out_2.bias",
             [ToFp16()], Attr("_proj_out_2_b")),

        # ── Per-embodiment dense matrices (32-slot raw — slice in frontend) ──
        Item("st_enc_l1_W", "action_head.state_encoder.layer1.W",   [ToFp16()], Attr("_st_enc_l1_W")),
        Item("st_enc_l1_b", "action_head.state_encoder.layer1.b",   [ToFp16()], Attr("_st_enc_l1_b")),
        Item("st_enc_l2_W", "action_head.state_encoder.layer2.W",   [ToFp16()], Attr("_st_enc_l2_W")),
        Item("st_enc_l2_b", "action_head.state_encoder.layer2.b",   [ToFp16()], Attr("_st_enc_l2_b")),
        Item("ac_enc_W1_W", "action_head.action_encoder.W1.W",      [ToFp16()], Attr("_ac_enc_W1_W")),
        Item("ac_enc_W1_b", "action_head.action_encoder.W1.b",      [ToFp16()], Attr("_ac_enc_W1_b")),
        Item("ac_enc_W2_W", "action_head.action_encoder.W2.W",      [ToFp16()], Attr("_ac_enc_W2_W")),
        Item("ac_enc_W2_b", "action_head.action_encoder.W2.b",      [ToFp16()], Attr("_ac_enc_W2_b")),
        Item("ac_enc_W3_W", "action_head.action_encoder.W3.W",      [ToFp16()], Attr("_ac_enc_W3_W")),
        Item("ac_enc_W3_b", "action_head.action_encoder.W3.b",      [ToFp16()], Attr("_ac_enc_W3_b")),
        Item("ac_dec_l1_W", "action_head.action_decoder.layer1.W",  [ToFp16()], Attr("_ac_dec_l1_W")),
        Item("ac_dec_l1_b", "action_head.action_decoder.layer1.b",  [ToFp16()], Attr("_ac_dec_l1_b")),
        Item("ac_dec_l2_W", "action_head.action_decoder.layer2.W",  [ToFp16()], Attr("_ac_dec_l2_W")),
        Item("ac_dec_l2_b", "action_head.action_decoder.layer2.b",  [ToFp16()], Attr("_ac_dec_l2_b")),
    ]


# ════════════════════════════════════════════════════════════════════
#  Spec assembly
# ════════════════════════════════════════════════════════════════════

def build_spec() -> ModelWeightSpec:
    """Full N1.7 FP16 WEIGHT_SPEC (singletons + 4 LayerBlocks).
    
    No FP8 quantization, no scale_into parameters.
    """
    return ModelWeightSpec(
        framework="torch",
        blocks=[
            _vit_block(),
            _llm_block(),
            _vl_self_attn_block(),
            _dit_block(),
        ],
        singletons=_deepstack_singletons() + _other_singletons(),
    )


WEIGHT_SPEC = build_spec()


__all__ = ["build_spec", "WEIGHT_SPEC"]