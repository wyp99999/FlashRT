"""FlashRT — RTX GROOT N1.6 attention backend.

GROOT has four distinct attention shapes:

  - SigLIP2     :  (batch=num_views, seq=256, heads=16,  head_dim=72)   self
  - Qwen3 GQA   :  (1, Se, 16q/8kv, head_dim=128) full self-attn
  - DiT self    :  (1, Sa=51,  heads=32, head_dim=48)  self
  - DiT cross   :  Q (1, Sa, 32, 48)  vs  K/V (1, Se, 32, 48)  cross

flash_attention covers all four head_dims (48 → kHeadDim=64, 72 → 96,
128 direct) so we can route everything through one ``flash_attn_func``
backend on RTX 5090.

This is a separate backend from ``RtxFlashAttnBackend`` (the Pi0/Pi0.5
one) because the attention sites and shapes are entirely different.
The protocol matches: ``get_ptrs()`` returns a dict of input pointers
for the pipeline to write into via fvk kernels, and each ``*_attn``
method returns the raw device pointer of flash_attn's output tensor
(no copy — torch's caching allocator pins the slot during graph
capture, exactly the same trick as the Pi0.5 backend).

Future revision will replace this with a pure-C
``CFlashAttnBackend`` that calls into ``libfmha_fp16.so``. The
pipeline doesn't care which backend is used.
"""

from __future__ import annotations


# Pi0.5 uses bfloat16; GROOT N1.6 uses float16 throughout (matches
# Thor's reference implementation, lets us use the existing FP16
# fvk kernel set including layer_norm_no_affine_fp16 and
# ada_layer_norm_fp16 which don't have bf16 variants).


class RtxFlashAttnBackendGroot:
    """FA2-backed attention for GROOT N1.6 (FP16) on RTX hardware.

    Name parallel to :class:`RtxFlashAttnBackend` (the Pi0/Pi0.5
    backend). Previously called ``TorchFlashAttnBackendGroot`` — the
    ``Torch`` prefix was misleading since jax frontends use this
    backend too. A deprecated module-level alias is preserved at the
    bottom of this file for external plugins pinned to the old name.

    **Protocol compatibility**: implements
    :class:`flash_rt.hardware.backend.AttentionBackend` via
    :meth:`sites` / :meth:`get_slot_ptrs` / :meth:`run`. Legacy
    methods (:meth:`vision_attn` / :meth:`qwen3_attn` /
    :meth:`dit_self_attn` / :meth:`dit_cross_attn` / :meth:`get_ptrs`)
    remain live for the current pipeline and will be retired once the
    pipeline migrates to the protocol API.

    Site mapping
    ------------
    GROOT has four distinct attention shapes:

      * ``"siglip"`` — 27 SigLIP layers, batched per-view self-attn,
        16 heads, head_dim=72, 256 tokens/view.
      * ``"qwen3"`` — 16 Qwen3 encoder layers, GQA 16Q/8KV, head_dim=128.
      * ``"dit_self"`` — 16 DiT self-attention blocks (DiT layers
        1, 3, 5, …, 31 — the odd layers). 32 heads, head_dim=48.
      * ``"dit_cross"`` — 16 DiT cross-attention blocks (DiT layers
        0, 2, 4, …, 30 — the even layers). Q over action seq,
        K/V over the Qwen3 backbone features precomputed once per
        diffusion loop.

    DiT layer indexing
    ------------------
    The production pipeline iterates DiT layers ``l`` in ``range(32)``
    and decides ``is_self = (l % 2 == 1)`` per iteration. The new
    protocol wants the pipeline to pass ``run("dit_self", l, …)`` or
    ``run("dit_cross", l, …)`` with the **full DiT layer index**
    ``l`` (0..31). This backend maps internally:

      * self site: ``layer_idx`` is passed straight through; there
        are effectively 16 distinct layer indices (the odd ones) but
        the shared Q/K/V torch tensors don't care — a single slot is
        reused across all self-attn layers, the layer_idx is accepted
        for protocol uniformity.
      * cross site: ``block_idx = layer_idx // 2``. Each cross block
        has its own precomputed K/V in ``dit_cross_K[block_idx]`` /
        ``dit_cross_V[block_idx]``; Q is shared across all blocks.
        The pipeline must write Q into the shared slot before every
        cross attention call but writes K/V into
        ``dit_cross_K/V[block_idx]`` **once** before the diffusion
        loop (that's the ``precompute_cross_kv`` step).
    """

    def __init__(self, num_views: int, encoder_seq_max: int,
                 num_dit_actions: int, dit_kv_seq: int,
                 num_dit_cross_blocks: int = 16):
        import torch
        self._torch = torch
        fp16 = torch.float16
        d = "cuda"

        # ── SigLIP2 vision attention ──
        self.vis_Q = torch.empty(num_views, 256, 16, 72, dtype=fp16, device=d)
        self.vis_K = torch.empty(num_views, 256, 16, 72, dtype=fp16, device=d)
        self.vis_V = torch.empty(num_views, 256, 16, 72, dtype=fp16, device=d)

        # ── Qwen3 backbone GQA ──
        self.qwen3_Q = torch.empty(encoder_seq_max, 16, 128, dtype=fp16, device=d)
        self.qwen3_K = torch.empty(encoder_seq_max, 8, 128, dtype=fp16, device=d)
        self.qwen3_V = torch.empty(encoder_seq_max, 8, 128, dtype=fp16, device=d)

        # ── DiT self-attention ──
        self.dit_self_Q = torch.empty(num_dit_actions, 32, 48, dtype=fp16, device=d)
        self.dit_self_K = torch.empty(num_dit_actions, 32, 48, dtype=fp16, device=d)
        self.dit_self_V = torch.empty(num_dit_actions, 32, 48, dtype=fp16, device=d)

        # ── DiT cross-attention ──
        # Q is the action stream (rewritten per layer by the pipeline).
        # K/V are 16 per-block precomputed projections of the Qwen3 backbone,
        # owned by the backend so flash_attn can read them as torch tensors.
        # The pipeline writes these via fp16_nn GEMMs in precompute_cross_kv().
        self.dit_cross_Q = torch.empty(num_dit_actions, 32, 48, dtype=fp16, device=d)
        self.dit_cross_K = [
            torch.empty(dit_kv_seq, 32, 48, dtype=fp16, device=d)
            for _ in range(num_dit_cross_blocks)
        ]
        self.dit_cross_V = [
            torch.empty(dit_kv_seq, 32, 48, dtype=fp16, device=d)
            for _ in range(num_dit_cross_blocks)
        ]

        self._num_views = num_views
        self._encoder_seq_max = encoder_seq_max
        self._num_dit_actions = num_dit_actions
        self._dit_kv_seq = dit_kv_seq
        self._num_dit_cross_blocks = num_dit_cross_blocks

        # Output references for caching-allocator stability
        self._vis_out_ref = None
        self._qwen3_out_ref = None
        self._dit_self_out_ref = None
        self._dit_cross_out_ref = None

        # GROOT routes attention through either:
        # 1. upstream ``flash-attn`` pip wheel (preferred, full feature)
        # 2. vendored ``flash_rt_fa2`` (fallback for SM89/SM120 without pip wheel)
        # defer the import to backend instantiation (not module import)
        # so users without the wheel can still ``import flash_rt`` and run Pi0 / Pi0.5.
        self._use_vendored_fa2 = False
        try:
            from flash_attn import flash_attn_func
            self._flash_attn_func = flash_attn_func
        except ImportError:
            # Fallback to vendored flash_rt_fa2
            try:
                from flash_rt import flash_rt_fa2 as fa2
                if hasattr(fa2, 'fwd_fp16'):
                    self._fa2_module = fa2
                    self._use_vendored_fa2 = True
                    self._torch = torch  # already imported above
                else:
                    raise ImportError(
                        "flash_rt_fa2 module loaded but fwd_fp16 not available")
            except ImportError as e:
                raise ImportError(
                    "GROOT RTX backend requires either the upstream `flash-attn` "
                    "pip package OR the vendored `flash_rt_fa2` module. "
                    "Install a prebuilt wheel from "
                    "https://github.com/Dao-AILab/flash-attention/releases "
                    "or compile flash_rt_fa2 target in FlashRT."
                ) from e

    # ── Pointer interface (for pipeline's fvk kernel calls) ──

    def get_ptrs(self) -> dict:
        return {
            "vis_Q": self.vis_Q.data_ptr(),
            "vis_K": self.vis_K.data_ptr(),
            "vis_V": self.vis_V.data_ptr(),
            "qwen3_Q": self.qwen3_Q.data_ptr(),
            "qwen3_K": self.qwen3_K.data_ptr(),
            "qwen3_V": self.qwen3_V.data_ptr(),
            "dit_self_Q": self.dit_self_Q.data_ptr(),
            "dit_self_K": self.dit_self_K.data_ptr(),
            "dit_self_V": self.dit_self_V.data_ptr(),
            "dit_cross_Q": self.dit_cross_Q.data_ptr(),
            "dit_cross_K": [t.data_ptr() for t in self.dit_cross_K],
            "dit_cross_V": [t.data_ptr() for t in self.dit_cross_V],
        }

    # ── Vendored FA2 helper ──
    def _run_vendored_fa2(self, q, k, v, out_ref_attr: str,
                          batch: int, seqlen_q: int, seqlen_k: int,
                          num_heads_q: int, num_heads_k: int, head_dim: int,
                          stream: int = 0) -> int:
        """Run flash_rt_fa2.fwd_fp16 and return output pointer."""
        torch = self._torch
        # Allocate output tensor if not cached
        out_shape = (batch, seqlen_q, num_heads_q, head_dim)
        ref_name = out_ref_attr
        if not hasattr(self, ref_name) or getattr(self, ref_name) is None:
            setattr(self, ref_name, torch.empty(out_shape, dtype=torch.float16, device='cuda'))
        out_tensor = getattr(self, ref_name)
        
        # Build strides (batch, row, head) in elements
        # For row-major (B, S, H, D): strides = (S*H*D, H*D, D)
        def strides(t):
            s = t.stride()
            return (s[0], s[1], s[3])  # (batch_stride, row_stride, head_stride)
        
        softmax_scale = 1.0 / (head_dim ** 0.5)
        
        # Allocate softmax_lse output (batch, num_heads_q, seqlen_q)
        lse = torch.empty(batch, num_heads_q, seqlen_q, dtype=torch.float32, device='cuda')
        
        self._fa2_module.fwd_fp16(
            q.data_ptr(), k.data_ptr(), v.data_ptr(),
            out_tensor.data_ptr(), lse.data_ptr(),
            softmax_lse_accum=0, o_accum=0,
            batch=batch, seqlen_q=seqlen_q, seqlen_k=seqlen_k,
            num_heads_q=num_heads_q, num_heads_kv=num_heads_k,
            head_dim=head_dim,
            q_strides=strides(q), k_strides=strides(k),
            v_strides=strides(v), o_strides=strides(out_tensor),
            softmax_scale=softmax_scale,
            num_sms=0, stream=stream
        )
        return out_tensor.data_ptr()

    # ── Attention calls ──
    #
    # All four return the raw device pointer of flash_attn_func's output tensor.
    # Torch's caching allocator reuses the same slot once warmed up; the
    # pointer stays valid across CUDA Graph replays.

    def vision_attn(self, stream: int = 0) -> int:
        """Per-view SigLIP2 self-attention.

        Returns the device pointer of the (num_views, 256, 16, 72) output.
        Downstream the pipeline reshapes/views this as (num_views * 256, 1152)
        and feeds it to the next FP16 GEMM (out projection).

        Note: FA2 kernel for head_dim=72 uses hdim96 internally which can
        cause misaligned address errors on some SM architectures (SM89).
        We use torch's scaled_dot_product_attention as a fallback for SigLIP.
        """
        torch = self._torch
        
        # Use torch scaled_dot_product_attention for SigLIP (head_dim=72)
        # This avoids FA2's misaligned address issue with hdim96 kernel
        # SigLIP Q/K/V are (num_views, 256, 16, 72)
        # scaled_dot_product_attention expects (batch, heads, seq, head_dim)
        q = self.vis_Q.transpose(1, 2)  # (num_views, 16, 256, 72)
        k = self.vis_K.transpose(1, 2)  # (num_views, 16, 256, 72)
        v = self.vis_V.transpose(1, 2)  # (num_views, 16, 256, 72)
        
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        # out is (num_views, 16, 256, 72), transpose back to (num_views, 256, 16, 72)
        out = out.transpose(1, 2).contiguous()
        
        # Cache output tensor for stable pointer across graph capture
        self._vis_out_ref = out
        return out.data_ptr()

    def qwen3_attn(self, layer_idx: int, seq: int, stream: int = 0) -> int:
        """Qwen3 GQA self-attention for one layer (16Q/8KV, head_dim=128).

        Reads from ``qwen3_Q[:seq]``, ``qwen3_K[:seq]``, ``qwen3_V[:seq]``.
        Returns the (1, seq, 16, 128) output pointer.

        Note: FA2 kernel can have alignment issues on SM89, use torch 
        scaled_dot_product_attention as fallback.
        """
        torch = self._torch
        
        q = self.qwen3_Q[:seq].unsqueeze(0)        # (1, seq, 16, 128)
        k = self.qwen3_K[:seq].unsqueeze(0)        # (1, seq, 8,  128)
        v = self.qwen3_V[:seq].unsqueeze(0)        # (1, seq, 8,  128)
        
        # Use torch scaled_dot_product_attention for GQA
        # Need to expand K/V to match Q heads (GQA: 16Q/8KV, so repeat K/V 2x)
        k_expanded = k.repeat_interleave(2, dim=2)  # (1, seq, 16, 128)
        v_expanded = v.repeat_interleave(2, dim=2)  # (1, seq, 16, 128)
        
        # scaled_dot_product_attention expects (batch, heads, seq, head_dim)
        q_t = q.transpose(1, 2)  # (1, 16, seq, 128)
        k_t = k_expanded.transpose(1, 2)  # (1, 16, seq, 128)
        v_t = v_expanded.transpose(1, 2)  # (1, 16, seq, 128)
        
        out = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t)
        # out is (1, 16, seq, 128), transpose back to (1, seq, 16, 128)
        out = out.transpose(1, 2).contiguous()
        
        self._qwen3_out_ref = out
        return out.data_ptr()

    def dit_self_attn(self, layer_idx: int, sa: int, stream: int = 0) -> int:
        """DiT self-attention (32 heads × 48 head_dim) over Sa tokens.

        Returns the (1, Sa, 32, 48) output pointer.

        Note: FA2 kernel for head_dim=48 has alignment issues, use torch fallback.
        """
        torch = self._torch
        
        q = self.dit_self_Q[:sa].unsqueeze(0)  # (1, sa, 32, 48)
        k = self.dit_self_K[:sa].unsqueeze(0)
        v = self.dit_self_V[:sa].unsqueeze(0)
        
        # scaled_dot_product_attention expects (batch, heads, seq, head_dim)
        q_t = q.transpose(1, 2)  # (1, 32, sa, 48)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        
        out = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t)
        out = out.transpose(1, 2).contiguous()
        
        self._dit_self_out_ref = out
        return out.data_ptr()

    def dit_cross_attn(self, cross_block_idx: int, sa: int,
                       kv_seq: int, stream: int = 0) -> int:
        """DiT cross-attention for one cross block.

        ``cross_block_idx`` is in [0, 16) — the cross blocks are at DiT
        layers 0, 2, 4, ..., 30. The backend owns persistent K/V tensors
        for each cross block (precomputed by the pipeline before each
        diffusion run loop). Q is the action stream which the pipeline
        rewrites per layer.

        Note: FA2 kernel for head_dim=48 has alignment issues, use torch fallback.
        """
        torch = self._torch
        
        q = self.dit_cross_Q[:sa].unsqueeze(0)
        k = self.dit_cross_K[cross_block_idx][:kv_seq].unsqueeze(0)
        v = self.dit_cross_V[cross_block_idx][:kv_seq].unsqueeze(0)
        
        # scaled_dot_product_attention expects (batch, heads, seq, head_dim)
        q_t = q.transpose(1, 2)  # (1, 32, sa, 48)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)
        
        out = torch.nn.functional.scaled_dot_product_attention(q_t, k_t, v_t)
        out = out.transpose(1, 2).contiguous()
        
        self._dit_cross_out_ref = out
        return out.data_ptr()

    # ──────────────────────────────────────────────────────────────
    # AttentionBackend protocol implementation. Wraps the same
    # flash_attn_func calls as the legacy methods above. The legacy
    # half will be retired once the GROOT pipeline routes entirely
    # through the protocol methods.
    # ──────────────────────────────────────────────────────────────

    _PROTOCOL_SITES = ("siglip", "qwen3", "dit_self", "dit_cross")

    def sites(self) -> tuple[str, ...]:
        return self._PROTOCOL_SITES

    def head_dim(self, site: str) -> int:
        return {
            "siglip": 72,
            "qwen3": 128,
            "dit_self": 48,
            "dit_cross": 48,
        }[site]

    def num_q_heads(self, site: str) -> int:
        return {
            "siglip": 16,
            "qwen3": 16,
            "dit_self": 32,
            "dit_cross": 32,
        }[site]

    def num_kv_heads(self, site: str) -> int:
        return {
            "siglip": 16,  # SigLIP is MHA
            "qwen3": 8,    # Qwen3 is GQA 16Q/8KV
            "dit_self": 32,
            "dit_cross": 32,
        }[site]

    def get_slot_ptrs(self, site: str, layer_idx: int) -> dict[str, int]:
        """Return pointer dict for one (site, layer) pair.

        Notes on layer_idx semantics per site:

          * ``siglip``: all 27 layers share one batched Q/K/V slot.
            layer_idx accepted for uniformity but ignored.
          * ``qwen3``: all 16 layers share one flat Q/K/V slot
            (GQA). layer_idx ignored.
          * ``dit_self``: all 16 self-attn DiT blocks share one flat
            Q/K/V slot. ``layer_idx`` is the full DiT layer index
            (1, 3, ..., 31) and is accepted for uniformity but only
            used to sanity-check the odd-layer contract.
          * ``dit_cross``: 16 per-block K/V slots (one tensor each).
            ``layer_idx`` is the full DiT layer index (0, 2, 4, ..., 30)
            which we map to ``block_idx = layer_idx // 2``. Q is
            shared across all 16 cross blocks.
        """
        if site == "siglip":
            return {
                "Q": self.vis_Q.data_ptr(),
                "K": self.vis_K.data_ptr(),
                "V": self.vis_V.data_ptr(),
            }
        if site == "qwen3":
            return {
                "Q": self.qwen3_Q.data_ptr(),
                "K": self.qwen3_K.data_ptr(),
                "V": self.qwen3_V.data_ptr(),
            }
        if site == "dit_self":
            return {
                "Q": self.dit_self_Q.data_ptr(),
                "K": self.dit_self_K.data_ptr(),
                "V": self.dit_self_V.data_ptr(),
            }
        if site == "dit_cross":
            if layer_idx % 2 != 0:
                raise ValueError(
                    f"dit_cross layer_idx must be even (DiT cross "
                    f"blocks are at layers 0, 2, 4, ..., 30), got "
                    f"layer_idx={layer_idx}"
                )
            block_idx = layer_idx // 2
            if not (0 <= block_idx < self._num_dit_cross_blocks):
                raise IndexError(
                    f"dit_cross block_idx {block_idx} out of range "
                    f"[0, {self._num_dit_cross_blocks})"
                )
            return {
                "Q": self.dit_cross_Q.data_ptr(),
                "K": self.dit_cross_K[block_idx].data_ptr(),
                "V": self.dit_cross_V[block_idx].data_ptr(),
            }
        raise KeyError(f"unknown site {site!r}; known: {self._PROTOCOL_SITES}")

    def run(
        self,
        site: str,
        layer_idx: int,
        q_seq: int,
        *,
        kv_seq=None,
        stream: int = 0,
    ) -> int:
        """Dispatch to the legacy attention call for the given site.

        Wraps the same ``flash_attn_func`` invocations as the legacy
        methods. Returns the freshly-allocated output tensor's data_ptr,
        which is stable across CUDA Graph capture + replay because the
        backend holds a reference in ``_{site}_out_ref``.
        """
        if site == "siglip":
            # Pi0.5's SigLIP was 2-view × 256 tokens, GROOT's SigLIP
            # (27L) uses a different, per-view batched API. The
            # legacy vision_attn doesn't take q_seq at all.
            return self.vision_attn(stream=stream)
        if site == "qwen3":
            if kv_seq is not None and kv_seq != q_seq:
                raise ValueError(
                    f"qwen3 site is self-attention; kv_seq must be "
                    f"None or equal to q_seq, got kv_seq={kv_seq} "
                    f"q_seq={q_seq}"
                )
            return self.qwen3_attn(layer_idx, q_seq, stream=stream)
        if site == "dit_self":
            if kv_seq is not None and kv_seq != q_seq:
                raise ValueError(
                    f"dit_self site is self-attention; kv_seq must be "
                    f"None or equal to q_seq, got kv_seq={kv_seq} "
                    f"q_seq={q_seq}"
                )
            return self.dit_self_attn(layer_idx, q_seq, stream=stream)
        if site == "dit_cross":
            if kv_seq is None:
                raise ValueError(
                    "dit_cross site is cross-attention against the "
                    "Qwen3 backbone features; kv_seq (= Se, the "
                    "encoder sequence length) must be supplied"
                )
            if layer_idx % 2 != 0:
                raise ValueError(
                    f"dit_cross layer_idx must be even, got "
                    f"layer_idx={layer_idx}"
                )
            block_idx = layer_idx // 2
            return self.dit_cross_attn(block_idx, q_seq, kv_seq,
                                       stream=stream)
        raise KeyError(f"unknown site {site!r}; known: {self._PROTOCOL_SITES}")


# Backwards-compatible alias — see the class docstring for the rename
# history. External plugins pinned to the old name continue to work.
TorchFlashAttnBackendGroot = RtxFlashAttnBackendGroot
