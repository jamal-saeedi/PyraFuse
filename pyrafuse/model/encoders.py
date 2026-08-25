from __future__ import annotations
from transformers import SamModel
import timm
from transformers import CLIPVisionConfig, CLIPVisionModel
from transformers import AutoModel
import time
import math
from abc import ABC, abstractmethod
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.decomposition import PCA


# Layer schedules: which transformer block indices to tap for 4-scale features.
# Index i means the output *after* block i (0-based).
LAYER_SCHEDULE = {
    12: [2, 5, 8, 11],   # ViT-B / ViT-S  (12 blocks)
    24: [5, 11, 17, 23],  # ViT-L          (24 blocks)
    32: [7, 15, 23, 31],  # ViT-H / SAM-H  (32 blocks)
}
DEVICE = "cuda:3"


class BackboneAdapter(ABC, nn.Module):
    """Common interface: 4 spatial feature maps at patch stride.

    Feature-map normalisation
    -------------------------
    Different encoders (DINOv3, CLIP, SigLIP, SAM, RADIO, …) emit feature maps
    with very different per-channel scale/variance.  To give the decoder a
    consistent input distribution regardless of which backbone is plugged in,
    each output map is passed through a channel-wise ``LayerNorm`` (per-token
    norm, matching how ViTs normalise internally and batch-size independent —
    important since the encoder is usually frozen and run at small batch).

    The norm lives at the *adapter boundary* (best practice for a pluggable /
    hybrid design): subclasses only implement ``extract_features`` to produce
    raw maps; the base ``forward`` applies the norm.  The norm is the *only*
    trainable part of an otherwise-frozen encoder, so its gradients are kept
    alive even though feature extraction itself runs under ``@torch.no_grad``.

    Set ``normalize_features=False`` to recover the original raw behaviour.
    """

    name: str = ''
    patch: int = 16

    def __init__(self, normalize_features: bool = True):
        super().__init__()
        self._normalize_features = normalize_features
        self._backbone_frozen = False
        # One LayerNorm per scale, built lazily on first forward (we need a
        # concrete channel count, and `embed_dim` is only valid after the
        # subclass __init__ has loaded its backbone).
        self.feature_norms: nn.ModuleList | None = None

    def set_backbone_frozen(self, frozen: bool) -> None:
        """Record whether the backbone is frozen so the (lazily-built) feature
        norms can stay trainable on top of a frozen backbone."""
        self._backbone_frozen = frozen
        if self.feature_norms is not None:
            for p in self.feature_norms.parameters():
                p.requires_grad_(True)

    def __deepcopy__(self, memo):
        import copy
        cls = self.__class__
        new = cls.__new__(cls)
        memo[id(self)] = new
        for k, v in self.__dict__.items():
            object.__setattr__(new, k, copy.deepcopy(v, memo))
        return new

    @property
    @abstractmethod
    def embed_dim(self) -> int: ...

    @abstractmethod
    def extract_features(self, pixel_values: torch.Tensor) -> List[torch.Tensor]:
        """Return 4 x [B, C, H/patch, W/patch] coarse-to-fine (raw, un-normalised)."""
        ...

    # ------------------------------------------------------------- normalize
    def init_feature_norms(self, num_scales: int = 4) -> None:
        """Create one channel-wise LayerNorm per output scale.

        Built eagerly (right after the backbone is loaded) rather than on the
        first forward, so the norm parameters already exist when the optimizer
        is constructed — otherwise they would never be registered for training.
        All current adapters emit ``num_scales`` maps with ``embed_dim``
        channels each.
        """
        if self.feature_norms is not None:
            return
        norms = [nn.LayerNorm(self.embed_dim) for _ in range(num_scales)]
        self.feature_norms = nn.ModuleList(norms)
        # The norm is the trainable calibration layer; keep it learnable even
        # when the backbone has been frozen via lock_encoder().
        for p in self.feature_norms.parameters():
            p.requires_grad_(True)

    def _apply_feature_norm(self, maps: List[torch.Tensor]) -> List[torch.Tensor]:
        """Channel-wise LayerNorm on each [B, C, H, W] map (normalises C)."""
        if self.feature_norms is None:
            self.init_feature_norms(len(maps))
            # Match the freshly-built norms to the feature device/dtype.
            ref = maps[0]
            self.feature_norms = self.feature_norms.to(
                device=ref.device, dtype=ref.dtype)
        out: List[torch.Tensor] = []
        for m, ln in zip(maps, self.feature_norms):
            # LayerNorm normalises the last dim → move C last, then back.
            m = ln(m.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
            out.append(m)
        return out

    def forward(self, pixel_values: torch.Tensor) -> List[torch.Tensor]:
        """Run the backbone (no-grad in subclasses) then the trainable norm.

        VLM adapters that mix in :class:`TextGatedMixin` and have had prompts
        set via ``set_text_prompts`` apply a text-conditioned residual gate
        *after* normalisation (so the gate sees calibrated features and the
        decoder still receives ``embed_dim``-channel maps unchanged).
        """
        maps = self.extract_features(pixel_values)
        if self._normalize_features:
            maps = self._apply_feature_norm(maps)
        gate = getattr(self, '_maybe_text_gate', None)
        if gate is not None:
            maps = gate(pixel_values, maps)
        return maps

    # ------------------------------------------------------------------ util
    @staticmethod
    def _tokens_to_map(tokens: torch.Tensor, ph: int, pw: int) -> torch.Tensor:
        """[B, T, C] → [B, C, ph, pw]"""
        return tokens.transpose(1, 2).reshape(tokens.shape[0], -1, ph, pw)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def throughput(self, x: torch.Tensor, n: int = 10) -> float:
        """Images/s averaged over n runs."""
        # warm-up
        for _ in range(3):
            self(x)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            self(x)
        if DEVICE == 'cuda':
            torch.cuda.synchronize()
        return x.shape[0] * n / (time.perf_counter() - t0)


# ── Text-conditioned gating (MaskCLIP / CLIPSeg) ────────────────────────────
# Only VLM backbones with a joint vision-language space (CLIP, SigLIP,
# EVA-CLIP) can do this: encode class prompts ("skin", "fabric", "background")
# through the text tower, cosine-match every patch token against them to get a
# per-class soft spatial mask, then residually gate the patch features.
#
# Contract for subclasses:
#   _text_encode(prompts) -> [K, D]   text embeddings (NOT yet L2-normed)
#   _project_patches(map) -> [B, T, D]  patch grid mapped into the *joint*
#                            space (so cosine sim with text is meaningful),
#                            NOT yet L2-normed.  T = ph*pw, D = joint dim.
# The gate keeps output shape identical to the input map: [B, C, ph, pw].


class TextGatedMixin:
    """Adds a fixed-prompt, cached, residual text gate to a VLM adapter."""

    # Sigmoid temperature for the saliency squash; lower = sharper contrast.
    _gate_temp: float = 2.
    # How per-patch saliency is reduced over the K classes before the sigmoid:
    #   'max'    — max cosine over classes. Saturates when the prompt set tiles
    #              the whole image (e.g. a 'background' class), so the gate goes
    #              spatially flat: every patch wins *some* class strongly.
    #   'margin' — (top-1 − top-2) cosine, z-scored per image. Measures how
    #              *confidently* one class wins, so ambiguous/uniform regions get
    #              a low gate and confidently-classified regions get a high one.
    _gate_mode: str = 'margin'
    # How many of the LAST blocks' value-only cosine maps to average before the
    # saliency squash. 1 = penultimate block only (original behaviour); >1 runs
    # the MaskCLIP value-only projection at several late blocks and averages the
    # per-class *cosine maps* (not the tokens — mid-depth tokens aren't in the
    # joint space). Averaging cancels per-layer speckle while keeping the shared
    # semantics. Costs ~N× the gate forward (the backbone is run once per depth).
    _gate_levels: int = 1
    # Gaussian blur applied to the (post-sigmoid) gate on the patch grid before
    # it is broadcast to each scale, in *patch* units. Per-patch saliency is
    # computed independently, so neighbours can disagree → salt-and-pepper; a
    # small blur makes them agree. 0.0 disables; ~0.8 removes single-patch noise
    # without melting the skin/background boundary.
    _gate_smooth_sigma: float = 0.8

    def set_text_prompts(self, prompts: List[str]) -> None:
        """Set (and cache-invalidate) the class prompts. Encoded lazily on the
        first forward so we land on the correct device/dtype."""
        self._prompts = list(prompts)
        self._class_emb = None  # [K, D] L2-normed, built lazily

    def _num_blocks(self) -> int:
        """Number of transformer blocks — used to pick the late blocks for
        multi-level gating. Adapters with a non-standard trunk override this."""
        return len(self.m.encoder.layers)

    def _ensure_class_emb(self, device, dtype) -> torch.Tensor:
        if getattr(self, '_class_emb', None) is None:
            with torch.no_grad():
                emb = self._text_encode(self._prompts)        # [K, D]
            self._class_emb = F.normalize(
                emb.float(), dim=-1).to(device, dtype)
        return self._class_emb

    def _gate_tokens(self, pixel_values, ref_map):
        """Patch tokens in the joint VLM space for the gate, [B, T, D].

        Default: project the (attention-mixed) last-block feature map. VLM
        adapters override this with the **MaskCLIP value-only** path —
        ``out_proj(v_proj(x))`` on the penultimate tokens — so each patch keeps
        its own identity instead of being globally attention-pooled. That is
        what makes the per-class similarity localise instead of looking like
        noise. Falls back to ``_project_patches`` if no value-only path exists.
        """
        fn = getattr(self, '_maskclip_tokens', None)
        if fn is not None:
            return fn(pixel_values)
        return self._project_patches(ref_map)

    def _gate_sim(self, pixel_values, ref_map, emb):
        """[B, T, K] cosine of patch tokens vs class embeddings.

        With ``_gate_levels > 1`` and a MaskCLIP value-only path available,
        average the cosine maps over the last ``_gate_levels`` blocks. The
        backbone is run **once** (``_gate_hidden_states``) and the value-only
        projection (``_maskclip_project``) is applied per depth on the shared
        hidden states — so multi-level costs one forward, not N. Falls back to
        the single default ``_gate_tokens`` path when no value-only path exists.
        """
        n_lvl = int(getattr(self, '_gate_levels', 1))
        proj = getattr(self, '_maskclip_project', None)
        if proj is None:                                       # no value-only path
            tok = F.normalize(self._gate_tokens(pixel_values, ref_map), dim=-1)
            return torch.einsum('btd,kd->btk', tok, emb)
        hs = self._gate_hidden_states(pixel_values)            # ONE forward
        nb = self._num_blocks()
        depths = range(nb - n_lvl, nb) if n_lvl > 1 else (nb - 1,)
        sims = []
        for d in depths:
            tok = F.normalize(proj(hs, d), dim=-1)
            sims.append(torch.einsum('btd,kd->btk', tok, emb))
        if len(sims) == 1:
            return sims[0]
        # fuse late blocks
        return torch.stack(sims, 0).mean(0)

    @staticmethod
    def _gate_smooth(gate_2d, sigma):
        """[B, 1, h, w] gate in (0,1) → separable Gaussian blur (replicate pad).
        sigma is in patch (grid) units; <= 0 is a no-op."""
        if sigma <= 0:
            return gate_2d
        r = max(1, int(round(3 * sigma)))
        xs = torch.arange(-r, r + 1, device=gate_2d.device,
                          dtype=gate_2d.dtype)
        k = torch.exp(-(xs ** 2) / (2 * sigma ** 2))
        k = k / k.sum()
        x = F.pad(gate_2d, (r, r, 0, 0), mode='replicate')
        x = F.conv2d(x, k.view(1, 1, 1, -1))                   # blur W
        x = F.pad(x, (0, 0, r, r), mode='replicate')
        x = F.conv2d(x, k.view(1, 1, -1, 1))                   # blur H
        return x

    def _gate_saliency(self, sim):
        """[B, T, K] cosine → [B, T] saliency in (0,1) via the configured mode.

        'max' squashes the top class cosine directly; 'margin' z-scores the
        top1−top2 gap per image so the sigmoid sees *relative* confidence rather
        than absolute cosine (which a 'background' class keeps uniformly high).
        """
        mode = getattr(self, '_gate_mode', 'margin')
        if mode == 'max':
            score = sim.max(dim=-1).values                     # [B, T]
        elif mode == 'margin':
            if sim.shape[-1] < 2:                              # single prompt
                score = sim.squeeze(-1)
            else:
                top2 = sim.topk(2, dim=-1).values              # [B, T, 2]
                margin = top2[..., 0] - top2[..., 1]           # [B, T] >= 0
                # per-image stats
                mu = margin.mean(dim=1, keepdim=True)
                sd = margin.std(dim=1, keepdim=True).clamp_min(1e-6)
                score = (margin - mu) / sd                     # z-scored
        else:
            raise ValueError(f'unknown _gate_mode: {mode!r}')
        # [B, T] in (0,1)
        return torch.sigmoid(score / self._gate_temp)

    def _maybe_text_gate(self, pixel_values, maps):
        if not getattr(self, '_prompts', None):
            return maps                                       # no prompts set
        emb = self._ensure_class_emb(maps[0].device, maps[0].dtype)  # [K, D]
        # One mask, computed from the finest scale where the joint-space
        # alignment is valid, then broadcast to every scale. Uses the MaskCLIP
        # value-only tokens (via _gate_tokens) for spatial coherence, and a
        # sigmoid-style saliency (see _gate_saliency / _gate_mode) rather than
        # softmax-over-classes — softmax across only K generic prompts flattens
        # contrast, washing the mask out.
        ref = maps[-1]
        b, _, rh, rw = ref.shape
        # [B, T, K] cosine — single layer, or averaged over the last
        # _gate_levels blocks (see _gate_sim).
        sim = self._gate_sim(pixel_values, ref, emb)
        # [B, T] in (0,1)
        gate = self._gate_saliency(sim)
        gate = gate.reshape(b, 1, rh, rw)                      # [B, 1, rh, rw]
        # smooth on the patch grid so neighbouring patches agree (anti-speckle)
        gate = self._gate_smooth(gate, getattr(
            self, '_gate_smooth_sigma', 0.0))
        out: List[torch.Tensor] = []
        for m in maps:
            g = gate
            if m.shape[-2:] != gate.shape[-2:]:                # match each scale
                g = F.interpolate(gate, size=m.shape[-2:],
                                  mode='bilinear', align_corners=False)
            out.append(m * (1.0 + g))                          # residual gate
        return out


# ── 2a  DINOv3 ViT-B/16 ─────────────────────────────────────────────────────
# This is the backbone used in model.py.  DINOv3 adds Gram Anchoring on top of
# DINOv2's SSL objective for better patch-level dense features.
# Token layout: 1 CLS + num_register_tokens (default 4) + patch grid.


class DINOv3Adapter(BackboneAdapter):
    """DINOv3 ViT-B/16  —  facebook/dinov3-vitb16-pretrain-lvd1689m.

    Mirrors DINOv3MultiScaleEmbed in model.py exactly:
      hidden_states[0]   = patch embeddings (before block 0)
      hidden_states[i+1] = output of block i
    Strips num_prefix = 1 CLS + num_register_tokens before reshaping to grid.
    """
    name = 'DINOv3 ViT-B/16'
    patch = 16

    def __init__(self, model_id: str = 'facebook/dinov3-vitb16-pretrain-lvd1689m',
                 normalize_features: bool = True):
        super().__init__(normalize_features=normalize_features)
        self.m = AutoModel.from_pretrained(model_id).eval()
        n = self.m.config.num_hidden_layers
        self._layers = LAYER_SCHEDULE[n]
        self._num_prefix = 1 + getattr(self.m.config, 'num_register_tokens', 0)

    @property
    def embed_dim(self) -> int:
        return self.m.config.hidden_size

    @torch.no_grad()
    def extract_features(self, pixel_values: torch.Tensor) -> List[torch.Tensor]:
        b, _, h, w = pixel_values.shape
        ph, pw = h // self.patch, w // self.patch
        hs = self.m(pixel_values=pixel_values,
                    output_hidden_states=True).hidden_states
        return [
            self._tokens_to_map(hs[i + 1][:, self._num_prefix:], ph, pw)
            for i in self._layers
        ]


# ── 2a  DINOv2 ViT-S/14 ─────────────────────────────────────────────────────


class DINOv2Adapter(BackboneAdapter):
    """DINOv2 ViT-S/14  —  facebook/dinov2-small  (22M, embed=384)."""
    name = 'DINOv2 ViT-S/14'
    patch = 14

    def __init__(self, model_id: str = 'facebook/dinov2-small',
                 normalize_features: bool = True):
        super().__init__(normalize_features=normalize_features)
        self.m = AutoModel.from_pretrained(model_id).eval()
        n = self.m.config.num_hidden_layers
        self._layers = LAYER_SCHEDULE[n]
        self._num_prefix = 1 + getattr(self.m.config, 'num_register_tokens', 0)

    @property
    def embed_dim(self) -> int:
        return self.m.config.hidden_size

    @torch.no_grad()
    def extract_features(self, pixel_values: torch.Tensor) -> List[torch.Tensor]:
        b, _, h, w = pixel_values.shape
        ph, pw = h // self.patch, w // self.patch
        hs = self.m(pixel_values=pixel_values,
                    output_hidden_states=True).hidden_states
        return [
            self._tokens_to_map(hs[i + 1][:, self._num_prefix:], ph, pw)
            for i in self._layers
        ]


# ── 2b  CLIP ViT-B/16 ───────────────────────────────────────────────────────


class CLIPAdapter(TextGatedMixin, BackboneAdapter):
    name = 'CLIP ViT-B/16'
    patch = 16

    def __init__(self, model_id: str = 'openai/clip-vit-base-patch16',
                 normalize_features: bool = True):
        super().__init__(normalize_features=normalize_features)
        vision_cfg = CLIPVisionConfig.from_pretrained(model_id)
        self.m = CLIPVisionModel(vision_cfg)
        from transformers import CLIPModel, AutoTokenizer
        joint = CLIPModel.from_pretrained(model_id)
        self.m.load_state_dict(joint.vision_model.state_dict())
        # Keep the text tower + projections for text-conditioned gating.  The
        # vision-side post-LN + visual_projection map patch tokens into the
        # joint space (MaskCLIP: project *every* patch, not just CLS).
        self._joint = joint.eval()
        self._tok = AutoTokenizer.from_pretrained(model_id)
        self.m.eval()
        self._layers = LAYER_SCHEDULE[vision_cfg.num_hidden_layers]

    @property
    def embed_dim(self): return self.m.config.hidden_size

    @torch.no_grad()
    def _text_encode(self, prompts):
        enc = self._tok(prompts, padding=True, return_tensors='pt').to(
            self._joint.device)
        out = self._joint.get_text_features(**enc)            # [K, 512]
        return getattr(out, 'pooler_output', out)

    def _project_patches(self, m):
        """[B, C, ph, pw] patch grid → [B, T, 512] in CLIP joint space."""
        b, c, ph, pw = m.shape
        tok = m.flatten(2).transpose(1, 2)                    # [B, T, C]
        tok = self.m.post_layernorm(tok)
        return self._joint.visual_projection(tok)             # [B, T, 512]

    def _num_blocks(self):
        return len(self.m.encoder.layers)

    @torch.no_grad()
    def _gate_hidden_states(self, pixel_values):
        """All hidden states from ONE backbone forward (shared across depths)."""
        return self.m(pixel_values=pixel_values,
                      interpolate_pos_encoding=True,
                      output_hidden_states=True).hidden_states

    def _maskclip_project(self, hs, depth):
        """Value-only projection of the pre-block-``depth`` hidden state → joint
        space [B, T, 512] (no CLS). Reuses already-computed ``hs`` (no re-run):
        block ``depth``'s value+output projections (skipping q·k attention
        pooling) so each patch keeps its own identity, then post-LN + visual
        projection."""
        x = hs[depth]                                        # [B, 1+T, C] pre block `depth`
        attn = self.m.encoder.layers[depth].self_attn
        x = self.m.encoder.layers[depth].layer_norm1(x)      # pre-attn LN
        v = attn.out_proj(attn.v_proj(x))                    # value-only path
        v = v[:, 1:]                                         # drop CLS
        v = self.m.post_layernorm(v)
        return self._joint.visual_projection(v)              # [B, T, 512]

    @torch.no_grad()
    def _maskclip_tokens(self, pixel_values, depth=None):
        """Single-depth convenience wrapper (used by the notebook viz)."""
        if depth is None:
            depth = len(self.m.encoder.layers) - 1
        return self._maskclip_project(
            self._gate_hidden_states(pixel_values), depth)

    @torch.no_grad()
    def extract_features(self, pixel_values):
        b, _, h, w = pixel_values.shape
        ph, pw = h // self.patch, w // self.patch
        hs = self.m(pixel_values=pixel_values,
                    interpolate_pos_encoding=True,
                    output_hidden_states=True).hidden_states
        return [self._tokens_to_map(hs[i+1][:, 1:], ph, pw) for i in self._layers]


# ── 2c  SigLIP ViT-B/16 ─────────────────────────────────────────────────────
class SigLIPAdapter(TextGatedMixin, BackboneAdapter):
    name = 'SigLIP ViT-B/16'
    patch = 16

    def __init__(self, model_id: str = 'google/siglip-base-patch16-224',
                 normalize_features: bool = True):
        super().__init__(normalize_features=normalize_features)
        from transformers import AutoModel as _AM, AutoTokenizer
        joint = _AM.from_pretrained(model_id).eval()
        self.m = joint.vision_model
        # Keep the text tower for gating. SigLIP's joint dim == vision hidden
        # (768); patch tokens are already in the joint space after the vision
        # encoder's final norm, so _project_patches just applies that norm.
        self._joint = joint
        self._tok = AutoTokenizer.from_pretrained(model_id)
        self._post_ln = self.m.post_layernorm
        self._layers = LAYER_SCHEDULE[self.m.config.num_hidden_layers]

    @property
    def embed_dim(self): return self.m.config.hidden_size

    @torch.no_grad()
    def _text_encode(self, prompts):
        enc = self._tok(prompts, padding='max_length', return_tensors='pt').to(
            self._joint.device)
        out = self._joint.get_text_features(**enc)            # [K, 768]
        return getattr(out, 'pooler_output', out)

    def _project_patches(self, m):
        b, c, ph, pw = m.shape
        tok = m.flatten(2).transpose(1, 2)                    # [B, T, C]
        return self._post_ln(tok)                             # [B, T, 768]

    def _num_blocks(self):
        return len(self.m.encoder.layers)

    @torch.no_grad()
    def _gate_hidden_states(self, pixel_values):
        return self.m(pixel_values=pixel_values,
                      interpolate_pos_encoding=True,
                      output_hidden_states=True).hidden_states

    def _maskclip_project(self, hs, depth):
        """Value-only projection from shared ``hs`` → [B, T, 768] (no CLS).
        SigLIP is pre-norm: layer_norm1 → block ``depth``'s value+output
        projections → post_layernorm (per-token joint space; the attention-pool
        ``head`` only collapses to one vector)."""
        x = hs[depth]                                        # [B, T, C] pre block `depth`
        blk = self.m.encoder.layers[depth]
        attn = blk.self_attn
        x = blk.layer_norm1(x)                               # pre-attn LN
        v = attn.out_proj(attn.v_proj(x))                    # value-only path
        return self._post_ln(v)                              # [B, T, 768]

    def _maskclip_tokens(self, pixel_values, depth=None):
        """Single-depth convenience wrapper (used by the notebook viz)."""
        if depth is None:
            depth = len(self.m.encoder.layers) - 1
        return self._maskclip_project(
            self._gate_hidden_states(pixel_values), depth)

    @torch.no_grad()
    def extract_features(self, pixel_values):
        b, _, h, w = pixel_values.shape
        ph, pw = h // self.patch, w // self.patch
        hs = self.m(pixel_values=pixel_values,
                    interpolate_pos_encoding=True,
                    output_hidden_states=True).hidden_states
        return [self._tokens_to_map(hs[i+1], ph, pw) for i in self._layers]


# ── 2d  EVA-CLIP ViT-B/16 ───────────────────────────────────────────────────


class EVACLIPAdapter(TextGatedMixin, BackboneAdapter):
    name = 'EVA-CLIP ViT-B/16'
    patch = 16

    def __init__(self, model_id: str = 'EVA02-B-16',
                 pretrained: str = 'merged2b_s8b_b131k',
                 img_size: int = 448, normalize_features: bool = True):
        super().__init__(normalize_features=normalize_features)
        # open_clip (not timm) so we keep the text tower for text gating. The
        # visual side is a TimmModel wrapping a `trunk` ViT (embed_dim=768) +
        # a head that projects to the joint space (512). Patch->joint is
        # trunk.head(trunk.norm(patch)).
        import open_clip
        clip, _, _ = open_clip.create_model_and_transforms(
            model_id, pretrained=pretrained)
        self._clip = clip.eval()
        self._tok = open_clip.get_tokenizer(model_id)
        self._trunk = clip.visual.trunk
        # Adapt to the larger working resolution: set_input_size re-interpolates
        # the position embeddings (224->img_size) so token count matches the
        # patch grid — otherwise pos_embed (197) clashes with the patches (785).
        self._trunk.set_input_size(img_size=(img_size, img_size))
        n = len(self._trunk.blocks)
        self._layer_idx = LAYER_SCHEDULE.get(
            n) or [n//4-1, n//2-1, 3*n//4-1, n-1]

    @property
    def embed_dim(self): return self._trunk.embed_dim

    @torch.no_grad()
    def _text_encode(self, prompts):
        toks = self._tok(prompts).to(next(self._clip.parameters()).device)
        return self._clip.encode_text(toks)                   # [K, 512]

    def _project_patches(self, m):
        b, c, ph, pw = m.shape
        tok = m.flatten(2).transpose(1, 2)                    # [B, T, 768]
        return self._trunk.head(self._trunk.norm(tok))        # [B, T, 512]

    def _num_blocks(self):
        return len(self._trunk.blocks)

    @torch.no_grad()
    def _gate_hidden_states(self, pixel_values):
        """EVA has no HF hidden_states; pull every block-input we'll need in ONE
        forward_intermediates call. Returns {block_input_index: [B, T, 768]}.

        We may be asked for the input to any of the last ``_gate_levels`` blocks
        (== output of the previous block), so request those indices up front."""
        n = len(self._trunk.blocks)
        n_lvl = max(1, int(getattr(self, '_gate_levels', 1)))
        depths = range(n - n_lvl, n) if n_lvl > 1 else (n - 1,)
        # block-input indices
        idx = sorted({d - 1 for d in depths})
        feats = self._trunk.forward_intermediates(
            pixel_values, indices=idx, output_fmt='NLC',
            intermediates_only=True)
        return dict(zip(idx, feats))

    def _maskclip_project(self, hs, depth):
        """Value-only projection from the shared intermediates → [B, T, 512].
        EVA is pre-norm with an inner attn norm: norm1 → v_proj → attn.norm →
        proj (skipping q·k attention) then the trunk's joint head (norm → head).
        ``hs`` is the dict from ``_gate_hidden_states``; tokens entering block
        ``depth`` == output of block ``depth-1``."""
        pen = hs[depth - 1]                                   # [B, T, 768]
        blk = self._trunk.blocks[depth]
        attn = blk.attn
        h = blk.norm1(pen)
        v = attn.proj(attn.norm(attn.v_proj(h)))              # value-only path
        return self._trunk.head(self._trunk.norm(v))          # [B, T, 512]

    @torch.no_grad()
    def _maskclip_tokens(self, pixel_values, depth=None):
        """Single-depth convenience wrapper (used by the notebook viz)."""
        n = len(self._trunk.blocks)
        if depth is None:
            depth = n - 1
        pen = self._trunk.forward_intermediates(
            pixel_values, indices=[depth - 1],
            output_fmt='NLC', intermediates_only=True)[0]
        return self._maskclip_project({depth - 1: pen}, depth)

    @torch.no_grad()
    def extract_features(self, pixel_values):
        b, _, h, w = pixel_values.shape
        ph, pw = h // self.patch, w // self.patch
        # forward_intermediates with NLC already strips prefix tokens (CLS/registers)
        feats = self._trunk.forward_intermediates(
            pixel_values, indices=self._layer_idx,
            output_fmt='NLC', intermediates_only=True)
        return [self._tokens_to_map(f, ph, pw) for f in feats]


# ── 2e  AM-RADIO-B ──────────────────────────────────────────────────────────
# Load via torch hub (NVlabs/RADIO) — avoids the transformers trust_remote_code
# path that hard-imports open_clip and breaks on transformers >= 5.x.
# Prefix token count read from patch_generator.num_skip (authoritative field).

class RADIOAdapter(BackboneAdapter):
    """AM-RADIO-B  —  radio_v2.5-b via torch.hub (NVlabs/RADIO)."""
    name = 'AM-RADIO-B'
    patch = 16

    def __init__(self, version: str = 'radio_v2.5-b',
                 normalize_features: bool = True):
        super().__init__(normalize_features=normalize_features)
        self.m = torch.hub.load('NVlabs/RADIO', 'radio_model',
                                version=version, progress=False).eval()
        vit = getattr(self.m, 'model', self.m)
        if not hasattr(vit, 'blocks'):
            raise AttributeError('Cannot find .blocks in RADIO model')
        n = len(vit.blocks)
        self._layer_idx = LAYER_SCHEDULE.get(n) or [
            n // 4 - 1, n // 2 - 1, 3 * n // 4 - 1, n - 1
        ]
        self._dim = vit.embed_dim
        self.patch = getattr(vit, 'patch_size', 16)
        self._vit = vit

    @property
    def embed_dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def extract_features(self, pixel_values):
        b, _, h, w = pixel_values.shape
        ph, pw = h // self.patch, w // self.patch
        x = pixel_values.to(next(self.m.parameters()).dtype)
        feats = self._vit.forward_intermediates(
            x, indices=self._layer_idx,
            output_fmt='NLC', intermediates_only=True)   # <-- True
        return [self._tokens_to_map(f, ph, pw) for f in feats]


# ── 2f  SAM ViT-B image encoder ─────────────────────────────────────────────
# SAM's patch_embed hard-checks input size and pos_embed is a fixed [1,64,64,768]
# parameter.  We adapt it to IMG_SIZE at init by:
#   1. patching patch_embed.image_size so the size check passes
#   2. bilinearly interpolating pos_embed from 64×64 → (IMG_SIZE//16)×(IMG_SIZE//16)
# Hidden_states from hooks are NHWC; we permute to NCHW.  embed_dim = hidden_size (768).


class SAMAdapter(BackboneAdapter):
    """SAM ViT-B image encoder  —  facebook/sam-vit-base."""
    name = 'SAM ViT-B'
    patch = 16

    def __init__(self, model_id: str = 'facebook/sam-vit-base',
                 img_size: int = 448, normalize_features: bool = True):
        super().__init__(normalize_features=normalize_features)
        sam = SamModel.from_pretrained(model_id).eval()
        self.encoder = sam.vision_encoder
        self._dim = self.encoder.config.hidden_size
        self._layers = LAYER_SCHEDULE[self.encoder.config.num_hidden_layers]

        # Adapt pos_embed and size check to img_size
        grid = img_size // self.patch
        if self.encoder.pos_embed.shape[1] != grid:
            orig = self.encoder.pos_embed.data          # [1, 64, 64, 768]
            resized = F.interpolate(
                orig.permute(0, 3, 1, 2).float(),
                size=(grid, grid), mode='bilinear', align_corners=False,
            ).permute(0, 2, 3, 1).to(orig.dtype)
            self.encoder.pos_embed = nn.Parameter(resized, requires_grad=False)
        self.encoder.patch_embed.image_size = (img_size, img_size)

    @property
    def embed_dim(self) -> int:
        return self._dim

    @torch.no_grad()
    def extract_features(self, pixel_values: torch.Tensor) -> List[torch.Tensor]:
        hs = self.encoder(
            pixel_values, output_hidden_states=True).hidden_states
        # hidden_states: tuple of NHWC tensors [B, H, W, C], one per layer
        maps = [hs[i].permute(0, 3, 1, 2).contiguous() for i in self._layers]
        return maps
