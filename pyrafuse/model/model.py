"""Segmentation model: a DINOv3 multi-scale backbone + a TPA-SAD dense decoder.

The backbone (:class:`DINOv3MultiScaleEmbed`) wraps a HuggingFace DINOv3
``AutoModel`` and emits multi-scale patch feature maps; it mirrors the wrapper
in ``scripts/export_dinov3_onnx.py`` so the trained backbone exports cleanly to
ONNX/TensorRT. The decoder (:class:`TPASADDecoder`) is the SegDINO-v2 dense head
(see ``other_projects/segdino_v2-main/dpt.py``), refit to consume spatial
feature maps instead of raw token sequences.

The top-level :class:`DINOv3Segmenter` ties them together, carries a
:class:`ModelConfig`, keeps an optional EMA shadow of its weights, and
saves/loads everything — including the DINOv3 backbone — to a local checkpoint
directory under ``models/checkpoints``.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import AutoModel

# ViT patch size: dynamic H/W must be a multiple of this (16 for the *16 models).
PATCH = 16

# HF token, read from the environment (HF_TOKEN / HUGGING_FACE_HUB_TOKEN).
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get(
    "HUGGING_FACE_HUB_TOKEN")

# Which transformer blocks to read for multi-scale features, per model depth.
# Mirrors other_projects/segdino*/dpt.py (get_intermediate_layers n=[...]).
# HF returns hidden_states[0] = embeddings (pre-block-0), so the output of
# block i lives at hidden_states[i + 1]; we add 1 when indexing below.
INTERMEDIATE_LAYER_IDX = {
    12: [2, 5, 8, 11],    # ViT-S / ViT-B (12 blocks)
    24: [4, 11, 17, 23],  # ViT-L (24 blocks)
}

# Subdirectory (inside a checkpoint) holding the HF DINOv3 backbone weights.
BACKBONE_SUBDIR = "dinov3"


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------
class DINOv3MultiScaleEmbed(nn.Module):
    """Emit multi-scale patch feature maps for a dense (DPT-style) decoder.

    Output: ``[B, num_levels, C, H/16, W/16]`` — one named input, one named
    output, so the ONNX graph stays clean. Unbind dim 1 to recover the
    per-level maps.
    """

    def __init__(self, m: nn.Module, patch: int = PATCH):
        super().__init__()
        self.m = m
        self.patch = patch
        n_blocks = m.config.num_hidden_layers
        if n_blocks not in INTERMEDIATE_LAYER_IDX:
            raise ValueError(
                f"No layer schedule for a {n_blocks}-block model.")
        self.layer_idx = INTERMEDIATE_LAYER_IDX[n_blocks]
        # Prefix tokens before the patch grid: 1 CLS + N register tokens.
        self.num_prefix = 1 + getattr(m.config, "num_register_tokens", 0)

    @property
    def embed_dim(self) -> int:
        return self.m.config.hidden_size

    @property
    def num_levels(self) -> int:
        return len(self.layer_idx)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        b, _, h, w = pixel_values.shape
        ph, pw = h // self.patch, w // self.patch
        # When the backbone is frozen (no param needs grad), run it under
        # no_grad so none of the ViT's activations are retained for backprop.
        # This is the dominant training-memory cost; the decoder still trains
        # normally on the detached features. When unfrozen for end-to-end
        # fine-tuning, the graph is kept so gradients flow into the backbone.
        frozen = not any(p.requires_grad for p in self.m.parameters())
        ctx = torch.no_grad() if frozen else torch.enable_grad()
        with ctx:
            hs = self.m(pixel_values=pixel_values,
                        output_hidden_states=True).hidden_states
            levels = []
            for i in self.layer_idx:
                # drop CLS + register
                patches = hs[i + 1][:, self.num_prefix:]
                levels.append(patches.transpose(1, 2).reshape(b, -1, ph, pw))
            out = torch.stack(levels, dim=1)  # [B, num_levels, C, H/16, W/16]
        return out


# ---------------------------------------------------------------------------
# Decoder (TPA-SAD, from segdino_v2)
# ---------------------------------------------------------------------------
def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Per-sample stochastic depth: zero the whole tensor for some batch items.

    Operates on the residual branch only; survivors are scaled by
    ``1 / (1 - drop_prob)`` so the expected value is unchanged.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    # broadcast mask over all non-batch dims
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep)
    return x * mask / keep


class ResidualDepthwiseBlock(nn.Module):
    def __init__(self, channels, use_group_norm=True, drop_prob=0.0, dropout=0.0):
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels, channels, 3, padding=1, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = (
            nn.GroupNorm(min(32, channels), channels)
            if use_group_norm
            else nn.BatchNorm2d(channels)
        )
        self.act = nn.GELU()
        # Channel (spatial) dropout on the residual features; no-op when 0.
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.drop_prob = drop_prob
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        residual = self.act(self.norm(self.pointwise(self.depthwise(x))))
        residual = self.dropout(residual)
        return x + drop_path(self.gamma * residual, self.drop_prob, self.training)


class TPAResampleProject(nn.Module):
    def __init__(self, channels, scale_factor):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

    def forward(self, x):
        if self.scale_factor != 1:
            x = F.interpolate(
                x,
                scale_factor=self.scale_factor,
                mode="bilinear",
                align_corners=False,
            )
        return self.conv(x)


class TPASADDecoder(nn.Module):
    def __init__(self, in_dims, decoder_channels=128, num_classes=2,
                 use_group_norm=True, drop_path_rate=0.0, dropout=0.0):
        super().__init__()
        assert len(in_dims) == 4

        # Linearly ramp drop-path across the 8 SAD blocks (intra 1-4, inter 4-1).
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 8)]
        def rb(i): return ResidualDepthwiseBlock(
            decoder_channels, use_group_norm=use_group_norm, drop_prob=dpr[i])

        # TPA: project backbone features into decoder channels and align them to
        # the four spatial branches used by the decoder.
        self.token_projections = nn.ModuleList(
            [nn.Conv2d(channels, decoder_channels, 1, bias=False)
             for channels in in_dims]
        )
        self.tpa_branch_1 = TPAResampleProject(
            decoder_channels, scale_factor=8)
        self.tpa_branch_2 = TPAResampleProject(
            decoder_channels, scale_factor=4)
        self.tpa_branch_3 = TPAResampleProject(
            decoder_channels, scale_factor=2)
        self.tpa_branch_4 = TPAResampleProject(
            decoder_channels, scale_factor=1)

        # SAD: refine each branch independently, then fuse them from coarse to fine.
        self.sad_intra_1 = rb(0)
        self.sad_intra_2 = rb(1)
        self.sad_intra_3 = rb(2)
        self.sad_intra_4 = rb(3)

        self.sad_inter_4 = rb(4)
        self.sad_inter_3 = rb(5)
        self.sad_inter_2 = rb(6)
        self.sad_inter_1 = rb(7)

        # needs num_groups <= num_channels, hence the max(1, ...).
        # refine_channels = max(1, decoder_channels // 2)
        # self.refine_block = ResidualDepthwiseBlock(
        #     decoder_channels, use_group_norm=use_group_norm
        # )
        # self.refine_up = nn.ConvTranspose2d(
        #     decoder_channels, refine_channels, kernel_size=2, stride=2
        # )

        self.head_dropout = nn.Dropout2d(
            dropout) if dropout > 0 else nn.Identity()
        self.out_conv = nn.Conv2d(decoder_channels, num_classes, 1)

    def forward(self, feature_maps):
        # feature_maps: list of four [B, C, ph, pw] tensors (coarse-to-fine input).
        # TPA: project each level, then resample into the four-branch pyramid.
        branches = [proj(fmap) for proj, fmap in zip(
            self.token_projections, feature_maps)]

        branch_1 = self.tpa_branch_1(branches[0])
        branch_2 = self.tpa_branch_2(branches[1])
        branch_3 = self.tpa_branch_3(branches[2])
        branch_4 = self.tpa_branch_4(branches[3])

        # SAD: each branch is refined, then merged top-down.
        level_1 = self.sad_intra_1(branch_1)
        level_2 = self.sad_intra_2(branch_2)
        level_3 = self.sad_intra_3(branch_3)
        level_4 = self.sad_intra_4(branch_4)

        x4 = self.sad_inter_4(level_4)
        x3_up = F.interpolate(
            x4, size=level_3.shape[-2:], mode="bilinear", align_corners=False)
        x3 = self.sad_inter_3(x3_up + level_3)

        x2_up = F.interpolate(
            x3, size=level_2.shape[-2:], mode="bilinear", align_corners=False)
        x2 = self.sad_inter_2(x2_up + level_2)

        x1_up = F.interpolate(
            x2, size=level_1.shape[-2:], mode="bilinear", align_corners=False)
        x1 = self.sad_inter_1(x1_up + level_1)

        # x1 = self._refine_head(x1)
        logits = self.out_conv(self.head_dropout(x1))
        return F.interpolate(logits, scale_factor=2, mode="bilinear", align_corners=False)

    # def _refine_head(self, x1):
    #     x1 = self.refine_block(x1)          # refine at coarse grid (cheap)
    #     return self.refine_up(x1)           # learned x2 -> fine grid


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """Configuration for :class:`DINOv3Segmenter`.

    Stored alongside the weights so a checkpoint reconstructs the exact model.
    """

    model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    num_classes: int = 2
    decoder_channels: int = 128
    patch_size: int = PATCH
    use_bn: bool = False           # GroupNorm when False, BatchNorm when True
    drop_path_rate: float = 0.05    # stochastic depth, ramped across SAD blocks
    dropout: float = 0.1           # spatial (channel) dropout in the decoder
    freeze_backbone: bool = True
    ema_decay: float = 0.999       # set <= 0 to disable EMA

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        fields = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in fields})


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
class ModelEMA:
    """Exponential moving average of a module's parameters and buffers.

    Keeps a detached deep copy on the same device; call :meth:`update` after
    each optimizer step. Swap it in for evaluation via :meth:`copy_to` /
    :meth:`state_dict`.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999,
                 warmup_steps: int = 1000):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.num_updates = 0
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.num_updates += 1
        # Warm up the decay so the EMA tracks the (fast-moving) early weights
        # closely instead of lagging behind a near-1.0 decay from step 0.
        # Ramps 0 -> self.decay linearly over warmup_steps.
        if self.warmup_steps > 0 and self.num_updates < self.warmup_steps:
            d = self.decay * (self.num_updates / self.warmup_steps)
        else:
            d = self.decay
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p.detach(), alpha=1.0 - d)
        # Buffers (e.g. BN running stats) are copied, not averaged.
        for ema_b, b in zip(self.ema.buffers(), model.buffers()):
            ema_b.copy_(b)

    @torch.no_grad()
    def copy_to(self, model: nn.Module) -> None:
        model.load_state_dict(self.ema.state_dict())

    def state_dict(self) -> dict:
        return self.ema.state_dict()

    def load_state_dict(self, sd: dict) -> None:
        self.ema.load_state_dict(sd)


# ---------------------------------------------------------------------------
# Segmenter
# ---------------------------------------------------------------------------
class DINOv3Segmenter(nn.Module):
    """DINOv3 multi-scale backbone + TPA-SAD decoder for dense segmentation.

    Forward takes ``pixel_values`` ``[B, 3, H, W]`` (H, W multiples of patch
    size) and returns logits ``[B, num_classes, H, W]``.
    """

    def __init__(self, config: ModelConfig, backbone: DINOv3MultiScaleEmbed):
        super().__init__()
        self.config = config
        self.backbone = backbone

        in_dims = [backbone.embed_dim] * backbone.num_levels
        self.decoder = TPASADDecoder(
            in_dims,
            decoder_channels=config.decoder_channels,
            num_classes=config.num_classes,
            use_group_norm=not config.use_bn,
            drop_path_rate=config.drop_path_rate,
            dropout=config.dropout,
        )

        if config.freeze_backbone:
            self.lock_backbone()

        self.ema: Optional[ModelEMA] = None

    # -- construction --------------------------------------------------------
    @classmethod
    def from_pretrained_backbone(cls, config: ModelConfig) -> "DINOv3Segmenter":
        """Build with a fresh DINOv3 backbone pulled from the HF hub."""
        base = AutoModel.from_pretrained(
            config.model_id, token=HF_TOKEN).eval()
        backbone = DINOv3MultiScaleEmbed(base, patch=config.patch_size)
        return cls(config, backbone)

    # -- backbone control ----------------------------------------------------
    def lock_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

    def unlock_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    # -- EMA -----------------------------------------------------------------
    def init_ema(self, decay: Optional[float] = None,
                 warmup_steps: int = 1000) -> Optional[ModelEMA]:
        decay = self.config.ema_decay if decay is None else decay
        self.ema = (ModelEMA(self, decay, warmup_steps)
                    if decay and decay > 0 else None)
        return self.ema

    def update_ema(self) -> None:
        if self.ema is not None:
            self.ema.update(self)

    # -- forward -------------------------------------------------------------
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        levels: Optional[torch.Tensor] = None,
        out_size: Optional[tuple] = None,
    ) -> torch.Tensor:
        """Run the decoder, optionally skipping the backbone.

        Pass ``pixel_values`` ``[B, 3, H, W]`` to run end-to-end, or pass
        pre-computed backbone ``levels`` ``[B, L, C, ph, pw]`` (e.g. from a
        separately exported/TRT backbone) to run the decoder only.

        ``out_size`` controls the final upsample:

        * ``(H, W)`` — upsample logits to this size (use this to upsample
          exactly once to the caller's target, e.g. the display size).
        * ``None`` — default to the ``pixel_values`` input size (requires
          ``pixel_values``).
        """
        if levels is None:
            if pixel_values is None:
                raise ValueError("Provide either `pixel_values` or `levels`.")
            levels = self.backbone(pixel_values)        # [B, L, C, ph, pw]

        feature_maps = list(levels.unbind(dim=1))       # L x [B, C, ph, pw]
        logits = self.decoder(feature_maps)

        if out_size is None:
            if pixel_values is None:
                raise ValueError(
                    "Provide `out_size` when passing `levels` without `pixel_values`.")
            out_size = pixel_values.shape[-2:]

        return F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)

    # -- save / load ---------------------------------------------------------
    def save_pretrained(self, save_dir, save_ema: bool = True) -> Path:
        """Save config, decoder weights, EMA, and the HF DINOv3 backbone locally.

        Layout::

            <save_dir>/
                config.json
                decoder.pt          # decoder state_dict
                ema.pt              # full-model EMA state_dict (if present)
                dinov3/             # HF backbone (save_pretrained)
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        (save_dir / "config.json").write_text(json.dumps(self.config.to_dict(), indent=2))
        torch.save(self.decoder.state_dict(), save_dir / "decoder.pt")

        # Persist the DINOv3 backbone in native HF format so it reloads with
        # AutoModel.from_pretrained(<dir>/dinov3) — no hub access needed.
        self.backbone.m.save_pretrained(save_dir / BACKBONE_SUBDIR)

        if save_ema and self.ema is not None:
            torch.save(self.ema.state_dict(), save_dir / "ema.pt")

        return save_dir

    @classmethod
    def from_pretrained(
        cls,
        load_dir,
        map_location: str = "cpu",
        load_ema_into_model: bool = False,
    ) -> "DINOv3Segmenter":
        """Reconstruct a saved segmenter (backbone + decoder + EMA) from disk."""
        load_dir = Path(load_dir)
        config = ModelConfig.from_dict(json.loads(
            (load_dir / "config.json").read_text()))

        # Reload the DINOv3 backbone from the local copy, falling back to the hub.
        backbone_dir = load_dir / BACKBONE_SUBDIR
        source = str(backbone_dir) if backbone_dir.exists(
        ) else config.model_id
        base = AutoModel.from_pretrained(source, token=HF_TOKEN).eval()
        backbone = DINOv3MultiScaleEmbed(base, patch=config.patch_size)

        model = cls(config, backbone)
        model.decoder.load_state_dict(
            torch.load(load_dir / "decoder.pt", map_location=map_location,
                       weights_only=True)
        )

        ema_path = load_dir / "ema.pt"
        if ema_path.exists():
            ema_sd = torch.load(ema_path, map_location=map_location,
                                weights_only=True)
            if load_ema_into_model:
                model.load_state_dict(ema_sd)
            else:
                model.init_ema()
                if model.ema is not None:
                    model.ema.load_state_dict(ema_sd)

        return model


# Default local directory for checkpoints (mirrors models/checkpoints layout).
DEFAULT_CHECKPOINT_DIR = Path("models/checkpoints")


def build_segmenter(config: Optional[ModelConfig] = None) -> DINOv3Segmenter:
    """Convenience factory: build a segmenter with a fresh HF backbone + EMA."""
    config = config or ModelConfig()
    model = DINOv3Segmenter.from_pretrained_backbone(config)
    model.init_ema()
    return model
