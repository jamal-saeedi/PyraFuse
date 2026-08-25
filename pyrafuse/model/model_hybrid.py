"""Hybrid segmentation model: pluggable encoder + pluggable decoder.

Encoders (from ``encoders.py``):
    ``dinov3``   DINOv3 ViT-B/16  (default)
    ``dinov2``   DINOv2 ViT-S/14
    ``clip``     CLIP ViT-B/16
    ``siglip``   SigLIP ViT-B/16
    ``evaclip``  EVA-CLIP ViT-B/16
    ``radio``    AM-RADIO-B
    ``sam``      SAM ViT-B

Decoders (from ``decoders.py``):
    ``tpa_sad``      TPA-SAD (SegDINO-v2 default, scale-factor pyramid)
    ``pyrafuse``     PyraFuse (TPA-SAD + SE gates + LightPPM)
    ``dpt``          DPT (4-level reassemble + top-down 2× fusion)
    ``allmlp``       SegFormer all-MLP head
    ``upernet``      UPerNet (PPM + FPN)
    ``mask2former``  simplified Mask2Former (pixel decoder + transformer queries)

``ModelConfig`` drives everything — encoder type, decoder type, image size,
channel widths, regularisation.  ``HybridSegmenter.from_pretrained_backbone``
and ``HybridSegmenter.from_pretrained`` are the public constructors.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── encoder adapters ──────────────────────────────────────────────────────────
from .encoders import (
    BackboneAdapter,
    DINOv3Adapter,
    DINOv2Adapter,
    CLIPAdapter,
    SigLIPAdapter,
    EVACLIPAdapter,
    RADIOAdapter,
    SAMAdapter,
)

# ── decoder heads ─────────────────────────────────────────────────────────────
from .decoders import (
    TPASADDecoder,
    PyraFuseDecoder,
    DPTDecoder,
    AllMLPDecoder,
    UPerNetDecoder,
    Mask2FormerHead,
)

HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get(
    "HUGGING_FACE_HUB_TOKEN")

# Subdirectory (inside a checkpoint) holding the serialised backbone weights.
BACKBONE_SUBDIR = "backbone"

# Default encoder model IDs per encoder type -- single source of truth shared
# by train_segmenter_hybrid.py and benchmark_segmenter_hybrid.py.
DEFAULT_MODEL_IDS: dict[str, str] = {
    "dinov3":  "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov2":  "facebook/dinov2-small",
    "clip":    "openai/clip-vit-base-patch16",
    "siglip":  "google/siglip-base-patch16-224",
    "evaclip": "EVA02-B-16",
    "radio":   "radio_v2.5-b",
    "sam":     "facebook/sam-vit-base",
}

# Short-name convenience map for DINOv3 pretrained backbone sizes (all
# LVD-1689M pretrain). All of s/s_plus/b share 12 transformer blocks, so they
# use the same LAYER_SCHEDULE entry and drop into DINOv3Adapter unmodified;
# embed_dim (384/384/768/1024) is read from the HF config at load time and
# propagates automatically into the decoder's per-scale projections.
DINOV3_SIZES: dict[str, str] = {
    "s":      "facebook/dinov3-vits16-pretrain-lvd1689m",
    "s_plus": "facebook/dinov3-vits16plus-pretrain-lvd1689m",
    "b":      "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "l":      "facebook/dinov3-vitl16-pretrain-lvd1689m",
}
_DINOV3_SIZES_REVERSE = {v: k for k, v in DINOV3_SIZES.items()}


def encoder_variant_tag(encoder_type: str, encoder_model_id: str) -> str:
    """Folder-name-safe tag identifying an encoder *and* its size/variant.

    Returns just ``encoder_type`` when ``encoder_model_id`` is the standard
    default for that encoder type -- this preserves existing checkpoint /
    benchmark paths (e.g. ``dinov3_pyrafuse_dice`` for the default ViT-B/16
    run). Otherwise appends a short size tag, e.g. ``dinov3-s_plus`` for
    ViT-S+/16, so differently-sized runs of the same encoder family never
    collide on disk.
    """
    if encoder_model_id == DEFAULT_MODEL_IDS.get(encoder_type):
        return encoder_type
    size_key = _DINOV3_SIZES_REVERSE.get(encoder_model_id)
    if size_key is None:
        size_key = encoder_model_id.rstrip("/").rsplit("/", 1)[-1].replace(".", "-")
    return f"{encoder_type}-{size_key}"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """All hyper-parameters needed to reconstruct a ``HybridSegmenter``.

    Stored as ``config.json`` next to the weights so a checkpoint is
    fully self-contained.
    """

    # Encoder
    encoder_type: str = "dinov3"
    encoder_model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    # Text-conditioned gating (MaskCLIP/CLIPSeg), VLM encoders only
    # (clip/siglip/evaclip). When set, build_encoder arms the text gate so it
    # runs automatically on every forward. Ignored by non-VLM encoders.
    text_prompts: List[str] | None = None

    # Decoder
    decoder_type: str = "tpa_sad"

    # Common
    num_classes: int = 2
    decoder_channels: int = 128
    image_size: int = 448          # H == W; input must be a multiple of patch size

    # TPA-SAD / PyraFuse regularisation (ignored by other decoders)
    use_bn: bool = False
    drop_path_rate: float = 0.05
    dropout: float = 0.1

    # Mask2Former only
    num_queries: int = 10
    num_transformer_layers: int = 3

    # Training
    freeze_backbone: bool = True
    ema_decay: float = 0.999

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


# ---------------------------------------------------------------------------
# Encoder factory
# ---------------------------------------------------------------------------
_ENCODER_REGISTRY: dict[str, type] = {
    "dinov3":   DINOv3Adapter,
    "dinov2":   DINOv2Adapter,
    "clip":     CLIPAdapter,
    "siglip":   SigLIPAdapter,
    "evaclip":  EVACLIPAdapter,
    "radio":    RADIOAdapter,
    "sam":      SAMAdapter,
}


def build_encoder(config: ModelConfig) -> BackboneAdapter:
    """Instantiate the encoder specified by *config.encoder_type*."""
    key = config.encoder_type.lower()
    if key not in _ENCODER_REGISTRY:
        raise ValueError(
            f"Unknown encoder '{key}'. "
            f"Choose from: {sorted(_ENCODER_REGISTRY)}"
        )
    cls = _ENCODER_REGISTRY[key]

    # Adapters that take an image_size at construction (SAM, EVA-CLIP).
    size_aware = {"sam", "evaclip"}
    if key in size_aware:
        enc = cls(model_id=config.encoder_model_id, img_size=config.image_size)
    elif key == "radio":
        # RADIO takes a version string, not a HF model id.
        enc = cls(version=config.encoder_model_id)
    else:
        enc = cls(model_id=config.encoder_model_id)

    # Build the per-encoder feature-norm eagerly so its parameters exist before
    # the optimizer is constructed (they are lazily skipped otherwise).
    enc.init_feature_norms()

    # Arm text-conditioned gating for VLM encoders (no-op / unsupported on the
    # rest). Done here so it is applied identically on first build and on
    # reload — the prompts live in the config, not the checkpoint weights.
    if config.text_prompts:
        if not hasattr(enc, "set_text_prompts"):
            raise ValueError(
                f"text_prompts set but encoder '{key}' has no text tower; "
                f"text gating is only supported for clip/siglip/evaclip."
            )
        enc.set_text_prompts(config.text_prompts)
    return enc


# ---------------------------------------------------------------------------
# Decoder factory
# ---------------------------------------------------------------------------
def build_decoder(config: ModelConfig, encoder: BackboneAdapter) -> nn.Module:
    """Instantiate the decoder specified by *config.decoder_type*.

    All decoders receive the same ``in_channels_list`` derived from the
    encoder so the shapes always match, regardless of which encoder is used.
    """
    key = config.decoder_type.lower()
    dc = config.decoder_channels
    nc = config.num_classes
    # All current encoder adapters return 4 uniform-channel feature maps.
    in_dims: List[int] = [encoder.embed_dim] * 4

    if key == "tpa_sad":
        return TPASADDecoder(
            in_dims,
            decoder_channels=dc,
            num_classes=nc,
            use_group_norm=not config.use_bn,
            drop_path_rate=config.drop_path_rate,
            dropout=config.dropout,
        )
    if key == "pyrafuse":
        return PyraFuseDecoder(in_dims, decoder_channels=dc, num_classes=nc)
    if key == "dpt":
        return DPTDecoder(
            in_channels=encoder.embed_dim,
            decoder_channels=dc,
            num_classes=nc,
        )
    if key == "allmlp":
        return AllMLPDecoder(in_dims, embed_dim=dc, num_classes=nc)
    if key == "upernet":
        return UPerNetDecoder(in_dims, decoder_channels=dc, num_classes=nc)
    if key == "mask2former":
        return Mask2FormerHead(
            in_channels=encoder.embed_dim,
            decoder_channels=dc,
            num_classes=nc,
            num_queries=config.num_queries,
            num_layers=config.num_transformer_layers,
        )
    raise ValueError(
        f"Unknown decoder '{key}'. "
        f"Choose from: tpa_sad, pyrafuse, dpt, allmlp, upernet, mask2former"
    )


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------
class ModelEMA:
    """Exponential moving average of a module's parameters and buffers.

    Keeps a detached deep copy on the same device; call :meth:`update` after
    each optimiser step. Swap in for evaluation via :meth:`copy_to` /
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
        d = (self.decay * (self.num_updates / self.warmup_steps)
             if self.warmup_steps > 0 and self.num_updates < self.warmup_steps
             else self.decay)
        for ema_p, p in zip(self.ema.parameters(), model.parameters()):
            ema_p.mul_(d).add_(p.detach(), alpha=1.0 - d)
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
class HybridSegmenter(nn.Module):
    """Pluggable encoder + decoder for dense segmentation.

    Forward accepts ``pixel_values`` ``[B, 3, H, W]`` (H and W must be
    multiples of the encoder's patch size) and returns logits
    ``[B, num_classes, H, W]``.

    Alternatively pass pre-computed ``feature_maps`` — a list of 4 tensors
    ``[B, C, ph, pw]`` — to run only the decoder (e.g. for profiling or when
    the encoder runs on a separate device).
    """

    def __init__(self, config: ModelConfig, encoder: BackboneAdapter):
        super().__init__()
        self.config = config
        self.encoder = encoder
        self.decoder = build_decoder(config, encoder)

        if config.freeze_backbone:
            self.lock_encoder()

        self.ema: Optional[ModelEMA] = None

    # -- construction --------------------------------------------------------

    @classmethod
    def from_pretrained_backbone(cls, config: ModelConfig) -> "HybridSegmenter":
        """Build with a fresh encoder pulled from the hub (or timm/torch.hub)."""
        encoder = build_encoder(config)
        return cls(config, encoder)

    # -- encoder control -----------------------------------------------------

    @property
    def backbone(self) -> BackboneAdapter:
        """Alias for ``encoder`` — keeps the Trainer compatible with both model classes."""
        return self.encoder

    def lock_encoder(self) -> None:
        """Freeze the backbone but keep the per-encoder feature-norm trainable.

        The adapter applies a channel-wise LayerNorm to its output feature maps
        (see ``BackboneAdapter``).  That norm is a cheap learnable calibration
        layer that lets the decoder adapt to this backbone's feature statistics
        without unfreezing the (expensive, pretrained) backbone — so it stays
        trainable even when the encoder is "locked".  ``feature_norms`` may be
        ``None`` here because it is built lazily on the first forward; the
        adapter re-applies this frozen state when it builds the norm.
        """
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        self.encoder.set_backbone_frozen(True)

    def unlock_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad_(True)
        self.encoder.set_backbone_frozen(False)

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
        feature_maps: Optional[List[torch.Tensor]] = None,
        out_size: Optional[tuple] = None,
    ) -> torch.Tensor:
        """Run end-to-end, or decoder-only when *feature_maps* are provided.

        Args:
            pixel_values: ``[B, 3, H, W]`` — run encoder + decoder.
            feature_maps: pre-computed list of 4 ``[B, C, ph, pw]`` tensors —
                skip the encoder.  Requires *out_size*.
            out_size: ``(H, W)`` to upsample logits to.  Defaults to the
                spatial size of *pixel_values* when that is provided.
        """
        if feature_maps is None:
            if pixel_values is None:
                raise ValueError(
                    "Provide either `pixel_values` or `feature_maps`.")
            # Encoder adapters use @torch.no_grad internally when all params
            # are frozen; no need to wrap here.
            feature_maps = self.encoder(pixel_values)

        logits = self.decoder(feature_maps)

        if out_size is None:
            if pixel_values is None:
                raise ValueError(
                    "Provide `out_size` when passing `feature_maps` without "
                    "`pixel_values`.")
            out_size = pixel_values.shape[-2:]

        if logits.shape[-2:] != torch.Size(out_size):
            logits = F.interpolate(
                logits, size=out_size, mode="bilinear", align_corners=False)
        return logits

    # -- save / load ---------------------------------------------------------

    def save_pretrained(self, save_dir, save_ema: bool = True) -> Path:
        """Save config, decoder weights, EMA, and the encoder to *save_dir*.

        Layout::

            <save_dir>/
                config.json
                decoder.pt
                ema.pt              (if EMA was initialised)
                backbone/           (HF save_pretrained for HF-based encoders;
                                     torch.save state_dict otherwise)
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        (save_dir / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2))
        torch.save(self.decoder.state_dict(), save_dir / "decoder.pt")

        # Persist encoder weights.  HF-based adapters expose .m with
        # save_pretrained; timm/hub adapters fall back to state_dict.
        backbone_dir = save_dir / BACKBONE_SUBDIR
        inner = getattr(self.encoder, "m", None) or getattr(
            self.encoder, "encoder", None)
        if inner is not None and hasattr(inner, "save_pretrained"):
            inner.save_pretrained(backbone_dir)
            # The HF path saves only the inner backbone, so persist the
            # adapter-level feature-norm (trainable calibration layer) too.
            if getattr(self.encoder, "feature_norms", None) is not None:
                backbone_dir.mkdir(parents=True, exist_ok=True)
                torch.save(self.encoder.feature_norms.state_dict(),
                           backbone_dir / "feature_norms.pt")
        else:
            backbone_dir.mkdir(parents=True, exist_ok=True)
            # state_dict() includes feature_norms for the non-HF path.
            torch.save(self.encoder.state_dict(),
                       backbone_dir / "encoder.pt")

        if save_ema and self.ema is not None:
            torch.save(self.ema.state_dict(), save_dir / "ema.pt")

        return save_dir

    @classmethod
    def from_pretrained(
        cls,
        load_dir,
        map_location: str = "cpu",
        load_ema_into_model: bool = False,
    ) -> "HybridSegmenter":
        """Reconstruct a saved segmenter from *load_dir*."""
        load_dir = Path(load_dir)
        config = ModelConfig.from_dict(
            json.loads((load_dir / "config.json").read_text()))

        backbone_dir = load_dir / BACKBONE_SUBDIR
        # Prefer the checkpoint's self-contained Hugging Face backbone when it
        # is present.  Constructing the adapter with config.encoder_model_id
        # first used to make every local checkpoint attempt a Hub download
        # before its saved `model.safetensors` was restored.  Apart from being
        # slow, that made offline inference impossible.  The standard HF
        # adapters all accept a local save_pretrained directory here.
        encoder_config = config
        if (backbone_dir / "config.json").is_file():
            encoder_config = replace(
                config, encoder_model_id=str(backbone_dir.resolve())
            )
        encoder = build_encoder(encoder_config)

        # Restore a non-HF encoder state dict, or apply the local HF weights
        # explicitly for adapter implementations that need their wrapper
        # state reconciled after construction.
        if backbone_dir.exists():
            pt_file = backbone_dir / "encoder.pt"
            if pt_file.exists():
                encoder.load_state_dict(
                    torch.load(pt_file, map_location=map_location,
                               weights_only=True))
            else:
                # HF-serialised backbone — reload via AutoModel / inner module.
                from transformers import AutoModel as _AM
                inner = getattr(encoder, "m", None) or getattr(
                    encoder, "encoder", None)
                if inner is not None and hasattr(inner, "from_pretrained"):
                    # Load the raw weights directly into inner rather than via
                    # AutoModel.from_pretrained, which would (a) re-download hub
                    # weights and (b) produce a wrapper model whose state_dict
                    # has an extra prefix (e.g. SamVisionModel → "vision_encoder.*")
                    # and (c) reset any in-place modifications made by the adapter
                    # constructor (e.g. SAMAdapter resizes pos_embed at init time).
                    from safetensors.torch import load_file as _load_sf
                    import os as _os
                    sf_path = backbone_dir / "model.safetensors"
                    if sf_path.exists():
                        sd = _load_sf(str(sf_path), device=map_location)
                        inner_keys = set(inner.state_dict().keys())
                        sd_keys = set(sd.keys())
                        overlap = len(inner_keys.intersection(sd_keys))
                        # DINOv3ViTModel stores transformer layers under a "model." prefix
                        # in state_dict, but save_pretrained() omits it.  Only add the
                        # prefix to keys that actually need it (i.e. keys that exist in
                        # inner with "model." but not without).
                        model_prefixed = {}
                        prefixed_count = 0
                        for k, v in sd.items():
                            if k not in inner_keys and ("model." + k) in inner_keys:
                                model_prefixed["model." + k] = v
                                prefixed_count += 1
                            else:
                                model_prefixed[k] = v
                        overlap_prefixed = len(inner_keys.intersection(model_prefixed))
                        if overlap_prefixed > overlap:
                            sd = model_prefixed
                            overlap = overlap_prefixed
                        # If there is still no overlap, strip a possible wrapper prefix
                        # (e.g. SamVisionModel saves "vision_encoder.layers.*" but the
                        # inner SamVisionEncoder expects "layers.*").
                        if overlap == 0:
                            prefix = next(iter(sd)).split(".")[0] + "."
                            sd = {
                                (k[len(prefix):] if k.startswith(prefix) else k): v
                                for k, v in sd.items()
                            }
                            overlap = len(inner_keys.intersection(set(sd.keys())))
                    else:
                        # Fall back to AutoModel for adapters that don't save safetensors
                        loaded = _AM.from_pretrained(
                            str(backbone_dir), token=HF_TOKEN).eval()
                        sd = loaded.state_dict()
                        # Strip wrapper prefix if needed (e.g. SamVisionModel adds
                        # "vision_encoder." around SamVisionEncoder weights)
                        inner_keys = set(inner.state_dict().keys())
                        if not inner_keys.intersection(sd.keys()):
                            prefix = next(iter(sd)).split(".")[0] + "."
                            sd = {
                                (k[len(prefix):] if k.startswith(prefix) else k): v
                                for k, v in sd.items()
                            }
                    inner.load_state_dict(sd, strict=True)

            # Restore the adapter-level feature-norm saved alongside the HF backbone.
            fn_path = backbone_dir / "feature_norms.pt"
            if fn_path.exists() and getattr(encoder, "feature_norms", None) is not None:
                encoder.feature_norms.load_state_dict(
                    torch.load(fn_path, map_location=map_location,
                               weights_only=True))

        model = cls(config, encoder)
        model.decoder.load_state_dict(
            torch.load(load_dir / "decoder.pt", map_location=map_location,
                       weights_only=True))

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


# ---------------------------------------------------------------------------
# Convenience factories
# ---------------------------------------------------------------------------
DEFAULT_CHECKPOINT_DIR = Path("models/checkpoints")


def build_segmenter(config: Optional[ModelConfig] = None) -> HybridSegmenter:
    """Build a segmenter with a fresh encoder from the hub + EMA ready."""
    config = config or ModelConfig()
    model = HybridSegmenter.from_pretrained_backbone(config)
    model.init_ema()
    return model
