"""Segmentation losses for the skin / fabric / background task.

These mirror the reference implementations explored in
``notebooks/segmentation_losses_metrics.ipynb`` and bundle the recommended
combination (Focal + Dice) into a single configurable :class:`FocalDiceLoss`
module that the :class:`~deep_sunscreen.src.train.trainer.Trainer` consumes.

All functions take raw ``logits`` ``[B, C, H, W]`` and integer ``target``
``[B, H, W]``. Pixels equal to ``ignore_index`` are excluded.

When ``target`` is *soft* (one-hot / mixed, ``[B, C, H, W]`` float) — as
produced by MixUp/CutMix — pass it through and the losses fall back to a
soft formulation that ignores ``ignore_index`` (there is no integer label to
mask in that case).
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _is_soft(target: torch.Tensor, logits: torch.Tensor) -> bool:
    """Soft targets share logits' rank (B, C, H, W) and are floating point."""
    return target.dim() == logits.dim() and target.is_floating_point()


def _flatten_valid(logits, target, ignore_index):
    """Flatten to (N, C) / (N,), dropping ignored pixels. Hard targets only."""
    b, c = logits.shape[:2]
    logits = logits.permute(0, 2, 3, 1).reshape(-1, c)  # (N, C)
    target = target.reshape(-1)                          # (N,)
    valid = target != ignore_index
    return logits[valid], target[valid]


def _onehot(target, num_classes, ignore_index):
    """One-hot a hard target, zeroing ignored pixels. -> (B, C, H, W) float."""
    mask = target != ignore_index
    safe = torch.where(mask, target, torch.zeros_like(target))
    oh = F.one_hot(safe, num_classes).permute(0, 3, 1, 2).float()
    return oh * mask.unsqueeze(1)


# ---------------------------------------------------------------------------
# cross-entropy
# ---------------------------------------------------------------------------
def cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """(Optionally class-weighted) cross-entropy — the universal baseline.

    ``weight`` is an optional per-class tensor ``[C]``; pass inverse-frequency
    weights for the "Weighted CE" variant. Supports soft targets (one-hot /
    mixed), in which case ``ignore_index`` does not apply.
    """
    logits = logits.float()  # compute the loss in fp32 even under AMP autocast
    if _is_soft(target, logits):
        logp = F.log_softmax(logits, dim=1)              # (B, C, H, W)
        ce = -(target * logp)                            # (B, C, H, W)
        if weight is not None:
            ce = ce * weight.view(1, -1, 1, 1)
        return ce.sum(1).mean()
    return F.cross_entropy(
        logits, target, weight=weight, ignore_index=ignore_index)


# ---------------------------------------------------------------------------
# focal
# ---------------------------------------------------------------------------
def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Focal loss (Lin et al. 2017): down-weights easy pixels via ``(1-p_t)^gamma``.

    ``alpha`` is an optional per-class weight tensor ``[C]`` (helps with the
    dominant-background imbalance). Supports soft targets.
    """
    logits = logits.float()  # compute the loss in fp32 even under AMP autocast
    if _is_soft(target, logits):
        logp = F.log_softmax(logits, dim=1)              # (B, C, H, W)
        p = logp.exp()
        focal = (1 - p).clamp_min(0).pow(gamma) * (-logp)
        if alpha is not None:
            focal = focal * alpha.view(1, -1, 1, 1)
        # sum over classes (soft target weights), mean over pixels
        return (target * focal).sum(1).mean()

    flat_logits, flat_target = _flatten_valid(logits, target, ignore_index)
    if flat_target.numel() == 0:
        return logits.sum() * 0.0
    logp = F.log_softmax(flat_logits, dim=1)
    logp_t = logp.gather(1, flat_target[:, None]).squeeze(1)
    p_t = logp_t.exp()
    loss = -((1 - p_t).pow(gamma)) * logp_t
    if alpha is not None:
        loss = loss * alpha[flat_target]
    return loss.mean()


# ---------------------------------------------------------------------------
# dice
# ---------------------------------------------------------------------------
def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = -100,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Soft (macro) Dice ``= 1 - mean_c (2|X∩Y| + s) / (|X| + |Y| + s)``."""
    num_classes = logits.shape[1]
    probs = logits.float().softmax(dim=1)  # fp32 for numerical stability under AMP

    if _is_soft(target, logits):
        tgt = target
    else:
        tgt = _onehot(target, num_classes, ignore_index)
        probs = probs * (target != ignore_index).unsqueeze(1)  # mask preds too

    dims = (0, 2, 3)
    inter = (probs * tgt).sum(dims)
    card = probs.sum(dims) + tgt.sum(dims)
    dice = (2 * inter + smooth) / (card + smooth)
    return 1 - dice.mean()


# ---------------------------------------------------------------------------
# combined: CE + Dice
# ---------------------------------------------------------------------------
def dice_ce(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    smooth: float = 1.0,
    w_ce: float = 1.0,
    w_dice: float = 1.0,
) -> torch.Tensor:
    """DiceCE (nnU-Net's default): ``w_ce * CE + w_dice * Dice``."""
    return (
        w_ce * cross_entropy(logits, target, weight=weight,
                             ignore_index=ignore_index)
        + w_dice * dice_loss(logits, target, ignore_index=ignore_index,
                             smooth=smooth)
    )


# ---------------------------------------------------------------------------
# module wrappers
# ---------------------------------------------------------------------------
def _alpha_buffer(module: nn.Module, alpha: Optional[torch.Tensor]) -> None:
    """Register ``alpha`` as a float buffer (follows ``.to(device)``) or None."""
    if alpha is not None:
        module.register_buffer(
            "weight", torch.as_tensor(alpha, dtype=torch.float32))
    else:
        module.weight = None


class CrossEntropyLoss(nn.Module):
    """(Optionally class-weighted) cross-entropy.

    Pass inverse-frequency ``weight`` ``[C]`` for the "Weighted CE" variant;
    leave it ``None`` for plain CE. The weight is registered as a buffer so it
    follows ``.to(device)``.
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        _alpha_buffer(self, weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return cross_entropy(
            logits, target, weight=self.weight, ignore_index=self.ignore_index)

    def extra_repr(self) -> str:
        return f"weighted={self.weight is not None}, ignore_index={self.ignore_index}"


class FocalLoss(nn.Module):
    """Focal loss (Lin et al. 2017) with optional per-class ``alpha`` ``[C]``."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        if alpha is not None:
            self.register_buffer(
                "alpha", torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return focal_loss(
            logits, target, gamma=self.gamma, alpha=self.alpha,
            ignore_index=self.ignore_index)

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}, ignore_index={self.ignore_index}"


class DiceLoss(nn.Module):
    """Soft (macro) Dice loss."""

    def __init__(self, ignore_index: int = -100, smooth: float = 1.0):
        super().__init__()
        self.ignore_index = ignore_index
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_loss(
            logits, target, ignore_index=self.ignore_index, smooth=self.smooth)

    def extra_repr(self) -> str:
        return f"ignore_index={self.ignore_index}, smooth={self.smooth}"


class CEDiceLoss(nn.Module):
    """``w_ce * CE + w_dice * Dice`` — nnU-Net's robust general-purpose default.

    Carries an optional per-class ``weight`` ``[C]`` for the CE term (registered
    as a buffer so it follows ``.to(device)``).
    """

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        w_ce: float = 1.0,
        w_dice: float = 1.0,
        ignore_index: int = -100,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.w_ce = w_ce
        self.w_dice = w_dice
        self.ignore_index = ignore_index
        self.smooth = smooth
        _alpha_buffer(self, weight)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return dice_ce(
            logits, target, weight=self.weight, ignore_index=self.ignore_index,
            smooth=self.smooth, w_ce=self.w_ce, w_dice=self.w_dice,
        )

    def extra_repr(self) -> str:
        return (
            f"w_ce={self.w_ce}, w_dice={self.w_dice}, "
            f"weighted={self.weight is not None}, ignore_index={self.ignore_index}"
        )


# ---------------------------------------------------------------------------
# combined: Focal + Dice
# ---------------------------------------------------------------------------
class FocalDiceLoss(nn.Module):
    """``w_focal * focal + w_dice * dice`` — robust default for class-imbalanced
    segmentation. Carries an optional per-class ``alpha`` weight for focal.

    Args:
        gamma: focal focusing parameter.
        alpha: per-class weight ``[C]`` for focal (registered as a buffer so it
            follows ``.to(device)``).
        w_focal, w_dice: term weights.
        ignore_index: label value to exclude (hard targets only).
        smooth: dice Laplace smoothing.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        w_focal: float = 1.0,
        w_dice: float = 1.0,
        ignore_index: int = -100,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.w_focal = w_focal
        self.w_dice = w_dice
        self.ignore_index = ignore_index
        self.smooth = smooth
        if alpha is not None:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))
        else:
            self.alpha = None

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        focal = focal_loss(
            logits, target, gamma=self.gamma, alpha=self.alpha,
            ignore_index=self.ignore_index,
        )
        dice = dice_loss(
            logits, target, ignore_index=self.ignore_index, smooth=self.smooth,
        )
        return self.w_focal * focal + self.w_dice * dice

    def extra_repr(self) -> str:
        return (
            f"gamma={self.gamma}, w_focal={self.w_focal}, w_dice={self.w_dice}, "
            f"ignore_index={self.ignore_index}"
        )


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------
#: Selectable loss names accepted by :func:`build_loss` (and ``TrainConfig.loss_name``).
LOSS_NAMES = ("ce", "weighted_ce", "focal", "dice", "ce_dice", "focal_dice")


def build_loss(
    name: str,
    alpha: Optional[torch.Tensor] = None,
    gamma: float = 2.0,
    w_focal: float = 1.0,
    w_ce: float = 1.0,
    w_dice: float = 1.0,
    ignore_index: int = -100,
    smooth: float = 1.0,
) -> nn.Module:
    """Build a segmentation loss module by ``name``.

    The names mirror the empirical comparison in
    ``notebooks/segmentation_losses_metrics.ipynb``:

    * ``"ce"``           — plain cross-entropy.
    * ``"weighted_ce"``  — inverse-frequency class-weighted CE (needs ``alpha``).
    * ``"focal"``        — focal (``gamma``, optional inverse-freq ``alpha``).
    * ``"dice"``         — soft macro Dice only.
    * ``"ce_dice"``      — CE + Dice (DiceCE), optional CE ``alpha`` weights.
    * ``"focal_dice"``   — Focal + Dice (**the adopted default**).

    ``alpha`` is the per-class weight ``[C]`` (inverse-frequency); it feeds the
    CE/focal term of the weighted variants and is ignored by the unweighted ones.

    Args:
        name: one of :data:`LOSS_NAMES`.
        alpha: per-class weight tensor ``[C]``.
        gamma: focal focusing parameter.
        w_focal, w_ce, w_dice: term weights for the compound losses.
        ignore_index: label value to exclude (hard targets only).
        smooth: dice Laplace smoothing.

    Raises:
        ValueError: if ``name`` is unknown, or ``"weighted_ce"`` without ``alpha``.
    """
    name = name.lower()
    if name == "ce":
        return CrossEntropyLoss(weight=None, ignore_index=ignore_index)
    if name == "weighted_ce":
        if alpha is None:
            raise ValueError(
                "loss 'weighted_ce' requires class weights (pass alpha / "
                "TrainConfig.class_weights)")
        return CrossEntropyLoss(weight=alpha, ignore_index=ignore_index)
    if name == "focal":
        return FocalLoss(gamma=gamma, alpha=alpha, ignore_index=ignore_index)
    if name == "dice":
        return DiceLoss(ignore_index=ignore_index, smooth=smooth)
    if name == "ce_dice":
        return CEDiceLoss(
            weight=alpha, w_ce=w_ce, w_dice=w_dice,
            ignore_index=ignore_index, smooth=smooth)
    if name == "focal_dice":
        return FocalDiceLoss(
            gamma=gamma, alpha=alpha, w_focal=w_focal, w_dice=w_dice,
            ignore_index=ignore_index, smooth=smooth)
    raise ValueError(
        f"unknown loss '{name}'; choose from {LOSS_NAMES}")
