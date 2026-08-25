"""MixUp & CutMix augmentation for *semantic segmentation*.

Unlike the classification recipes, segmentation labels are dense maps, so:

* **MixUp** blends two images by ``lam`` and returns a *soft* label map
  ``[B, C, H, W]`` = ``lam * onehot(a) + (1 - lam) * onehot(b)``.
* **CutMix** pastes a rectangular patch from a second sample into both the
  image *and* the (hard) label; the label stays integer so it is one-hotted to
  the same soft ``[B, C, H, W]`` form for a consistent loss interface.

Both operate on a single batch by mixing it with a shuffled copy of itself
(the standard in-batch trick — no extra data loading). The losses in
:mod:`deep_sunscreen.src.train.losses` accept these soft targets directly.

:class:`MixCollator` wraps the two with per-batch random selection and
probability gating, so the trainer just calls ``mix(image, label)``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _onehot(label: torch.Tensor, num_classes: int) -> torch.Tensor:
    """(B, H, W) int -> (B, C, H, W) float one-hot."""
    return F.one_hot(label.long(), num_classes).permute(0, 3, 1, 2).float()


def mixup(
    image: torch.Tensor,
    label: torch.Tensor,
    num_classes: int,
    alpha: float = 0.2,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """MixUp a batch with a shuffled copy of itself.

    Returns ``(mixed_image, soft_label)`` with ``soft_label`` shaped
    ``[B, C, H, W]``.
    """
    b = image.size(0)
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(b, generator=generator, device=image.device)

    mixed_image = lam * image + (1 - lam) * image[perm]
    oh = _onehot(label, num_classes)
    soft_label = lam * oh + (1 - lam) * oh[perm]
    return mixed_image, soft_label


def _rand_bbox(h: int, w: int, lam: float) -> Tuple[int, int, int, int]:
    """Random box whose area is ``(1 - lam)`` of the image."""
    cut_ratio = np.sqrt(1.0 - lam)
    cut_h, cut_w = int(h * cut_ratio), int(w * cut_ratio)
    cy, cx = np.random.randint(h), np.random.randint(w)
    y1, y2 = np.clip([cy - cut_h // 2, cy + cut_h // 2], 0, h)
    x1, x2 = np.clip([cx - cut_w // 2, cx + cut_w // 2], 0, w)
    return int(y1), int(y2), int(x1), int(x2)


def cutmix(
    image: torch.Tensor,
    label: torch.Tensor,
    num_classes: int,
    alpha: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CutMix a batch with a shuffled copy of itself.

    Returns ``(mixed_image, soft_label)``; the label is one-hotted so the patch
    region carries the donor's classes and the rest the original's.
    """
    b, _, h, w = image.shape
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    perm = torch.randperm(b, generator=generator, device=image.device)

    y1, y2, x1, x2 = _rand_bbox(h, w, lam)
    mixed_image = image.clone()
    mixed_image[:, :, y1:y2, x1:x2] = image[perm][:, :, y1:y2, x1:x2]

    oh = _onehot(label, num_classes)
    oh[:, :, y1:y2, x1:x2] = oh[perm][:, :, y1:y2, x1:x2]
    return mixed_image, oh


class MixCollator:
    """Apply MixUp/CutMix to a batch with probability ``prob``.

    On each call, with probability ``prob`` one of the enabled augmentations is
    chosen uniformly and applied; otherwise the batch is returned unchanged with
    its original *hard* label (so the loss path is unmodified).

    Args:
        num_classes: number of segmentation classes.
        prob: probability of applying any mix at all.
        mixup_alpha: Beta parameter for MixUp (0 disables MixUp).
        cutmix_alpha: Beta parameter for CutMix (0 disables CutMix).

    Returns ``(image, target, is_soft)`` where ``target`` is either the original
    hard ``[B, H, W]`` label or a soft ``[B, C, H, W]`` map.
    """

    def __init__(
        self,
        num_classes: int,
        prob: float = 0.5,
        mixup_alpha: float = 0.2,
        cutmix_alpha: float = 1.0,
    ):
        self.num_classes = num_classes
        self.prob = prob
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.modes = []
        if mixup_alpha > 0:
            self.modes.append("mixup")
        if cutmix_alpha > 0:
            self.modes.append("cutmix")

    def __call__(
        self, image: torch.Tensor, label: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        if not self.modes or np.random.rand() >= self.prob:
            return image, label, False

        mode = self.modes[np.random.randint(len(self.modes))]
        if mode == "mixup":
            img, soft = mixup(image, label, self.num_classes, self.mixup_alpha)
        else:
            img, soft = cutmix(image, label, self.num_classes, self.cutmix_alpha)
        return img, soft, True
