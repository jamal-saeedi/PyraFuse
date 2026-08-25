"""Streaming segmentation metrics.

:class:`ConfusionMatrix` accumulates predictions over an epoch and reads off
mIoU / mean-Dice / pixel-accuracy etc. — the same implementation as
``notebooks/segmentation_losses_metrics.ipynb``, so notebook and trainer report
identical numbers. :class:`MetricTracker` records per-epoch metric history and
remembers the best epoch under a chosen monitor metric (best-practice
checkpoint selection).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import torch


class ConfusionMatrix:
    """Streaming confusion matrix — accumulate over batches, then read metrics."""

    def __init__(self, num_classes: int, ignore_index: int = -100):
        self.n = num_classes
        self.ignore = ignore_index
        self.mat = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    def reset(self) -> None:
        self.mat.zero_()

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor) -> None:
        pred = logits.argmax(1).reshape(-1)
        tgt = target.reshape(-1)
        valid = tgt != self.ignore
        pred, tgt = pred[valid].cpu(), tgt[valid].cpu()
        k = tgt * self.n + pred
        self.mat += torch.bincount(k, minlength=self.n ** 2).reshape(self.n, self.n)

    def _stats(self):
        m = self.mat.float()
        tp = m.diag()
        fp = m.sum(0) - tp
        fn = m.sum(1) - tp
        return tp, fp, fn, m

    def iou(self) -> torch.Tensor:
        tp, fp, fn, _ = self._stats()
        return tp / (tp + fp + fn).clamp_min(1e-9)

    def dice(self) -> torch.Tensor:
        tp, fp, fn, _ = self._stats()
        return 2 * tp / (2 * tp + fp + fn).clamp_min(1e-9)

    def miou(self) -> float:
        return self.iou().mean().item()

    def mdice(self) -> float:
        return self.dice().mean().item()

    def pixel_acc(self) -> float:
        _, _, _, m = self._stats()
        return (m.diag().sum() / m.sum().clamp_min(1e-9)).item()

    def mean_acc(self) -> float:
        _, _, _, m = self._stats()
        return (m.diag() / m.sum(1).clamp_min(1e-9)).mean().item()

    def fw_iou(self) -> float:
        _, _, _, m = self._stats()
        freq = m.sum(1) / m.sum().clamp_min(1e-9)
        return (freq * self.iou()).sum().item()

    def summary(self) -> Dict[str, float]:
        """Scalar metrics, plus per-class IoU/Dice as ``iou_<i>`` / ``dice_<i>``."""
        out = {
            "miou": self.miou(),
            "mdice": self.mdice(),
            "pixel_acc": self.pixel_acc(),
            "mean_acc": self.mean_acc(),
            "fw_iou": self.fw_iou(),
        }
        for i, (iou, dice) in enumerate(zip(self.iou().tolist(), self.dice().tolist())):
            out[f"iou_{i}"] = iou
            out[f"dice_{i}"] = dice
        return out


@dataclass
class MetricTracker:
    """Track per-epoch metrics and the best epoch under a monitored metric.

    Args:
        monitor: key in the metric dict to select the best epoch on (e.g. ``miou``).
        mode: ``"max"`` (higher is better) or ``"min"``.
    """

    monitor: str = "miou"
    mode: str = "max"
    history: List[Dict[str, float]] = field(default_factory=list)
    best_value: float = field(init=False)
    best_epoch: int = field(init=False, default=-1)
    best_metrics: Dict[str, float] = field(init=False, default_factory=dict)

    def __post_init__(self):
        assert self.mode in ("max", "min")
        self.best_value = -float("inf") if self.mode == "max" else float("inf")

    def _is_better(self, value: float) -> bool:
        return value > self.best_value if self.mode == "max" else value < self.best_value

    def update(self, epoch: int, metrics: Dict[str, float]) -> bool:
        """Record this epoch's metrics. Returns True if it is the new best."""
        record = {"epoch": epoch, **metrics}
        self.history.append(record)
        value = metrics[self.monitor]
        if self._is_better(value):
            self.best_value = value
            self.best_epoch = epoch
            self.best_metrics = dict(record)
            return True
        return False
