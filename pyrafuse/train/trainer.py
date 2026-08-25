"""Training loop for :class:`~deep_sunscreen.src.model.DINOv3Segmenter`.

:class:`Trainer` wraps the standard fit / evaluate loop with the project's
best-practice defaults:

* **Focal + Dice** loss (:class:`~deep_sunscreen.src.train.losses.FocalDiceLoss`),
  the robust choice for the background-dominated skin/fabric task.
* **MixUp / CutMix** (:class:`~deep_sunscreen.src.train.mixup.MixCollator`) with
  soft-label loss, gated by probability.
* **EMA** of the weights (driven by the model's own :class:`ModelEMA`) and
  evaluation under the EMA shadow.
* **AMP** mixed-precision with gradient scaling.
* **Best-checkpoint tracking** on a monitored metric
  (:class:`~deep_sunscreen.src.train.metrics.MetricTracker`), saving the best
  model via ``DINOv3Segmenter.save_pretrained``.

The decoder is trained while the DINOv3 backbone stays frozen by default (set
``ModelConfig.freeze_backbone = False`` to fine-tune end-to-end).
"""

from __future__ import annotations

import copy
import gc
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..model import DINOv3Segmenter
from .losses import build_loss
from .metrics import ConfusionMatrix, MetricTracker
from .mixup import MixCollator

# Variable batch/input shapes (e.g. the smaller last eval batch) make cuDNN's
# benchmark autotuner re-tune and reserve fresh workspace per shape, which is
# pure waste here -- disable it. (The matching allocator setting,
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, must be set before the CUDA
# context is created, so it lives at the top of the entry-point script, not
# here -- by import time it would be too late to take effect.)
torch.backends.cudnn.benchmark = False


@dataclass
class TrainConfig:
    """Hyper-parameters for :class:`Trainer`."""

    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-3
    backbone_lr: float = 1e-5          # used only when backbone is unfrozen
    warmup_epochs: int = 1
    grad_clip: Optional[float] = 1.0
    amp: bool = True

    # loss
    # one of losses.LOSS_NAMES: "ce", "weighted_ce", "focal", "dice",
    # "ce_dice", "focal_dice" (the adopted default).
    loss_name: str = "focal_dice"
    focal_gamma: float = 2.0
    w_focal: float = 1.0
    w_ce: float = 1.0
    w_dice: float = 1.0
    dice_smooth: float = 1.0
    class_weights: Optional[List[float]] = None  # CE/focal alpha [C], inv-freq

    # mix augmentation
    mix_prob: float = 0.5
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0

    # metric / checkpointing
    monitor: str = "miou"
    monitor_mode: str = "max"
    eval_with_ema: bool = True
    ckpt_dir: str = "models/checkpoints"
    save_best: bool = True
    save_last: bool = False

    ignore_index: int = -100
    log_every: int = 20


class Trainer:
    """Fit / evaluate a :class:`DINOv3Segmenter` on segmentation data.

    Args:
        model: the segmenter (with backbone + decoder; EMA optional).
        num_classes: number of classes (must match the model head).
        config: :class:`TrainConfig` of hyper-parameters.
        device: torch device; defaults to CUDA when available.
    """

    def __init__(
        self,
        model: DINOv3Segmenter,
        num_classes: int,
        config: Optional[TrainConfig] = None,
        device: Optional[torch.device] = None,
    ):
        self.cfg = config or TrainConfig()
        self.num_classes = num_classes
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.to(self.device)
        # Re-init EMA after the move so its shadow copy lives on `device` too
        # (build_segmenter creates the EMA on CPU before this point).
        if self.cfg.eval_with_ema:
            self.model.init_ema()
        elif self.model.ema is not None:
            self.model.init_ema(decay=0.0)  # disable a stale CPU EMA

        alpha = None
        if self.cfg.class_weights is not None:
            if len(self.cfg.class_weights) != num_classes:
                raise ValueError(
                    f"class_weights has {len(self.cfg.class_weights)} entries "
                    f"but num_classes={num_classes}"
                )
            alpha = torch.tensor(self.cfg.class_weights, dtype=torch.float32)
        self.criterion = build_loss(
            self.cfg.loss_name,
            alpha=alpha,
            gamma=self.cfg.focal_gamma,
            w_focal=self.cfg.w_focal,
            w_ce=self.cfg.w_ce,
            w_dice=self.cfg.w_dice,
            ignore_index=self.cfg.ignore_index,
            smooth=self.cfg.dice_smooth,
        ).to(self.device)

        self.mixer = MixCollator(
            num_classes=num_classes,
            prob=self.cfg.mix_prob,
            mixup_alpha=self.cfg.mixup_alpha,
            cutmix_alpha=self.cfg.cutmix_alpha,
        )

        self.optimizer = self._build_optimizer()
        # bf16 autocast on CUDA. bf16 has fp32's exponent range, so unlike fp16
        # it needs no GradScaler (no gradient underflow / overflow to manage).
        self.amp_enabled = self.cfg.amp and self.device.type == "cuda"
        self.tracker = MetricTracker(
            monitor=self.cfg.monitor, mode=self.cfg.monitor_mode)
        self.scheduler = None  # set in fit() once steps_per_epoch is known
        self.train_loss_history: List[float] = []

    # -- setup ---------------------------------------------------------------
    def _build_optimizer(self) -> torch.optim.Optimizer:
        decoder_params = list(self.model.decoder.parameters())
        groups = [{"params": decoder_params, "lr": self.cfg.lr}]
        # Only add backbone params if any are trainable (fine-tuning).
        bb_params = [p for p in self.model.backbone.parameters()
                     if p.requires_grad]
        if bb_params:
            groups.append({"params": bb_params, "lr": self.cfg.backbone_lr})
        return torch.optim.AdamW(groups, weight_decay=self.cfg.weight_decay)

    def _build_scheduler(self, steps_per_epoch: int):
        total = steps_per_epoch * self.cfg.epochs
        warmup = steps_per_epoch * self.cfg.warmup_epochs

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(1, warmup)
            # cosine decay to 0 over the remaining steps
            progress = (step - warmup) / max(1, total - warmup)
            import math
            return 0.5 * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    # -- one training epoch --------------------------------------------------
    def train_one_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        if self.model.config.freeze_backbone:
            self.model.backbone.eval()  # keep frozen BN/stats in eval

        running, n = 0.0, 0
        pbar = tqdm(loader, desc=f"train {epoch}", leave=False)
        for step, batch in enumerate(pbar):
            image = batch["image"].to(self.device, non_blocking=True)
            label = batch["label"].to(self.device, non_blocking=True)

            image, target, _ = self.mixer(image, label)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=self.amp_enabled
            ):
                logits = self.model(pixel_values=image)
                loss = self.criterion(logits, target)

            # Skip non-finite steps so a single bad batch can't poison the live
            # weights (and, via update_ema, the EMA shadow used for eval).
            if not torch.isfinite(loss):
                # Skip the whole step, including the LR schedule, so a bad batch
                # neither updates weights nor burns a scheduler step before any
                # optimizer.step() has run (which triggers a PyTorch warning).
                continue

            # bf16 needs no GradScaler: plain backward / step in fp32-range grads.
            loss.backward()
            if self.cfg.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.model.update_ema()

            bs = image.size(0)
            running += loss.item() * bs
            n += bs
            if step % self.cfg.log_every == 0:
                # Report both: `alloc` is live tensors (a real leak shows here),
                # `resv` is the allocator's cache high-water (ramps then plateaus
                # under variable shapes -- benign).
                alloc_mb = torch.cuda.memory_allocated(self.device) / 1024**2
                resv_mb = torch.cuda.memory_reserved(self.device) / 1024**2
                pbar.set_postfix(loss=f"{running / max(1, n):.4f}",
                                 lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
                                 mem=f"{alloc_mb:.0f}/{resv_mb:.0f}MB")
        return running / max(1, n)

    # -- evaluation ----------------------------------------------------------
    @torch.no_grad()
    def evaluate(
        self, loader: DataLoader, use_ema: Optional[bool] = None
    ) -> Dict[str, float]:
        """Evaluate on ``loader`` and return the metric summary dict.

        When ``use_ema`` (default: ``cfg.eval_with_ema``) and an EMA shadow
        exists, evaluation runs under a temporary copy with the EMA weights so
        the live model is left untouched.
        """
        use_ema = self.cfg.eval_with_ema if use_ema is None else use_ema

        eval_model = self.model
        ema_copy = None
        if use_ema and self.model.ema is not None:
            ema_copy = copy.deepcopy(self.model)
            self.model.ema.copy_to(ema_copy)
            eval_model = ema_copy
        eval_model.eval()

        cm = ConfusionMatrix(
            self.num_classes, ignore_index=self.cfg.ignore_index)
        loss_sum, n = 0.0, 0
        for batch in tqdm(loader, desc="eval", leave=False):
            image = batch["image"].to(self.device, non_blocking=True)
            label = batch["label"].to(self.device, non_blocking=True)
            with torch.amp.autocast(
                "cuda", dtype=torch.bfloat16, enabled=self.amp_enabled
            ):
                logits = eval_model(pixel_values=image)
                loss = self.criterion(logits, label)
            cm.update(logits.float(), label)
            loss_sum += loss.item() * image.size(0)
            n += image.size(0)

        if ema_copy is not None:
            # The deepcopy goes through BackboneAdapter.__deepcopy__, which builds
            # reference cycles, so `del` alone doesn't drop the refcount to zero --
            # the copy lingers until Python's generational GC happens to fire. For
            # a large backbone (RADIO ~750MB) that means one full model's worth of
            # CUDA memory leaks per eval and only frees sporadically (the "grows
            # for several epochs then drops" pattern). Force a collection so the
            # cycle is reclaimed now, before empty_cache returns it to the driver.
            del eval_model, ema_copy
            gc.collect()
            torch.cuda.empty_cache()

        metrics = cm.summary()
        metrics["loss"] = loss_sum / max(1, n)
        return metrics

    # -- full fit ------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> MetricTracker:
        """Train for ``cfg.epochs``, evaluating each epoch and tracking the best.

        Returns the :class:`MetricTracker` with full history and the best epoch.
        """
        self.scheduler = self._build_scheduler(len(train_loader))
        ckpt_dir = Path(self.cfg.ckpt_dir)

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            train_loss = self.train_one_epoch(train_loader, epoch)
            self.train_loss_history.append(train_loss)

            metrics = self.evaluate(val_loader)

            is_best = self.tracker.update(epoch, metrics)
            dt = time.time() - t0

            print(
                f"epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                f"val_loss {metrics['loss']:.4f} | mIoU {metrics['miou']:.4f} | "
                f"mDice {metrics['mdice']:.4f} | pixacc {metrics['pixel_acc']:.4f} | "
                f"{dt:.0f}s" + ("  <-- best" if is_best else "")
            )

            if is_best and self.cfg.save_best:
                self.model.save_pretrained(ckpt_dir / "best")
            if self.cfg.save_last:
                self.model.save_pretrained(ckpt_dir / "last")

        print(
            f"\nbest epoch {self.tracker.best_epoch} | "
            f"{self.cfg.monitor}={self.tracker.best_value:.4f}"
        )
        return self.tracker

    # -- inference helper for viz -------------------------------------------
    @torch.no_grad()
    def predict(
        self, image: torch.Tensor, use_ema: Optional[bool] = None
    ) -> torch.Tensor:
        """Predict class maps for a ``[B, 3, H, W]`` (normalised) image batch.

        Returns ``[B, H, W]`` integer predictions. Mirrors :meth:`evaluate`'s
        EMA handling.
        """
        use_ema = self.cfg.eval_with_ema if use_ema is None else use_ema
        eval_model = self.model
        ema_copy = None
        if use_ema and self.model.ema is not None:
            ema_copy = copy.deepcopy(self.model)
            self.model.ema.copy_to(ema_copy)
            eval_model = ema_copy
        eval_model.eval()

        image = image.to(self.device)
        with torch.amp.autocast(
            "cuda", dtype=torch.bfloat16, enabled=self.amp_enabled
        ):
            logits = eval_model(pixel_values=image)
        result = logits.argmax(1).cpu()
        if ema_copy is not None:
            # See evaluate(): the deepcopy forms reference cycles, so force a GC
            # pass before empty_cache or a full backbone copy leaks per call.
            del eval_model, ema_copy
            gc.collect()
            torch.cuda.empty_cache()
        return result
