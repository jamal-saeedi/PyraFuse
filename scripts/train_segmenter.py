#!/usr/bin/env python
"""Train the configurable RGB skin/fabric segmentation model.

This is the canonical training entry point. It uses the pluggable hybrid model,
so encoder, decoder, loss, optimisation, augmentation, split, and checkpoint
settings are all visible from the command line.

The dataset has three fixed classes: ``background=0``, ``fabric=1``, and
``skin=2``. By default the runner combines ``data/raw/train`` and
``data/raw/test`` and makes the same reproducible 75/15/15 split used by the
benchmark scripts (split seed 42).

Examples
--------
Train a normal run::

    python scripts/train_segmenter.py \\
        --encoder dinov3 --decoder pyrafuse \\
        --loss focal_dice --device cuda:0 --epochs 25

Run a small CPU smoke test::

    python scripts/train_segmenter.py \\
        --encoder dinov3 --decoder allmlp \\
        --device cpu --size 64 --batch-size 2 --epochs 1 --smoke-batches 1 \\
        --num-workers 0 --checkpoint-path /tmp/deep-sunscreen-smoke

Warm-start from a saved model checkpoint (model architecture is read from its
``config.json``; this restores model weights, not optimizer state)::

    python scripts/train_segmenter.py \\
        --resume models/checkpoints/.../best \\
        --checkpoint-path models/checkpoints/finetune
"""
from __future__ import annotations

# Third-party imports intentionally follow the CUDA allocator environment
# setup below; keep isort from moving torch above that assignment.
# isort: skip_file

import argparse
import json
import os
import pathlib
import random
import sys
from collections import Counter
from dataclasses import asdict
from typing import Iterable

# Resolve the repository before importing third-party libraries so their
# process-wide settings are deterministic for both local and CI runs.
REPO = pathlib.Path(__file__).resolve().parent.parent

# Must be set before importing torch / creating a CUDA context.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# Avoid an optional version check/network request on every CLI invocation.
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
# Keep Matplotlib's cache inside the ignored artifacts directory.
os.environ.setdefault("MPLCONFIGDIR", str(REPO / "artifacts" / "mplconfig"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import (  # noqa: E402
    ConcatDataset,
    DataLoader,
    Subset,
)


sys.path.insert(0, str(REPO))

from pyrafuse.data import NUM_CLASSES, SkinFabricDataset  # noqa: E402
from pyrafuse.model.model_hybrid import (  # noqa: E402
    DEFAULT_MODEL_IDS,
    DINOV3_SIZES,
    HybridSegmenter,
    ModelConfig,
    build_segmenter,
    encoder_variant_tag,
)
from pyrafuse.train import LOSS_NAMES, TrainConfig, Trainer  # noqa: E402

CLASS_NAMES = ["background", "fabric", "skin"]
ENCODER_NAMES = tuple(DEFAULT_MODEL_IDS)
DECODER_NAMES = ("tpa_sad", "pyrafuse", "dpt",
                 "allmlp", "upernet", "mask2former")


def _bool_arg(parser: argparse.ArgumentParser, *flags: str, **kwargs):
    """Add a Python 3.12 boolean optional argument with a useful default."""
    kwargs.setdefault("action", argparse.BooleanOptionalAction)
    return parser.add_argument(*flags, **kwargs)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    data = p.add_argument_group("data and split")
    data.add_argument("--train-images", type=pathlib.Path,
                      default=REPO / "data/raw/train")
    data.add_argument("--train-labels", type=pathlib.Path,
                      default=REPO / "data/processed/train_labels")
    data.add_argument("--val-images", type=pathlib.Path,
                      default=REPO / "data/raw/test")
    data.add_argument("--val-labels", type=pathlib.Path,
                      default=REPO / "data/processed/val_labels")
    data.add_argument("--split-seed", type=int, default=42,
                      help="seed for the reproducible 75/15/15 split")
    data.add_argument("--train-fraction", type=float, default=0.75)
    data.add_argument("--val-fraction", type=float, default=0.15)
    data.add_argument("--max-train-samples", type=int, default=-1,
                      help="cap train samples after splitting (-1 = all)")
    data.add_argument("--max-val-samples", type=int, default=-1,
                      help="cap validation samples after splitting (-1 = all)")
    data.add_argument("--max-test-samples", type=int, default=-1,
                      help="cap test samples after splitting (-1 = all)")

    model = p.add_argument_group("model")
    model.add_argument("--encoder-type", "--encoder", dest="encoder_type",
                       choices=ENCODER_NAMES, default="dinov3",
                       help="encoder family")
    model.add_argument("--encoder-model-id", "--model-id",
                       dest="encoder_model_id", default=None,
                       help="HF model id, timm name, or RADIO version string")
    model.add_argument("--encoder-size", choices=tuple(DINOV3_SIZES), default=None,
                       help="DINOv3 shorthand: s, s_plus, b, or l")
    model.add_argument("--decoder-type", "--decoder", dest="decoder_type",
                       choices=DECODER_NAMES, default="tpa_sad",
                       help="decoder head")
    model.add_argument("--size", type=int, default=448,
                       help="square input size; must be a multiple of the patch size")
    model.add_argument("--decoder-channels", type=int, default=128)
    model.add_argument("--use-bn", action="store_true",
                       help="use BatchNorm rather than GroupNorm where supported")
    model.add_argument("--drop-path-rate", type=float, default=0.05)
    model.add_argument("--dropout", type=float, default=0.1)
    model.add_argument("--num-queries", type=int, default=10,
                       help="Mask2Former query count")
    model.add_argument("--num-transformer-layers", type=int, default=3,
                       help="Mask2Former transformer depth")
    model.add_argument("--freeze-backbone", action="store_true", default=True)
    model.add_argument("--finetune-backbone", dest="freeze_backbone",
                       action="store_false",
                       help="unfreeze the encoder and train it end-to-end")
    model.add_argument("--ema-decay", type=float, default=0.9995,
                       help="EMA decay; 0 disables EMA")
    model.add_argument("--text-prompts", nargs="+", default=None, metavar="PROMPT",
                       help="class prompts for CLIP/SigLIP/EVA-CLIP text gating")
    model.add_argument("--no-text-prompts", action="store_true",
                       help="disable text gating even for a VLM encoder")

    loss = p.add_argument_group("loss and augmentation")
    loss.add_argument("--loss", choices=LOSS_NAMES, default="focal_dice")
    loss.add_argument("--class-weights", nargs=NUM_CLASSES, type=float, default=None,
                      metavar=("BACKGROUND", "FABRIC", "SKIN"),
                      help="explicit CE/focal weights; otherwise estimate them")
    loss.add_argument("--no-class-weights", action="store_true",
                      help="do not estimate inverse-frequency class weights")
    loss.add_argument("--alpha-sample", type=int, default=750,
                      help="images used for automatic class weights; 0 = all")
    loss.add_argument("--focal-gamma", type=float, default=2.0)
    loss.add_argument("--w-focal", type=float, default=1.0)
    loss.add_argument("--w-ce", type=float, default=1.0)
    loss.add_argument("--w-dice", type=float, default=1.0)
    loss.add_argument("--dice-smooth", type=float, default=1.0)
    loss.add_argument("--mix-prob", type=float, default=0.5,
                      help="probability of MixUp/CutMix per batch; 0 disables")
    loss.add_argument("--mixup-alpha", type=float, default=0.2)
    loss.add_argument("--cutmix-alpha", type=float, default=1.0)

    train = p.add_argument_group("optimisation and runtime")
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--num-workers", type=int, default=4)
    train.add_argument("--prefetch-factor", type=int, default=2)
    train.add_argument("--epochs", type=int, default=150)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--backbone-lr", type=float, default=1e-5)
    train.add_argument("--weight-decay", type=float, default=1e-3)
    train.add_argument("--warmup-epochs", type=int, default=2)
    train.add_argument("--grad-clip", type=float, default=1.0,
                       help="max gradient norm; negative disables clipping")
    _bool_arg(train, "--amp", default=True,
              help="use CUDA bfloat16 autocast when available")
    _bool_arg(train, "--pin-memory", default=None,
              help="pin data-loader memory; default follows the device")
    _bool_arg(train, "--persistent-workers", default=False)
    _bool_arg(train, "--drop-last", default=False,
              help="drop an incomplete training batch")
    train.add_argument("--log-every", type=int, default=20)
    train.add_argument("--ignore-index", type=int, default=-100)

    checkpoint = p.add_argument_group("checkpointing")
    checkpoint.add_argument("--ckpt-dir", type=pathlib.Path,
                            default=REPO / "models/checkpoints/skin_fabric_hybrid",
                            help="root for automatic <encoder>_<decoder>_<loss> runs")
    checkpoint.add_argument("--checkpoint-path", "--run-dir", "--output-dir",
                            dest="run_dir", type=pathlib.Path, default=None,
                            help="exact run directory; contains best/ and last/")
    checkpoint.add_argument("--resume", "--init-checkpoint", dest="init_checkpoint",
                            type=pathlib.Path, default=None,
                            help="warm-start from a saved model directory")
    _bool_arg(checkpoint, "--save-best", default=True)
    _bool_arg(checkpoint, "--save-last", default=True)
    checkpoint.add_argument("--monitor", default="miou",
                            help="metric key used for best-checkpoint selection")
    checkpoint.add_argument(
        "--monitor-mode", choices=("min", "max"), default="max")
    _bool_arg(checkpoint, "--eval-with-ema", default=True)

    misc = p.add_argument_group("miscellaneous")
    misc.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    misc.add_argument("--seed", type=int, default=42,
                      help="seed for data-loader order and augmentation workers")
    misc.add_argument("--smoke-batches", type=int, default=0,
                      help="cap train and validation to this many batches")
    misc.add_argument("--smoke", action="store_true",
                      help="one-epoch, one-batch convenience smoke configuration")
    misc.add_argument("--skip-test", action="store_true",
                      help="skip held-out test evaluation after training")
    misc.add_argument("--dry-run", action="store_true",
                      help=("validate data/configuration and build the model, "
                            "but do not fit"))

    args = p.parse_args(argv)
    if args.smoke:
        args.epochs = 1
        args.smoke_batches = max(args.smoke_batches, 1)
        args.num_workers = 0
    return args


def _validate_args(args: argparse.Namespace) -> None:
    if args.encoder_size and args.encoder_type != "dinov3":
        raise ValueError("--encoder-size is only valid with --encoder dinov3")
    if args.encoder_model_id and args.encoder_size:
        print("warning: --encoder-model-id takes precedence over --encoder-size")
    if args.no_text_prompts and args.text_prompts is not None:
        raise ValueError(
            "use either --text-prompts or --no-text-prompts, not both")
    if args.no_class_weights and args.class_weights is not None:
        raise ValueError(
            "use either --class-weights or --no-class-weights, not both")
    if (args.loss == "weighted_ce" and args.no_class_weights
            and args.class_weights is None):
        raise ValueError("loss weighted_ce requires class weights")
    if args.train_fraction <= 0 or args.val_fraction <= 0:
        raise ValueError(
            "--train-fraction and --val-fraction must be positive")
    if args.train_fraction + args.val_fraction >= 1:
        raise ValueError(
            "train and validation fractions must leave a non-empty test split")
    if args.size <= 0 or args.size % 16:
        raise ValueError(
            f"--size must be a positive multiple of 16, got {args.size}")
    if args.decoder_channels <= 0:
        raise ValueError("--decoder-channels must be positive")
    if not 0 <= args.drop_path_rate < 1:
        raise ValueError("--drop-path-rate must be in [0, 1)")
    if not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be in [0, 1)")
    for name in ("batch_size", "epochs"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("lr", "backbone_lr", "weight_decay"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.warmup_epochs < 0:
        raise ValueError("--warmup-epochs must be non-negative")
    for name in ("num_workers", "prefetch_factor"):
        if (getattr(args, name) < 0
                or (name == "prefetch_factor" and getattr(args, name) == 0)):
            raise ValueError(
                f"--{name.replace('_', '-')} must be positive or zero as applicable")
    for name in ("max_train_samples", "max_val_samples", "max_test_samples"):
        if getattr(args, name) < -1:
            raise ValueError(f"--{name.replace('_', '-')} must be >= -1")
    if args.smoke_batches < 0:
        raise ValueError("--smoke-batches must be non-negative")
    if not 0 <= args.mix_prob <= 1:
        raise ValueError("--mix-prob must be between 0 and 1")
    for name in ("mixup_alpha", "cutmix_alpha", "focal_gamma", "dice_smooth"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    for name in ("w_focal", "w_ce", "w_dice"):
        if getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.alpha_sample < 0:
        raise ValueError("--alpha-sample must be non-negative (0 = all)")
    if args.ema_decay < 0 or args.ema_decay >= 1:
        raise ValueError("--ema-decay must be in [0, 1)")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    valid_monitors = {
        "miou", "mdice", "pixel_acc", "mean_acc", "fw_iou", "loss",
        *(f"iou_{i}" for i in range(NUM_CLASSES)),
        *(f"dice_{i}" for i in range(NUM_CLASSES)),
    }
    if args.monitor not in valid_monitors:
        raise ValueError(
            "--monitor must be one of: miou, mdice, pixel_acc, mean_acc, "
            "fw_iou, loss, iou_0..2, dice_0..2"
        )
    if args.init_checkpoint is not None:
        required = (args.init_checkpoint / "config.json",
                    args.init_checkpoint / "decoder.pt")
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "--resume does not look like a saved model checkpoint; "
                f"missing {missing}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"requested {args.device}, but CUDA is not available")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_concat_index(
    concat_ds: ConcatDataset, global_idx: int
) -> tuple[SkinFabricDataset, int]:
    import bisect

    component_idx = bisect.bisect_right(concat_ds.cumulative_sizes, global_idx)
    previous = (0 if component_idx == 0
                else concat_ds.cumulative_sizes[component_idx - 1])
    return concat_ds.datasets[component_idx], global_idx - previous


def estimate_alpha(train_subset: Subset, n_sample: int) -> np.ndarray:
    """Estimate inverse-frequency class weights from raw, unaugmented masks."""
    counts: Counter[int] = Counter()
    indices = list(train_subset.indices)
    n = len(indices) if n_sample <= 0 else min(len(indices), n_sample)
    concat_ds = train_subset.dataset
    if not isinstance(concat_ds, ConcatDataset):
        raise TypeError(
            "alpha estimation expects a Subset over a ConcatDataset")
    for global_idx in indices[:n]:
        component, local_idx = _resolve_concat_index(concat_ds, global_idx)
        label_path = component.labels_dir / f"{component.stems[local_idx]}.png"
        with Image.open(label_path) as image:
            label = np.asarray(image, dtype=np.int64)
        binc = np.bincount(label.reshape(-1), minlength=NUM_CLASSES)
        for class_id in range(NUM_CLASSES):
            counts[class_id] += int(binc[class_id])
    freq = np.array([counts[c] for c in range(NUM_CLASSES)], dtype=np.float64)
    if freq.sum() <= 0:
        raise ValueError(
            "could not find any labeled pixels for class-weight estimation")
    freq /= freq.sum()
    alpha = 1.0 / np.clip(freq, 1e-6, None)
    alpha = alpha / alpha.sum() * NUM_CLASSES
    print(f"class freq : {np.round(freq, 4)} (over {n} images)")
    print(f"class weight: {np.round(alpha, 3)}")
    return alpha


def _cap_indices(indices: list[int], limit: int) -> list[int]:
    return indices if limit < 0 else indices[:limit]


def _worker_init(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _loader(
    dataset: Subset,
    args: argparse.Namespace,
    device: torch.device,
    shuffle: bool,
    drop_last: bool = False,
) -> DataLoader:
    pin_memory = device.type == "cuda" if args.pin_memory is None else args.pin_memory
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    kwargs = dict(
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        generator=generator,
        worker_init_fn=_worker_init,
    )
    if args.num_workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
        kwargs["persistent_workers"] = args.persistent_workers
    return DataLoader(dataset, **kwargs)


def _model_config(
    args: argparse.Namespace, encoder_model_id: str, prompts: list[str] | None
) -> ModelConfig:
    return ModelConfig(
        encoder_type=args.encoder_type,
        encoder_model_id=encoder_model_id,
        decoder_type=args.decoder_type,
        text_prompts=prompts,
        num_classes=NUM_CLASSES,
        decoder_channels=args.decoder_channels,
        image_size=args.size,
        use_bn=args.use_bn,
        drop_path_rate=args.drop_path_rate,
        dropout=args.dropout,
        num_queries=args.num_queries,
        num_transformer_layers=args.num_transformer_layers,
        freeze_backbone=args.freeze_backbone,
        ema_decay=args.ema_decay,
    )


def _jsonable(value):
    if isinstance(value, pathlib.Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _print_class_metrics(metrics: dict[str, float], prefix: str) -> None:
    rounded = {key: round(value, 4) for key, value in metrics.items()}
    print(f"{prefix}: {rounded}")
    for class_id, name in enumerate(CLASS_NAMES):
        iou = metrics.get(f"iou_{class_id}")
        dice = metrics.get(f"dice_{class_id}")
        if iou is not None and dice is not None:
            print(f"{name:>11}: IoU={iou:.3f} Dice={dice:.3f}")


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    _validate_args(args)
    set_seed(args.seed)
    device = torch.device(args.device)

    if args.encoder_model_id:
        encoder_model_id = args.encoder_model_id
    elif args.encoder_size:
        encoder_model_id = DINOV3_SIZES[args.encoder_size]
    else:
        encoder_model_id = DEFAULT_MODEL_IDS[args.encoder_type]
    prompts = None if args.no_text_prompts else args.text_prompts

    print(
        f"device: {device} | classes: {NUM_CLASSES} | encoder: "
        f"{args.encoder_type} ({encoder_model_id}) | decoder: {args.decoder_type} | "
        f"loss: {args.loss} | image_size: {args.size}"
    )

    for path in (
        args.train_images, args.train_labels, args.val_images, args.val_labels
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    all_aug = ConcatDataset([
        SkinFabricDataset(args.train_images, args.train_labels,
                          size=args.size, train=True),
        SkinFabricDataset(args.val_images, args.val_labels,
                          size=args.size, train=True),
    ])
    all_eval = ConcatDataset([
        SkinFabricDataset(args.train_images, args.train_labels,
                          size=args.size, train=False),
        SkinFabricDataset(args.val_images, args.val_labels,
                          size=args.size, train=False),
    ])
    total = len(all_aug)
    split_rng = np.random.default_rng(args.split_seed)
    shuffled = split_rng.permutation(total).tolist()
    train_end = int(round(args.train_fraction * total))
    val_end = train_end + int(round(args.val_fraction * total))
    train_indices = _cap_indices(shuffled[:train_end], args.max_train_samples)
    val_indices = _cap_indices(
        shuffled[train_end:val_end], args.max_val_samples)
    test_indices = _cap_indices(shuffled[val_end:], args.max_test_samples)
    if args.smoke_batches > 0:
        smoke_limit = args.smoke_batches * args.batch_size
        train_indices = train_indices[:smoke_limit]
        val_indices = val_indices[:smoke_limit]
        test_indices = test_indices[:smoke_limit]

    train_ds = Subset(all_aug, train_indices)
    val_ds = Subset(all_eval, val_indices)
    test_ds = Subset(all_eval, test_indices)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError(
            f"empty split after limits: train={len(train_ds)}, val={len(val_ds)}")
    print(
        f"total: {total} | train: {len(train_ds)} | val: {len(val_ds)} | "
        f"test: {len(test_ds)}"
    )

    train_drop_last = args.drop_last and len(train_ds) >= args.batch_size
    train_loader = _loader(train_ds, args, device,
                           shuffle=True, drop_last=train_drop_last)
    val_loader = _loader(val_ds, args, device, shuffle=False)
    test_loader = _loader(test_ds, args, device,
                          shuffle=False) if len(test_ds) else None

    if args.no_class_weights:
        class_weights = None
    elif args.class_weights is not None:
        class_weights = list(args.class_weights)
    else:
        class_weights = estimate_alpha(train_ds, args.alpha_sample).tolist()

    if args.init_checkpoint is not None:
        print(f"warm-starting model from: {args.init_checkpoint}")
        model = HybridSegmenter.from_pretrained(
            args.init_checkpoint,
            map_location="cpu",
            load_ema_into_model=True,
        )
        print("resume note: architecture is taken from checkpoint config.json")
    else:
        model = build_segmenter(_model_config(args, encoder_model_id, prompts))

    variant = encoder_variant_tag(
        model.config.encoder_type, model.config.encoder_model_id)
    default_run_dir = args.ckpt_dir / \
        f"{variant}_{model.config.decoder_type}_{args.loss}"
    run_dir = args.run_dir or default_run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    grad_clip = None if args.grad_clip < 0 else args.grad_clip
    train_cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay,
        warmup_epochs=args.warmup_epochs,
        grad_clip=grad_clip,
        amp=args.amp,
        loss_name=args.loss,
        focal_gamma=args.focal_gamma,
        w_focal=args.w_focal,
        w_ce=args.w_ce,
        w_dice=args.w_dice,
        dice_smooth=args.dice_smooth,
        class_weights=class_weights,
        mix_prob=args.mix_prob,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        monitor=args.monitor,
        monitor_mode=args.monitor_mode,
        eval_with_ema=args.eval_with_ema,
        ckpt_dir=str(run_dir),
        save_best=args.save_best,
        save_last=args.save_last,
        ignore_index=args.ignore_index,
        log_every=args.log_every,
    )
    (run_dir / "run_config.json").write_text(json.dumps(_jsonable({
        "args": vars(args),
        "model": model.config.to_dict(),
        "train": asdict(train_cfg),
        "split_sizes": {
            "total": total,
            "train": len(train_ds),
            "val": len(val_ds),
            "test": len(test_ds),
        },
    }), indent=2) + "\n")

    trainer = Trainer(model, num_classes=NUM_CLASSES,
                      config=train_cfg, device=device)
    trainable = sum(parameter.numel()
                    for parameter in model.parameters() if parameter.requires_grad)
    print(
        f"trainable params: {trainable / 1e6:.2f}M | checkpoint run: {run_dir}")
    if args.dry_run:
        print(
            "dry-run complete: data loaders, model, loss, and checkpoint "
            "config are valid")
        return

    tracker = trainer.fit(train_loader, val_loader)
    _print_class_metrics(tracker.best_metrics, "best validation metrics")

    results = {"best_validation": tracker.best_metrics,
               "best_epoch": tracker.best_epoch}
    if not args.skip_test and test_loader is not None:
        test_metrics = trainer.evaluate(test_loader)
        results["test"] = test_metrics
        _print_class_metrics(test_metrics, "test metrics")
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(_jsonable(results), indent=2) + "\n")


if __name__ == "__main__":
    main()
