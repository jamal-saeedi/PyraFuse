#!/usr/bin/env python
"""CLI: export PyraFuse checkpoints to ExecuTorch ``.pte`` programs for mobile/edge.

Thin wrapper around :mod:`pyrafuse.deploy.mobile_export` (see its docstring for
the on-device I/O contract). For each variant it writes
``<out-dir>/<variant>/pyrafuse_<variant>_<target>.pte`` and, with ``--eval``,
scores every program the local ExecuTorch runtime can execute (XNNPACK on
Linux) on the held-out test split used by ``scripts/train_segmenter.py``
(75/15/10, split seed 42) against the eager PyTorch model. Core ML and Vulkan
programs cannot execute on a Linux host and are marked as not evaluated.
Results go to ``<out-dir>/manifest.json``.

Requires the ``mobile`` extra (``pip install "pyrafuse[mobile]"``); ExecuTorch
pins its matching torch release, so use a dedicated environment.

Usage:
    python scripts/export_mobile.py --variants small small_plus base --eval

    # CPU programs only, 320 px, quick check:
    python scripts/export_mobile.py --variants small \
        --targets xnnpack_fp32 xnnpack_int8 --image-size 320 --eval-images 50
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from importlib.metadata import version
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from pyrafuse.deploy.mobile_export import (  # noqa: E402
    CLASS_NAMES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    MOBILE_TARGETS,
    export_mobile,
    load_checkpoint,
    load_mobile_model,
)
from pyrafuse.train.metrics import ConfusionMatrix  # noqa: E402
from pyrafuse.zoo import resolve_checkpoint  # noqa: E402

RUNNABLE_BACKENDS = {"xnnpack"}  # compiled into the Linux executorch wheel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variants", nargs="+", default=["small", "small_plus", "base"],
                   help="Model-zoo variants or checkpoint directories.")
    p.add_argument("--targets", nargs="+", choices=list(MOBILE_TARGETS),
                   default=list(MOBILE_TARGETS))
    p.add_argument("--out-dir", type=Path, default=REPO / "models/mobile")
    p.add_argument("--image-size", type=int, default=448)
    p.add_argument("--eval", action="store_true",
                   help="score programs on the held-out test split")
    p.add_argument("--eval-images", type=int, default=200)
    p.add_argument("--train-images", type=Path, default=REPO / "data/raw/train")
    p.add_argument("--train-labels", type=Path,
                   default=REPO / "data/processed/train_labels")
    p.add_argument("--val-images", type=Path, default=REPO / "data/raw/test")
    p.add_argument("--val-labels", type=Path,
                   default=REPO / "data/processed/val_labels")
    p.add_argument("--split-seed", type=int, default=42)
    return p.parse_args()


def test_pairs(args) -> list:
    """Rebuild train_segmenter's held-out test (image, label) path list."""
    pairs = []
    for images, labels in ((args.train_images, args.train_labels),
                           (args.val_images, args.val_labels)):
        stems = {p.stem for p in labels.glob("*.png")}
        kept = sorted(p.stem for p in images.glob("*.jpg") if p.stem in stems)
        pairs += [(images / f"{s}.jpg", labels / f"{s}.png") for s in kept]
    order = np.random.default_rng(args.split_seed).permutation(len(pairs)).tolist()
    train_end = int(round(0.75 * len(pairs)))
    val_end = train_end + int(round(0.15 * len(pairs)))
    return [pairs[i] for i in order[val_end:]]


def load_pair(image_path: Path, label_path: Path, size: int):
    """Device-style preprocessing: stretch-resize RGB, scale to [0, 1]."""
    image = Image.open(image_path).convert("RGB").resize((size, size), Image.BILINEAR)
    x = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
    label = Image.open(label_path).resize((size, size), Image.NEAREST)
    return x.permute(2, 0, 1).unsqueeze(0), torch.from_numpy(np.asarray(label).copy())


def score(predict, samples) -> dict:
    cm, times, preds = ConfusionMatrix(len(CLASS_NAMES)), [], []
    for x, y in samples:
        start = time.perf_counter()
        logits = predict(x)
        times.append((time.perf_counter() - start) * 1000)
        cm.update(logits.float(), y.long().unsqueeze(0))
        preds.append(logits.argmax(1))
    iou = cm.iou().tolist()
    return {
        "miou": round(cm.miou(), 4),
        "iou": {c: round(v, 4) for c, v in zip(CLASS_NAMES, iou)},
        "pixel_acc": round(cm.pixel_acc(), 4),
        "median_ms_host_cpu": round(statistics.median(times[1:] or times), 1),
        "_preds": preds,
    }


def main() -> None:
    args = parse_args()
    samples = ([load_pair(*p, args.image_size)
                for p in test_pairs(args)[:args.eval_images]] if args.eval else [])
    print(f"eval images: {len(samples)} | torch threads {torch.get_num_threads()}")

    manifest_path = args.out_dir / "manifest.json"
    manifest = (json.loads(manifest_path.read_text()) if manifest_path.exists()
                else {"programs": []})
    manifest.update({
        "format": "ExecuTorch .pte",
        "input": {"name": "image", "dtype": "float32",
                  "shape": [1, 3, args.image_size, args.image_size],
                  "layout": "NCHW RGB", "range": "[0, 1] (pixel / 255)",
                  "normalization": "ImageNet mean/std applied inside the graph",
                  "mean": IMAGENET_MEAN, "std": IMAGENET_STD},
        "output": {"name": "logits", "dtype": "float32",
                   "shape": [1, len(CLASS_NAMES), args.image_size, args.image_size],
                   "classes": dict(enumerate(CLASS_NAMES)),
                   "postprocess": "argmax over dim 1"},
        "toolchain": {"executorch": version("executorch"), "torch": torch.__version__,
                      "coremltools": version("coremltools")},
        "eval_protocol": (f"first {args.eval_images} images of the held-out test "
                          f"split (seed {args.split_seed}), {args.image_size} px "
                          "stretch-resize; host CPU "
                          f"{platform.processor() or platform.machine()}"),
    })

    for variant in args.variants:
        ckpt = resolve_checkpoint(variant)
        label = Path(variant).name
        records = export_mobile(ckpt, args.out_dir / label, label, args.targets,
                                args.image_size)
        for record in records:
            record["file"] = f"{label}/{record['file']}"

        if samples:
            module = load_checkpoint(ckpt)
            with torch.no_grad():
                reference = score(module, samples)
            ref_preds = reference.pop("_preds")
            print(f"[{label}] PyTorch fp32: mIoU {reference['miou']}")
            for record in records:
                if MOBILE_TARGETS[record["target"]].backend not in RUNNABLE_BACKENDS:
                    record["eval"] = None  # needs an Apple / Android device
                    continue
                result = score(load_mobile_model(args.out_dir / record["file"]),
                               samples)
                result["evaluated_with"] = "ExecuTorch runtime, host CPU"
                preds = result.pop("_preds")
                result["pixel_agreement_vs_pytorch"] = round(float(np.mean(
                    [(a == b).float().mean().item() for a, b in zip(preds, ref_preds)]
                )), 5)
                record["eval"] = result
                print(f"[{label}] {record['target']}: mIoU {result['miou']} "
                      f"(agreement {result['pixel_agreement_vs_pytorch']}, "
                      f"{result['median_ms_host_cpu']} ms)")
            manifest.setdefault("pytorch_reference", {})[label] = reference

        kept = [r for r in manifest["programs"]
                if not (r["variant"] == label and r["image_size"] == args.image_size
                        and r["target"] in args.targets)]
        manifest["programs"] = kept + records
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
