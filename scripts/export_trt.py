#!/usr/bin/env python
"""CLI: export a HybridSegmenter checkpoint to ONNX + build TensorRT engines.

Thin wrapper around pyrafuse.deploy.trt_export.export_and_build --
see that module's docstring for the three correctness fixes (ONNX external-
data sidecar parsing, FP16 ELEMENTWISE pinning, TRTRunner stream sync) baked
into every engine this produces.

Usage:
    python scripts/export_trt.py \
        --ckpt models/finals/large \
        --out-dir models/trt_pipeline/large \
        --precision mixed int8 \
        --verify

    # all three precisions (default), default batch profile 1/16/16:
    python scripts/export_trt.py \
        --ckpt models/finals/base \
        --out-dir models/trt_pipeline/base --verify
"""

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch  # noqa: E402

from pyrafuse.deploy.trt_export import (  # noqa: E402
    PRECISIONS,
    export_and_build,
    verify_engines,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Checkpoint dir (contains config.json, decoder.pt, backbone/).",
    )
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument(
        "--label",
        default=None,
        help="Basename for output files; defaults to the checkpoint's parent dir name.",
    )
    p.add_argument(
        "--precision",
        nargs="+",
        choices=list(PRECISIONS),
        default=list(PRECISIONS),
        help="One or more of: fp32 mixed int8.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--min-batch", type=int, default=1)
    p.add_argument("--opt-batch", type=int, default=16)
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument(
        "--micro-batch",
        type=int,
        default=4,
        help="Batch size used to trace the model for ONNX export.",
    )
    p.add_argument("--workspace-mib", type=int, default=4096)
    p.add_argument(
        "--calib-batches",
        type=int,
        default=10,
        help="Number of INT8 calibration batches (random noise; "
        "see export_and_build(calib_images=...) to use real data instead).",
    )
    p.add_argument(
        "--verify",
        action="store_true",
        help="Cosine-similarity sanity check of each built engine vs. eager PyTorch.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not (args.ckpt / "config.json").exists():
        raise SystemExit(
            f"{args.ckpt} does not look like a HybridSegmenter checkpoint "
            "(no config.json)."
        )
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required to build/verify TensorRT engines.")

    result = export_and_build(
        args.ckpt,
        args.out_dir,
        precisions=args.precision,
        label=args.label,
        device=args.device,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        micro_batch=args.micro_batch,
        workspace_mib=args.workspace_mib,
        n_calib_batches=args.calib_batches,
    )
    print(f"\nwrote {len(result.manifest)} engine pair(s) to {args.out_dir}")
    print(f"manifest: {args.out_dir / 'manifest.json'}")

    if args.verify:
        # cos threshold is precision-aware: int8 inherently has lower logit
        # cosine similarity than fp32/mixed even when correctly built (random
        # noise is out-of-distribution and pushes this further -- confirmed
        # cos~0.86-0.92 at int8 on real images across S/S+/B/L in this repo's
        # benchmarks, with <1 mIoU point of real accuracy cost). A broken
        # engine shows up as NaN or a catastrophic cos (<0.5, the range seen
        # for L's actual pre-fix FP16 collapse), not a moderate int8 dip.
        print(
            "\nverifying engines vs. eager PyTorch (random-input cosine-sim check)..."
        )
        ok = True
        for entry in result.manifest:
            v = verify_engines(args.ckpt, entry, device=args.device)
            threshold = 0.5 if entry["precision"] == "int8" else 0.999
            passed = not v["has_nan"] and v["cos"] > threshold
            status = "NaN!!" if v["has_nan"] else ("ok" if passed else "LOW COS")
            ok &= passed
            print(
                f"  [{entry['precision']:5s}] cos={v['cos']:.5f}  "
                f"max_abs={v['max_abs']:.4g}  {status}"
            )
        if not ok:
            raise SystemExit(
                "\nverification FAILED for at least one precision -- "
                "do not deploy these engines."
            )
        print(
            "\nall engines verified (int8's lower cosine similarity vs. "
            "fp32/mixed is expected quantization noise, not a defect -- "
            "see the accuracy notebook for the real segmentation-accuracy check)."
        )


if __name__ == "__main__":
    main()
