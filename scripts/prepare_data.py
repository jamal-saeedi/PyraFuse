#!/usr/bin/env python3
"""Download and build the skin/fabric/background dataset.

Ensures everything under ``data/`` needed for :mod:`pyrafuse.data` exists,
downloading or building only what is missing (or corrupt/truncated):

    1. annotations    Fashionpedia instance/attribute JSON (train + val)
    2. images         Fashionpedia images -> data/raw/{train,test}/
    3. visuaal-masks  visuAAL binary skin masks -> data/raw/visuaal/
    4. labels         fused 3-class masks -> data/processed/{train,val}_labels/
                       (via pyrafuse.data.masks.build_split)

Every step first checks whether its output already looks complete and skips
itself if so, so this is safe (and fast) to re-run any time, e.g. after a
partial/interrupted download.

Usage:
    python scripts/prepare_data.py                 # do whatever is missing
    python scripts/prepare_data.py --only labels    # (re)build labels only
    python scripts/prepare_data.py --force          # redo every step
    python scripts/prepare_data.py --limit 50       # cap masks/split (smoke test)
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))  # importable even without `pip install -e .`

DEFAULT_RAW = REPO_ROOT / "data" / "raw"
DEFAULT_PROCESSED = REPO_ROOT / "data" / "processed"

FASHIONPEDIA_BASE = "https://s3.amazonaws.com/ifashionist-dataset"
ZENODO_VISUAAL_ZIP = (
    "https://zenodo.org/records/6973396/files/"
    "visuAAL%20Skin%20Segmentation%20Dataset.zip?download=1"
)

STEP_ORDER = ["annotations", "images", "visuaal-masks", "labels"]


# --------------------------------------------------------------------------- #
# low-level helpers
# --------------------------------------------------------------------------- #

def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    done = min(block_num * block_size, total_size)
    pct = done / total_size * 100
    print(f"\r    {done / 1e6:9.1f} / {total_size / 1e6:.1f} MB ({pct:5.1f}%)",
          end="", flush=True)


def download(url: str, dest: Path) -> None:
    """Download ``url`` to ``dest`` atomically (via a sibling ``.part`` file)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    print(f"  GET {url}")
    try:
        urllib.request.urlretrieve(url, tmp, _reporthook)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    print()
    tmp.rename(dest)


def extract_zip(zip_path: Path, target_dir: Path, flatten: bool) -> None:
    """Extract ``zip_path`` into ``target_dir``.

    If ``flatten``, strip a single common top-level directory from every
    archive member so files land directly in ``target_dir`` regardless of
    whether the archive wraps them in a folder (Fashionpedia's zips vary).
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if not n.endswith("/")]
        strip = ""
        if flatten:
            tops = {n.split("/", 1)[0] for n in members if "/" in n}
            if len(tops) == 1:
                strip = next(iter(tops)) + "/"
        for name in members:
            rel = name[len(strip):] if strip and name.startswith(strip) else name
            if not rel:
                continue
            out_path = target_dir / rel
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, open(out_path, "wb") as dst:
                shutil.copyfileobj(src, dst)


def file_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def dir_populated(path: Path, min_count: int = 1, pattern: str = "*") -> bool:
    """True if `path` has >= min_count files matching `pattern`, none of them
    zero-byte. A previous interrupted download/extraction can leave a
    directory with the right file *count* but mostly empty (0-byte) stubs,
    so checking count alone is not enough to call a directory "ready"."""
    if not path.is_dir():
        return False
    files = list(path.glob(pattern))
    if len(files) < min_count:
        return False
    return all(f.stat().st_size > 0 for f in files)


# --------------------------------------------------------------------------- #
# steps
# --------------------------------------------------------------------------- #

@dataclass
class Step:
    name: str
    is_ready: Callable[[], bool]
    run: Callable[[], None]


def _step_annotations(raw: Path) -> Step:
    train_json = raw / "instances_attributes_train2020.json"
    val_json = raw / "instances_attributes_val2020.json"

    def ready() -> bool:
        return file_nonempty(train_json) and file_nonempty(val_json)

    def run() -> None:
        if file_nonempty(train_json):
            print("  train annotations already present, skipping")
        else:
            download(f"{FASHIONPEDIA_BASE}/annotations/instances_attributes_train2020.json",
                      train_json)
        if file_nonempty(val_json):
            print("  val annotations already present, skipping")
        else:
            download(f"{FASHIONPEDIA_BASE}/annotations/instances_attributes_val2020.json",
                      val_json)

    return Step("annotations", ready, run)


def _step_images(raw: Path) -> Step:
    train_dir, test_dir = raw / "train", raw / "test"

    def ready() -> bool:
        return (dir_populated(train_dir, 100, "*.jpg")
                and dir_populated(test_dir, 100, "*.jpg"))

    def run() -> None:
        if dir_populated(train_dir, 100, "*.jpg"):
            print("  train images already present, skipping")
        else:
            zip_path = raw / "_tmp_train2020.zip"
            download(f"{FASHIONPEDIA_BASE}/images/train2020.zip", zip_path)
            print("  extracting train images...")
            extract_zip(zip_path, train_dir, flatten=True)
            zip_path.unlink()
        if dir_populated(test_dir, 100, "*.jpg"):
            print("  val/test images already present, skipping")
        else:
            zip_path = raw / "_tmp_val_test2020.zip"
            download(f"{FASHIONPEDIA_BASE}/images/val_test2020.zip", zip_path)
            print("  extracting val/test images...")
            extract_zip(zip_path, test_dir, flatten=True)
            zip_path.unlink()

    return Step("images", ready, run)


def _step_visuaal(raw: Path) -> Step:
    # Must match VISUAAL_SUBDIR in pyrafuse/data/masks.py. The Zenodo archive
    # is flat (train_masks/, val_masks/ at its root) -> extract straight into
    # raw/visuaal with no extra wrapping folder.
    root = raw / "visuaal"
    train_masks, val_masks = root / "train_masks", root / "val_masks"

    def ready() -> bool:
        return dir_populated(train_masks, 100) and dir_populated(val_masks, 100)

    def run() -> None:
        zip_path = raw / "_tmp_visuaal.zip"
        download(ZENODO_VISUAAL_ZIP, zip_path)
        print("  extracting visuAAL masks...")
        extract_zip(zip_path, root, flatten=False)
        zip_path.unlink()

    return Step("visuaal-masks", ready, run)


def _step_labels(raw: Path, processed: Path, limit: int | None) -> Step:
    train_labels, val_labels = processed / "train_labels", processed / "val_labels"

    def ready() -> bool:
        return (dir_populated(train_labels, 100, "*.png")
                and dir_populated(val_labels, 100, "*.png"))

    def run() -> None:
        from pyrafuse.data.masks import build_split
        build_split("val", limit=limit, raw_dir=raw, processed_dir=processed)
        build_split("train", limit=limit, raw_dir=raw, processed_dir=processed)

    return Step("labels", ready, run)


def build_steps(raw: Path, processed: Path, limit: int | None) -> dict[str, Step]:
    return {
        "annotations": _step_annotations(raw),
        "images": _step_images(raw),
        "visuaal-masks": _step_visuaal(raw),
        "labels": _step_labels(raw, processed, limit),
    }


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="+", choices=STEP_ORDER,
                    help="run only these steps (default: all, in order)")
    p.add_argument("--force", action="store_true",
                    help="redo steps even if their output already looks complete")
    p.add_argument("--limit", type=int, default=None,
                    help="cap masks built per split, for a quick smoke test")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW)
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    args = p.parse_args()

    steps = build_steps(args.raw_dir, args.processed_dir, args.limit)
    names = args.only or STEP_ORDER

    for name in names:
        step = steps[name]
        print(f"[{name}]")
        if not args.force and step.is_ready():
            print("  already present, skipping (use --force to redo)")
            continue
        step.run()

    print("\ndone.")


if __name__ == "__main__":
    main()
