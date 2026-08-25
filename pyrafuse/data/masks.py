"""
Build 3-class segmentation masks (background / fabric / skin) by fusing the
visuAAL Skin Segmentation Dataset with Fashionpedia instance annotations.

Both datasets share image IDs: a visuAAL mask file `<id>.jpg` corresponds to the
Fashionpedia image whose `file_name`/`kaggle_id` is `<id>.jpg`. Fashionpedia
gives polygon/RLE instance masks for 46 apparel categories (the "fabric"); the
visuAAL mask gives pixel-level skin. We combine them with priority

    skin > fabric > background

so skin always wins where the two overlap (e.g. an arm in front of a garment).

Label values in the output PNG:
    0 = background   1 = fabric   2 = skin
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as coco_mask

BACKGROUND, FABRIC, SKIN = 0, 1, 2
SKIN_THRESH = 127  # visuAAL masks are JPEG-compressed binary (0/255)

# Repo root: this file lives at <root>/pyrafuse/data/masks.py
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = REPO_ROOT / "data" / "raw"
DEFAULT_PROCESSED = REPO_ROOT / "data" / "processed"
# The Zenodo archive ("visuAAL Skin Segmentation Dataset.zip") is flat --
# train_masks/ and val_masks/ sit at its root, no wrapping folder.
VISUAAL_SUBDIR = Path("visuaal")


def load_fashionpedia(split: str, raw_dir: Path = DEFAULT_RAW):
    """Return (annotations_by_image_id, image_meta_by_id, name_to_id) for split."""
    ann_path = raw_dir / f"instances_attributes_{split}2020.json"
    data = json.load(open(ann_path))

    img_meta = {im["id"]: im for im in data["images"]}
    # map the kaggle/file-name id (what visuAAL uses) -> Fashionpedia integer id
    name_to_id = {Path(im["file_name"]).stem: im["id"]
                  for im in data["images"]}

    anns_by_img: dict[int, list] = {}
    for ann in data["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)
    return anns_by_img, img_meta, name_to_id


def fabric_mask(anns: list, height: int, width: int) -> np.ndarray:
    """Union of all instance segmentations for one image -> bool (H, W)."""
    mask = np.zeros((height, width), dtype=bool)
    for ann in anns:
        seg = ann["segmentation"]
        if isinstance(seg, list):  # polygons
            rles = coco_mask.frPyObjects(seg, height, width)
            rle = coco_mask.merge(rles)
        elif isinstance(seg["counts"], list):  # uncompressed RLE
            rle = coco_mask.frPyObjects(seg, height, width)
        else:  # compressed RLE
            rle = seg
        mask |= coco_mask.decode(rle).astype(bool)
    return mask


def build_label(
    skin_path: Path,
    anns: list,
    height: int,
    width: int,
) -> np.ndarray:
    """Composite one image's skin mask and fabric instances into a label map."""
    skin = np.array(Image.open(skin_path).convert("L"))
    if skin.shape != (height, width):  # guard against any size mismatch
        skin = np.array(Image.fromarray(skin).resize(
            (width, height), Image.NEAREST))
    skin = skin > SKIN_THRESH

    fabric = fabric_mask(anns, height, width)

    label = np.full((height, width), BACKGROUND, dtype=np.uint8)
    label[fabric] = FABRIC
    label[skin] = SKIN  # skin overrides fabric on overlap
    return label


def build_split(
    split: str,
    limit: int | None = None,
    raw_dir: Path = DEFAULT_RAW,
    processed_dir: Path = DEFAULT_PROCESSED,
) -> None:
    mask_dir = raw_dir / VISUAAL_SUBDIR / f"{split}_masks"
    out_dir = processed_dir / f"{split}_labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    anns_by_img, img_meta, name_to_id = load_fashionpedia(split, raw_dir)

    skin_files = sorted(mask_dir.glob("*.jpg"))
    if limit:
        skin_files = skin_files[:limit]

    written = missing = 0
    for skin_path in skin_files:
        stem = skin_path.stem
        img_id = name_to_id.get(stem)
        if img_id is None:
            missing += 1
            continue

        meta = img_meta[img_id]
        label = build_label(skin_path, anns_by_img.get(img_id, []),
                            meta["height"], meta["width"])
        Image.fromarray(label).save(out_dir / f"{stem}.png")
        written += 1

    print(f"[{split}] wrote {written} masks to {out_dir} "
          f"({missing} skin masks had no Fashionpedia match)")
