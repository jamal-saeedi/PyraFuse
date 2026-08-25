"""
PyTorch Dataset for 3-class skin / fabric / background segmentation.

Pairs Fashionpedia RGB images with the pre-built label masks produced by
:func:`pyrafuse.data.masks.build_split` (0=background, 1=fabric,
2=skin). Images and labels are joined by stem (`<id>.jpg` <-> `<id>.png`).

The DINOv3 backbone uses a patch size of 16, so the spatial size must be a
multiple of 16; ``size`` defaults to 512.
"""

from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.core.transforms_interface import DualTransform
from PIL import Image
from torch.utils.data import Dataset

from .masks import BACKGROUND, FABRIC, SKIN  # noqa: F401  (re-exported for callers)

# ImageNet statistics (DINOv3 pretraining normalisation).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
NUM_CLASSES = 3
PATCH = 16


class SkinToneShift(DualTransform):
    """Shift hue, saturation, and brightness of skin pixels only.

    Operates in HSV space on the pixels where ``mask == SKIN``.  Shifts are
    sampled uniformly within the given limits and then **biased toward darker,
    more-saturated values** to over-generate the under-represented dark-tone
    examples (Fitzpatrick Types V–VI) while remaining within a realistic skin
    envelope.

    Args:
        hue_shift_limit: max absolute hue shift in degrees (±).  Kept small
            (default 8°) so skin never drifts into unnatural colours.
        sat_shift_limit: max absolute saturation shift as a fraction of 255
            (default 30).  Positive bias pushes toward richer, darker tones.
        val_shift_limit: max absolute brightness shift (default 40).  A
            downward bias darkens skin to mimic Types V–VI.
        dark_bias: fraction [0, 1] of the ``val_shift_limit`` subtracted from
            the sampled shift, biasing toward darker outputs.  0 = no bias,
            1 = always darken by the full limit.  Default 0.3.
        p: probability of applying the transform.
    """

    def __init__(
        self,
        hue_shift_limit: int = 8,
        sat_shift_limit: int = 30,
        val_shift_limit: int = 40,
        dark_bias: float = 0.3,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(p=p)
        self.hue_shift_limit = hue_shift_limit
        self.sat_shift_limit = sat_shift_limit
        self.val_shift_limit = val_shift_limit
        self.dark_bias = dark_bias

    def apply(self, img: np.ndarray, skin_mask: np.ndarray | None = None, **params) -> np.ndarray:
        if skin_mask is None or img.dtype != np.uint8:
            return img
        return self._shift_skin(img, skin_mask)

    def apply_to_mask(self, mask: np.ndarray, **params) -> np.ndarray:
        return mask  # mask is never modified

    def get_params_dependent_on_data(self, params: dict, data: dict) -> dict:
        # Expose the segmentation mask to apply() so the shift is skin-only.
        return {"skin_mask": data.get("mask")}

    def _shift_skin(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        skin_px = mask == SKIN
        if not skin_px.any():
            return img

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int32)

        # OpenCV HSV: H in [0,179], S and V in [0,255]
        h_shift = int(
            np.random.uniform(-self.hue_shift_limit, self.hue_shift_limit))
        s_shift = int(
            np.random.uniform(-self.sat_shift_limit, self.sat_shift_limit))
        v_shift = int(
            np.random.uniform(-self.val_shift_limit, self.val_shift_limit))
        # Subtract a fraction of val_shift_limit to bias toward darker values.
        v_shift -= int(self.dark_bias * self.val_shift_limit)

        # Skin hue envelope: ~0–20° in standard degrees → 0–10 in OpenCV [0,179].
        # Allow small excursions but clamp hard so we never leave warm skin tones.
        HUE_MIN, HUE_MAX = 0, 20   # OpenCV units

        hsv[skin_px, 0] = np.clip(hsv[skin_px, 0] + h_shift, HUE_MIN, HUE_MAX)
        hsv[skin_px, 1] = np.clip(hsv[skin_px, 1] + s_shift, 30, 255)
        hsv[skin_px, 2] = np.clip(hsv[skin_px, 2] + v_shift, 20, 255)

        rgb_shifted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        result = img.copy()
        result[skin_px] = rgb_shifted[skin_px]
        return result

    def get_transform_init_args_dict(self) -> dict:
        return {
            "hue_shift_limit": self.hue_shift_limit,
            "sat_shift_limit": self.sat_shift_limit,
            "val_shift_limit": self.val_shift_limit,
            "dark_bias": self.dark_bias,
        }


def build_train_transform(size: int, normalize: bool = True) -> A.Compose:
    """Augmentation pipeline for training semantic segmentation.

    Geometric and photometric transforms are applied jointly to the image and
    its label mask (``additional_targets`` is not needed because albumentations
    routes the ``mask`` argument through mask-safe interpolation). Mask values
    {0, 1, 2} are preserved by using nearest-neighbour resampling for any
    geometric op.
    """
    transforms = [
        # --- geometric (applied to image + mask) ---
        A.HorizontalFlip(p=0.5),
        A.Affine(
            scale=(0.85, 1.15),
            translate_percent=(0.0, 0.05),
            rotate=(-15, 15),
            shear=(-5, 5),
            interpolation=1,        # bilinear for image
            mask_interpolation=0,   # nearest for mask -> labels stay {0,1,2}
            border_mode=0,          # constant pad
            fill=0,
            fill_mask=BACKGROUND,
            p=0.5,
        ),
        A.RandomResizedCrop(
            size=(size, size),
            scale=(0.5, 1.0),
            ratio=(0.75, 1.333),
            interpolation=1,
            mask_interpolation=0,
            p=0.5,
        ),
        # Guarantee final spatial size regardless of which crop branch ran.
        A.Resize(size, size, interpolation=1, mask_interpolation=0),
        # --- skin-tone shift: mask-conditioned, biased toward darker tones ---
        # Shifts hue/sat/val only on skin pixels so fabric and background are
        # untouched. The dark_bias compensates for the scarcity of Fitzpatrick
        # Types V-VI in the val set (132 and 61 images vs ~260 for Types II-III).
        SkinToneShift(
            hue_shift_limit=8,
            sat_shift_limit=30,
            val_shift_limit=40,
            dark_bias=0.3,
            p=0.6,
        ),
        # --- photometric: skin-tone & lighting variation (image only) ---
        # Skin appearance varies enormously by tone, illuminant colour and
        # exposure. We push these harder than generic defaults so the model
        # learns colour-/lighting-invariant skin features rather than memorising
        # a narrow "average" skin colour from the training set.
        A.RandomBrightnessContrast(
            brightness_limit=0.3, contrast_limit=0.3, p=0.7
        ),
        # Hue/sat shift widens the *range* of skin tones the model sees; keep hue
        # modest so skin doesn't turn unnatural (green/blue), but allow large
        # saturation/value swings to mimic tone + exposure differences.
        A.HueSaturationValue(
            hue_shift_limit=15, sat_shift_limit=35, val_shift_limit=20, p=0.5
        ),
        # Coloured illuminant (warm tungsten vs cool daylight vs shade) heavily
        # changes apparent skin colour; RGBShift simulates a colour cast.
        A.RGBShift(r_shift_limit=15, g_shift_limit=15,
                   b_shift_limit=15, p=0.3),
        # Non-linear tone response (under/over-exposure, different cameras).
        A.RandomGamma(gamma_limit=(70, 130), p=0.3),
        # Uneven lighting: cast shadows / highlights across the body & fabric.
        A.OneOf(
            [
                A.RandomShadow(shadow_roi=(0, 0, 1, 1),
                               num_shadows_limit=(1, 3)),
                A.RandomToneCurve(scale=0.2),
            ],
            p=0.3,
        ),
        # Sensor / motion degradations.
        A.OneOf(
            [
                A.GaussianBlur(blur_limit=(3, 5)),
                A.GaussNoise(),
                A.MotionBlur(blur_limit=5),
                A.ImageCompression(quality_range=(40, 90)),
            ],
            p=0.3,
        ),
        # Occlusion: force reliance on local skin cues by hiding random patches.
        # fill_mask=BACKGROUND so dropped regions are ignored / treated as bg in
        # the label rather than leaking the original class.
        A.CoarseDropout(
            num_holes_range=(1, 6),
            hole_height_range=(0.05, 0.15),
            hole_width_range=(0.05, 0.15),
            fill=0,
            fill_mask=BACKGROUND,
            p=0.3,
        ),
    ]
    if normalize:
        transforms.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return A.Compose(transforms)


def build_eval_transform(size: int, normalize: bool = True) -> A.Compose:
    """Deterministic resize-only pipeline for validation / inference."""
    transforms = [A.Resize(size, size, interpolation=1, mask_interpolation=0)]
    if normalize:
        transforms.append(A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    return A.Compose(transforms)


class SkinFabricDataset(Dataset):
    """3-class segmentation: RGB image -> {0: background, 1: fabric, 2: skin}.

    Args:
        images_dir: directory of Fashionpedia ``<id>.jpg`` images.
        labels_dir: directory of ``<id>.png`` label maps from ``build_split``.
        size: square resize edge (multiple of 16).
        normalize: apply ImageNet mean/std to the image tensor.
        train: if True, apply the randomised training augmentation pipeline
            (flips, affine, random-resized crop, photometric jitter); if False,
            use a deterministic resize-only pipeline.
        transform: optional explicit albumentations ``Compose`` overriding the
            ``train`` flag. Must output an image (and pass ``mask`` through) and,
            if ``normalize`` is desired, include an ``A.Normalize`` step.
    """

    def __init__(
        self,
        images_dir: str | Path,
        labels_dir: str | Path,
        size: int = 512,
        normalize: bool = True,
        train: bool = False,
        transform: A.Compose | None = None,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        if size % PATCH != 0:
            raise ValueError(f"size must be a multiple of {PATCH}, got {size}")
        self.size = size
        self.normalize = normalize
        self.train = train

        if transform is not None:
            self.transform = transform
        elif train:
            self.transform = build_train_transform(size, normalize=normalize)
        else:
            self.transform = build_eval_transform(size, normalize=normalize)

        # Only keep stems that have both an image and a label.
        label_stems = {p.stem for p in self.labels_dir.glob("*.png")}
        self.stems = sorted(
            p.stem for p in self.images_dir.glob("*.jpg")
            if p.stem in label_stems
        )
        if not self.stems:
            raise RuntimeError(
                f"no image/label pairs found between {self.images_dir} "
                f"and {self.labels_dir}"
            )

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> dict:
        stem = self.stems[idx]
        img = Image.open(self.images_dir / f"{stem}.jpg").convert("RGB")
        lbl = Image.open(self.labels_dir / f"{stem}.png")

        # albumentations works on numpy HWC uint8 image + HW mask.
        out = self.transform(
            image=np.asarray(img, dtype=np.uint8),
            mask=np.asarray(lbl, dtype=np.uint8),
        )
        image_np, label_np = out["image"], out["mask"]

        # A.Normalize already scales to ImageNet stats and float32; otherwise the
        # image is still uint8 [0, 255] and needs manual scaling to [0, 1].
        image = torch.from_numpy(image_np.transpose(2, 0, 1))
        if not self.normalize:
            image = image.float() / 255.0
        else:
            image = image.float()

        label = torch.from_numpy(label_np.astype(np.int64))  # (H, W)

        return {"image": image, "label": label, "stem": stem}
