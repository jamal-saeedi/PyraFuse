"""Model-zoo helpers for local and Hugging Face PyraFuse checkpoints.

PyraFuse checkpoints are directories, rather than a single pickle, so they
remain auditable and can be loaded offline after their first download::

    from pyrafuse import load_pretrained
    model = load_pretrained("base", revision="v1.0.0")

The official public repository is ``jamal-one/PyraFuse``. A caller can set
``PYRAFUSE_MODEL_REPO`` or pass ``repo_id`` to use a private mirror or fork.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Variant = Literal["small", "small_plus", "base", "large"]

# Official public release. Set PYRAFUSE_MODEL_REPO or pass repo_id to use a
# private mirror or a fork instead.
OFFICIAL_MODEL_REPO = "jamal-one/PyraFuse"


@dataclass(frozen=True)
class ModelSpec:
    """Published PyraFuse checkpoint variant metadata."""

    name: Variant
    backbone: str
    checkpoint_dir: str


MODEL_VARIANTS: dict[str, ModelSpec] = {
    "small": ModelSpec("small", "DINOv3 ViT-S/16", "models/finals/small"),
    "small_plus": ModelSpec(
        "small_plus", "DINOv3 ViT-S+/16", "models/finals/small_plus"
    ),
    "base": ModelSpec("base", "DINOv3 ViT-B/16", "models/finals/base"),
    "large": ModelSpec("large", "DINOv3 ViT-L/16", "models/finals/large"),
}


def available_models() -> tuple[ModelSpec, ...]:
    """Return the official model variants in deployment-size order."""
    return tuple(MODEL_VARIANTS.values())


def _is_checkpoint(path: Path) -> bool:
    return (path / "config.json").is_file() and (path / "decoder.pt").is_file()


def resolve_checkpoint(
    model: str | Path = "base",
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    local_files_only: bool = False,
) -> Path:
    """Resolve a local checkpoint or download one named model variant.

    A checked-out ``models/finals/<variant>`` is preferred. Otherwise, a
    complete variant folder is fetched from the configured Hugging Face model
    repository and returned from the Hub cache. Pin *revision* to a Hub tag
    or commit SHA for reproducible experiments.
    """
    candidate = Path(model).expanduser()
    if _is_checkpoint(candidate):
        return candidate.resolve()

    name = str(model)
    if name not in MODEL_VARIANTS:
        choices = ", ".join(MODEL_VARIANTS)
        raise ValueError(
            f"Unknown model '{name}'. Pass a checkpoint directory or one of: {choices}."
        )

    local = Path(MODEL_VARIANTS[name].checkpoint_dir)
    if _is_checkpoint(local):
        return local.resolve()

    repo_id = repo_id or os.environ.get("PYRAFUSE_MODEL_REPO") or OFFICIAL_MODEL_REPO

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:  # pragma: no cover
        message = "Install `huggingface-hub` to download PyraFuse weights."
        raise ImportError(message) from error

    root = Path(
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            token=token,
            allow_patterns=[f"{name}/**"],
            local_files_only=local_files_only,
        )
    )
    checkpoint = root / name
    if not _is_checkpoint(checkpoint):
        raise FileNotFoundError(
            f"Model repository '{repo_id}' does not contain a complete "
            f"'{name}' checkpoint."
        )
    return checkpoint


def load_pretrained(
    model: str | Path = "base",
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    device: str = "cpu",
    use_ema: bool = True,
    token: str | bool | None = None,
    local_files_only: bool = False,
):
    """Load an eager PyTorch PyraFuse model in evaluation mode.

    For best reproducibility, use an immutable Hugging Face *revision* and
    the EMA weights (the default) used for evaluation in this repository.
    """
    from .model.model_hybrid import HybridSegmenter

    checkpoint = resolve_checkpoint(
        model,
        repo_id=repo_id,
        revision=revision,
        token=token,
        local_files_only=local_files_only,
    )
    return (
        HybridSegmenter.from_pretrained(
            checkpoint,
            map_location=device,
            load_ema_into_model=use_ema,
        )
        .to(device)
        .eval()
    )
