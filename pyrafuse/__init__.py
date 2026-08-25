"""PyraFuse: skin, fabric, and background semantic segmentation."""

from importlib.metadata import PackageNotFoundError, version

from .zoo import (
    MODEL_VARIANTS,
    OFFICIAL_MODEL_REPO,
    available_models,
    load_pretrained,
    resolve_checkpoint,
)

try:
    __version__ = version("pyrafuse")
except PackageNotFoundError:
    __version__ = "1.0.0"

__all__ = [
    "MODEL_VARIANTS",
    "OFFICIAL_MODEL_REPO",
    "__version__",
    "available_models",
    "load_pretrained",
    "resolve_checkpoint",
]
