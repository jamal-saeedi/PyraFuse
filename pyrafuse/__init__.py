"""PyraFuse: skin, fabric, and background semantic segmentation."""

from importlib.metadata import PackageNotFoundError, version

from .zoo import MODEL_VARIANTS, available_models, load_pretrained, resolve_checkpoint

try:
    __version__ = version("pyrafuse")
except PackageNotFoundError:
    __version__ = "dev"

__all__ = [
    "MODEL_VARIANTS",
    "__version__",
    "available_models",
    "load_pretrained",
    "resolve_checkpoint",
]
