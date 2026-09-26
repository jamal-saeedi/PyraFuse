"""Deployment helpers: TensorRT (``trt_export``) and ExecuTorch (``mobile_export``).

Both submodules pull in heavy optional dependencies (``tensorrt`` and
``executorch``), so names are resolved lazily: importing
``pyrafuse.deploy.mobile_export`` never requires TensorRT and vice versa.
"""

from importlib import import_module

_TRT_NAMES = (
    "PRECISIONS",
    "DecoderWrap",
    "EncoderWrap",
    "ExportResult",
    "TRTRunner",
    "build_engine",
    "export_and_build",
    "export_onnx",
    "load_checkpoint",
    "pin_sensitive_layers_fp32",
    "verify_engines",
)
_MOBILE_NAMES = (
    "MOBILE_TARGETS",
    "MobileSegmenter",
    "export_mobile",
    "load_mobile_model",
)

__all__ = [*_TRT_NAMES, *_MOBILE_NAMES]


def __getattr__(name: str):
    if name in _TRT_NAMES:
        return getattr(import_module(".trt_export", __name__), name)
    if name in _MOBILE_NAMES:
        return getattr(import_module(".mobile_export", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
