"""ExecuTorch export of PyraFuse checkpoints for Android, iOS, and edge devices.

A PyraFuse checkpoint is wrapped in :class:`MobileSegmenter`, which bakes the
ImageNet normalisation into the graph, captured with ``torch.export``, and
lowered to one ``.pte`` program per target:

================  ============================================  ==================
target            runs on                                       precision
================  ============================================  ==================
``xnnpack_fp32``  CPU: Android, iOS, macOS, Linux/ARM edge      FP32
``xnnpack_int8``  same CPUs; ~4x smaller                        INT8 weights, dyn.
                                                                INT8 activations
``coreml_fp32``   iOS 17+ / macOS 14+: GPU and CPU via Core ML  FP32
``vulkan_fp32``   Android GPU via Vulkan                        FP32
================  ============================================  ==================

Every program shares one I/O contract, matching the ``forward`` schema that
``react-native-executorch``'s semantic-segmentation task validates:

* input  ``image``  ``float32 [1, 3, H, W]`` — RGB, **scaled to [0, 1]**
  (divide 8-bit pixels by 255; no mean/std step on the device).
* output ``logits`` ``float32 [1, 3, H, W]`` — classes ``0 = background``,
  ``1 = fabric``, ``2 = skin``; take ``argmax`` over dim 1 for the mask.

Shapes are static (``H = W = image_size``, default 448, a multiple of 16):
resize the camera frame to that size and resize the mask back afterwards.

Four export details matter:

1. The DINOv3 adapter runs its backbone under ``@torch.no_grad``. Exporting
   with grad enabled turns that into a ``wrap_with_set_grad_enabled``
   higher-order op that no mobile backend can lower, so :func:`export_program`
   always traces under :func:`torch.no_grad`, which flattens it.
2. The backbone is switched to Hugging Face ``eager`` attention before export.
   The default SDPA path decomposes into masking ops XNNPACK cannot delegate,
   which fragments the CPU graph and costs ~1.6x latency for the same output.
3. **No FP16 targets.** DINOv3's first block has massive activation outliers:
   its attention logits reach ~2.5e6 (ViT-S) and ~1.4e5 (ViT-B), beyond the
   FP16 maximum of 65504. Any backend that materialises that matmul in FP16
   (Core ML on the Neural Engine, Vulkan ``force_fp16``) yields inf/NaN and a
   single-class mask. Core ML and Vulkan programs are therefore FP32.
4. For the same reason INT8 uses *dynamic* per-token activation quantisation
   (weights per-channel). Static per-tensor activation scales, calibrated on
   real images, collapse to a single-class mask (mIoU ~0.25).
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

from ..model.model_hybrid import HybridSegmenter

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLASS_NAMES = ("background", "fabric", "skin")


@dataclass(frozen=True)
class MobileTarget:
    """One ExecuTorch backend/precision combination."""

    name: str
    backend: str
    precision: str
    runs_on: str


MOBILE_TARGETS: dict[str, MobileTarget] = {
    t.name: t
    for t in (
        MobileTarget("xnnpack_fp32", "xnnpack", "fp32",
                     "CPU on Android, iOS, macOS, Linux/ARM edge"),
        MobileTarget("xnnpack_int8", "xnnpack", "int8",
                     "CPU on Android, iOS, macOS, Linux/ARM edge"),
        MobileTarget("coreml_fp32", "coreml", "fp32",
                     "Apple GPU / CPU via Core ML (iOS 17+, macOS 14+)"),
        MobileTarget("vulkan_fp32", "vulkan", "fp32",
                     "Android GPU via Vulkan"),
    )
}


class MobileSegmenter(nn.Module):
    """``[1, 3, H, W]`` RGB in ``[0, 1]`` -> ``[1, num_classes, H, W]`` logits."""

    def __init__(self, model: HybridSegmenter):
        super().__init__()
        self.model = model
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.model((image - self.mean) / self.std)


def load_checkpoint(ckpt_dir: str | Path) -> MobileSegmenter:
    """Load a checkpoint's EMA weights on CPU, frozen and in eval mode."""
    model = HybridSegmenter.from_pretrained(
        ckpt_dir, map_location="cpu", load_ema_into_model=True).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    # Plain matmul/softmax attention: SDPA decomposes into masking ops that
    # XNNPACK cannot delegate (~1.6x slower on CPU), with identical outputs.
    model.encoder.m.set_attn_implementation("eager")
    return MobileSegmenter(model).eval()


def export_program(module: nn.Module, image_size: int) -> torch.export.ExportedProgram:
    """Capture *module* with a static ``[1, 3, S, S]`` input (see note 1 above)."""
    sample = (torch.rand(1, 3, image_size, image_size),)
    with torch.no_grad():
        return torch.export.export(module, sample, strict=False)


def quantize_int8(module: MobileSegmenter, image_size: int) -> nn.Module:
    """PT2E INT8 for XNNPACK: per-channel weights, dynamic per-token activations.

    Activation scales are computed at run time, so no calibration set is
    needed and the first block's outliers stay local to their tokens.
    """
    from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
        XNNPACKQuantizer,
        get_symmetric_quantization_config,
    )
    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    quantizer = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=True, is_dynamic=True))
    prepared = prepare_pt2e(export_program(module, image_size).module(), quantizer)
    with torch.no_grad():
        prepared(torch.rand(1, 3, image_size, image_size))  # populate observers
    return convert_pt2e(prepared)


def _partitioners(target: MobileTarget):
    if target.backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )

        return [XnnpackPartitioner()]
    if target.backend == "coreml":
        import coremltools as ct
        from executorch.backends.apple.coreml.compiler import CoreMLBackend
        from executorch.backends.apple.coreml.partition import CoreMLPartitioner

        specs = CoreMLBackend.generate_compile_specs(
            compute_unit=ct.ComputeUnit.ALL,
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT32,  # see note 3
        )
        return [CoreMLPartitioner(compile_specs=specs)]
    if target.backend == "vulkan":
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
            VulkanPartitioner,
        )

        return [VulkanPartitioner()]  # FP32; see note 3
    raise ValueError(f"Unknown backend '{target.backend}'.")


def _delegation_stats(edge_program) -> tuple[int, list[str]]:
    """Return (#delegate calls, sorted non-delegated op names) of a lowered graph."""
    delegates, leftovers = 0, set()
    for node in edge_program.graph_module.graph.nodes:
        if node.op != "call_function":
            continue
        name = getattr(node.target, "__name__", str(node.target))
        if "executorch_call_delegate" in name:
            delegates += 1
        elif name != "getitem":
            leftovers.add(name)
    return delegates, sorted(leftovers)


def lower_to_pte(program: torch.export.ExportedProgram, target: MobileTarget,
                 path: Path) -> dict:
    """Lower *program* for *target*, write ``path``, and return build metadata."""
    from executorch.exir import to_edge_transform_and_lower

    start = time.time()
    edge = to_edge_transform_and_lower(program, partitioner=_partitioners(target))
    delegates, leftovers = _delegation_stats(edge.exported_program())
    et_program = edge.to_executorch()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(et_program.buffer)
    return {
        "file": path.name,
        "target": target.name,
        "backend": target.backend,
        "precision": target.precision,
        "runs_on": target.runs_on,
        "size_bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "delegate_calls": delegates,
        "non_delegated_ops": leftovers,
        "build_time_s": round(time.time() - start, 1),
    }


def export_mobile(
    ckpt_dir: str | Path,
    out_dir: str | Path,
    label: str,
    targets: Sequence[str] = tuple(MOBILE_TARGETS),
    image_size: int = 448,
) -> list[dict]:
    """Export one checkpoint to ``<out_dir>/pyrafuse_<label>_<target>.pte`` files.

    Returns one metadata record per written program.
    """
    if image_size % 16:
        raise ValueError(f"image_size must be a multiple of 16, got {image_size}.")
    unknown = set(targets) - set(MOBILE_TARGETS)
    if unknown:
        raise ValueError(f"Unknown targets {sorted(unknown)}; choose from "
                         f"{list(MOBILE_TARGETS)}.")
    out_dir = Path(out_dir)
    module = load_checkpoint(ckpt_dir)

    records = []
    for name in targets:
        target = MOBILE_TARGETS[name]
        if target.precision == "int8":
            program = export_program(quantize_int8(module, image_size), image_size)
        else:
            # Lowering consumes the program's constants: export per target.
            program = export_program(module, image_size)
        path = out_dir / f"pyrafuse_{label}_{name}.pte"
        record = lower_to_pte(program, target, path)
        record.update(image_size=image_size, variant=label)
        records.append(record)
        print(f"[{label}] {name}: {record['size_bytes'] / 2**20:.1f} MiB, "
              f"{record['delegate_calls']} delegate call(s), "
              f"non-delegated: {record['non_delegated_ops'] or 'none'}")
    return records


def load_mobile_model(path: str | Path):
    """Load a ``.pte`` with the ExecuTorch Python runtime; returns ``image -> logits``.

    Only backends compiled into the installed runtime can execute (the PyPI
    ``executorch`` wheel ships XNNPACK on Linux; Core ML needs macOS).
    """
    from executorch.runtime import Runtime

    method = Runtime.get().load_program(Path(path)).load_method("forward")

    def run(image: torch.Tensor) -> torch.Tensor:
        return method.execute([image.contiguous()])[0]

    return run
