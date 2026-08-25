"""Reusable ONNX export + TensorRT engine build for HybridSegmenter checkpoints.

Generalizes the encoder/decoder export -> build -> runner pipeline verified in
notebooks/backbone_size_trt_pipeline.ipynb to work with *any* HybridSegmenter
checkpoint, not just the four backbone-size checkpoints it was built for. It
carries forward three bug fixes found and verified while building that
notebook (relative to the original notebooks/benchmark_trt_pipeline.ipynb and
scripts/run_trt_pipeline_sweep.py):

  1. `build_engine` parses via `OnnxParser.parse_from_file(...)`, not
     `.parse(path.read_bytes())`. The latter cannot resolve a `*.onnx.data`
     external-weight sidecar (it does a relative-path lookup that fails once
     bytes are read into memory) -- every TRT row silently SKIPPED for any
     export needing a sidecar (anything near/over the ~2GB inline-protobuf
     limit, e.g. ViT-L/16's encoder).
  2. `pin_sensitive_layers_fp32` pins `ELEMENTWISE` (residual adds) in
     addition to `NORMALIZATION` / `SOFTMAX` / the RoPE subgraph. Without it,
     deep backbones (ViT-L/16, 24 blocks) accumulate FP16 overflow across
     residual adds and mixed/int8 inference silently produces NaN logits --
     even though the FP32 build matches eager exactly and small/random-noise
     sanity checks on shallower backbones look fine.
  3. `TRTRunner.__call__` synchronizes its private CUDA stream against the
     caller's current stream on both ends (`wait_stream` before and after
     `execute_async_v3`). Without this, TensorRT execution can start before an
     async (`non_blocking=True`) H2D copy into the input tensor -- e.g. from a
     pinned-memory DataLoader -- has actually landed; CUDA gives no ordering
     guarantee between independent streams. This silently corrupted
     end-to-end accuracy (mIoU ~0.42 vs. the correct ~0.91) while every
     isolated numerical check (random noise, single real batch) still passed.

Splits a `HybridSegmenter` checkpoint on its clean architectural boundary --
`model.encoder(pixel_values) -> List[4] of [B,C,ph,pw]` and
`model.decoder(feature_maps) -> logits [B,num_classes,H,W]` -- and exports
each half as its own ONNX graph (feature maps stacked/unstacked on dim 1 so
each graph has exactly one input tensor and one output tensor), independently
compiled to FP32 / mixed(FP16) / INT8 TensorRT engines.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

try:
    import tensorrt as trt
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "tensorrt is required for deep_sunscreen.src.deploy.trt_export; "
        "install the tensorrt-cu12 wheel matching your CUDA version."
    ) from e

from ..model.model_hybrid import HybridSegmenter

Precision = str  # "fp32" | "mixed" | "int8"
PRECISIONS: Tuple[Precision, ...] = ("fp32", "mixed", "int8")

_TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


# --------------------------------------------------------------------------
# Encoder / decoder wrappers -- the clean ONNX-exportable boundary.
# --------------------------------------------------------------------------


class EncoderWrap(torch.nn.Module):
    """pixel_values [B,3,H,W] -> feats [B,4,C,ph,pw] (4 levels stacked on dim 1)."""

    def __init__(self, model: HybridSegmenter):
        super().__init__()
        self.encoder = model.encoder

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        maps = self.encoder(pixel_values)
        return torch.stack(maps, dim=1)


class DecoderWrap(torch.nn.Module):
    """feats [B,4,C,ph,pw] -> logits [B,num_classes,ph,pw]."""

    def __init__(self, model: HybridSegmenter):
        super().__init__()
        self.decoder = model.decoder

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        maps = [feats[:, i] for i in range(feats.shape[1])]
        return self.decoder(maps)


def load_checkpoint(
    ckpt_dir, device: str = "cuda:0"
) -> Tuple[HybridSegmenter, int, int]:
    """Load a HybridSegmenter checkpoint (EMA weights) in eval mode.

    Returns (model, image_size, num_classes).
    """
    model = (
        HybridSegmenter.from_pretrained(
            ckpt_dir,
            map_location=device,
            load_ema_into_model=True,
        )
        .to(device)
        .eval()
    )
    return model, model.config.image_size, model.config.num_classes


# --------------------------------------------------------------------------
# ONNX export
# --------------------------------------------------------------------------


def export_onnx(
    module, sample_inputs, input_names, output_names, path, dynamic_shapes=None
) -> Path:
    """Export a torch module to ONNX with a clean named graph.

    Opset 18 avoids version-conversion fallbacks that can fail for some ops
    (Resize in particular). Tries to inline all weights (no external data) so
    TensorRT can parse the model from any working directory; for models near
    or over the ~2GB protobuf limit (e.g. ViT-L/16's encoder) ONNX keeps a
    `.onnx.data` sidecar next to the `.onnx` file regardless -- `build_engine`
    below resolves that correctly via `parse_from_file`.
    """
    import onnx as _onnx

    module.eval()
    path = Path(path)
    torch.onnx.export(
        module,
        sample_inputs,
        str(path),
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
        opset_version=18,
        do_constant_folding=True,
    )
    onnx_model = _onnx.load(str(path), load_external_data=True)
    _onnx.save(onnx_model, str(path), save_as_external_data=False)
    return path


# --------------------------------------------------------------------------
# TensorRT: precision pinning, INT8 calibration, engine build, runner
# --------------------------------------------------------------------------


def pin_sensitive_layers_fp32(network: "trt.INetworkDefinition") -> int:
    """Force RoPE / LayerNorm / softmax / elementwise layers to FP32.

    Fixes the FP16 numerical collapse confirmed on ViT-L/16 (24 blocks):
    residual-add (ELEMENTWISE) accumulation across many transformer blocks
    overflows FP16 range even though the model's own FP32 output magnitude is
    modest. `NORMALIZATION` / `SOFTMAX` and the RoPE subgraph's genuine float
    ops are the other classic FP16-unsafe reduction points. Shape-math layers
    (SHAPE, GATHER, ...) emit INT32/INT64 and must not be forced to float.
    """
    sensitive = {
        trt.LayerType.NORMALIZATION,
        trt.LayerType.SOFTMAX,
        trt.LayerType.ELEMENTWISE,
    }
    int_types = {
        trt.LayerType.SHAPE,
        trt.LayerType.GATHER,
        trt.LayerType.SLICE,
        trt.LayerType.CONCATENATION,
        trt.LayerType.SHUFFLE,
        trt.LayerType.CONSTANT,
        trt.LayerType.UNSQUEEZE,
        trt.LayerType.FILL,
        trt.LayerType.IDENTITY,
        trt.LayerType.CAST,
    }
    pinned = 0
    for i in range(network.num_layers):
        layer = network.get_layer(i)
        is_rope = "rope_embeddings" in layer.name.lower()
        if not (layer.type in sensitive or is_rope):
            continue
        if is_rope and layer.type in int_types:
            continue
        layer.precision = trt.float32
        for j in range(layer.num_outputs):
            if layer.get_output(j).dtype not in (trt.int32, trt.int64, trt.bool):
                layer.set_output_type(j, trt.float32)
        pinned += 1
    return pinned


class _Int8Calibrator(trt.IInt8EntropyCalibrator2):
    """INT8 calibrator. Feeds real data if provided, else random noise.

    Random-noise calibration is what the size/pipeline TRT benchmarks were
    validated with (INT8 mIoU cost was ~0.5-0.9 points across S/S+/B/L), so
    it's the default -- results stay comparable to those numbers. Pass
    `real_batches` (a list of already-device-resident float tensor lists, one
    per calibration step, matching `shapes`) for potentially tighter dynamic
    ranges when representative data is available.
    """

    def __init__(
        self,
        shapes: Sequence[Tuple[int, ...]],
        device: str,
        n_batches: int = 10,
        real_batches: Optional[Sequence[Sequence[torch.Tensor]]] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.shapes = shapes
        self.device = device
        self.real_batches = list(real_batches) if real_batches else None
        self.n = len(self.real_batches) if self.real_batches else n_batches
        self.i = 0
        self.dev = [torch.empty(s, device=device, dtype=torch.float32) for s in shapes]
        # Fixed seed removes *this* source of build-to-build INT8 variance
        # (the calibration input itself). It does not make the resulting
        # engine fully reproducible: TensorRT's own timing-based kernel/
        # tactic autotuning is independent of calibration data and still
        # varies run to run -- confirmed empirically, two builds from the
        # same checkpoint/profile/seed landed at mIoU 0.9080 and 0.9046 (both
        # within the ~0.5-0.9 point INT8 cost band already established across
        # S/S+/B/L). That residual variance is inherent to TensorRT, not
        # something a calibration seed can eliminate.
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(seed)

    def get_batch_size(self):
        return self.shapes[0][0]

    def get_batch(self, names):
        if self.i >= self.n:
            return None
        if self.real_batches:
            for t, src in zip(self.dev, self.real_batches[self.i]):
                t.copy_(src.to(self.device, dtype=torch.float32))
        else:
            for t in self.dev:
                t.normal_(mean=0.0, std=1.0, generator=self.gen)
        self.i += 1
        return [int(t.data_ptr()) for t in self.dev]

    def read_calibration_cache(self):
        return None

    def write_calibration_cache(self, cache):
        return None


def build_engine(
    onnx_path,
    engine_path,
    input_profiles: Dict[str, Tuple],
    precision: Precision,
    device: str = "cuda:0",
    workspace_mib: int = 4096,
    calib_batches: Optional[Sequence[Sequence[torch.Tensor]]] = None,
    n_calib_batches: int = 10,
    calib_seed: int = 42,
) -> Path:
    """Compile an ONNX file into a TensorRT engine.

    input_profiles: {tensor_name: (min_shape, opt_shape, max_shape)}.
    precision: "fp32" | "mixed" | "int8".
    """
    onnx_path, engine_path = Path(onnx_path), Path(engine_path)
    builder = trt.Builder(_TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, _TRT_LOGGER)
    # parse_from_file resolves a *.onnx.data external-weight sidecar relative
    # to the ONNX path; parser.parse(read_bytes()) cannot find it and silently
    # fails for any export using external data. See module docstring, fix (1).
    if not parser.parse_from_file(str(onnx_path)):
        msgs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed for {onnx_path}:\n{msgs}")

    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib << 20)
    profile = builder.create_optimization_profile()
    for name, (mn, opt, mx) in input_profiles.items():
        profile.set_shape(name, mn, opt, mx)
    cfg.add_optimization_profile(profile)

    if precision in ("mixed", "int8"):
        cfg.set_flag(trt.BuilderFlag.FP16)
        cfg.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        pinned = pin_sensitive_layers_fp32(network)
        print(f"    [{engine_path.name}] pinned {pinned} sensitive layers to FP32")
    if precision == "int8":
        cfg.set_flag(trt.BuilderFlag.INT8)
        calib_shapes = [opt for (_, opt, _) in input_profiles.values()]
        try:
            cfg.int8_calibrator = _Int8Calibrator(
                calib_shapes,
                device,
                n_batches=n_calib_batches,
                real_batches=calib_batches,
                seed=calib_seed,
            )
        except Exception as e:
            print(f"    INT8 calibrator unavailable ({e}); falling back to FP16")
            cfg.clear_flag(trt.BuilderFlag.INT8)

    plan = builder.build_serialized_network(network, cfg)
    if plan is None:
        raise RuntimeError(f"engine build returned None for {onnx_path} ({precision})")
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.write_bytes(plan)
    return engine_path


class TRTRunner:
    """Feed torch CUDA tensors in, get a torch CUDA tensor back.

    Executes on a private CUDA stream, synchronized against the caller's
    current stream on both ends. Without this, TensorRT execution can start
    before an async (`non_blocking=True`) H2D copy into the input tensor has
    landed -- CUDA gives no ordering guarantee between independent streams --
    which silently corrupts every downstream result. See module docstring,
    fix (3).
    """

    def __init__(
        self,
        engine_path,
        input_names: Sequence[str],
        output_name: str,
        device: str = "cuda:0",
    ):
        self.device = device
        rt = trt.Runtime(_TRT_LOGGER)
        self.engine = rt.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if self.engine is None:
            raise RuntimeError(
                f"Could not deserialize TensorRT engine: {engine_path}. "
                "Serialized TensorRT engines are specific to compatible "
                "TensorRT/CUDA/GPU environments; rebuild this engine on the "
                "deployment host."
            )
        self.ctx = self.engine.create_execution_context()
        self.input_names = list(input_names)
        self.output_name = output_name
        self.stream = torch.cuda.Stream(device=device)
        self._out: Optional[torch.Tensor] = None

    def __call__(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
        cur = torch.cuda.current_stream(self.device)
        # wait for any pending async copy into `inputs`
        self.stream.wait_stream(cur)
        for name, x in zip(self.input_names, inputs):
            x = x.float().contiguous()
            self.ctx.set_input_shape(name, tuple(x.shape))
            self.ctx.set_tensor_address(name, x.data_ptr())
        out_shape = tuple(self.ctx.get_tensor_shape(self.output_name))
        if self._out is None or tuple(self._out.shape) != out_shape:
            self._out = torch.empty(out_shape, device=self.device, dtype=torch.float32)
        self.ctx.set_tensor_address(self.output_name, self._out.data_ptr())
        with torch.cuda.stream(self.stream):
            self.ctx.execute_async_v3(self.stream.cuda_stream)
        # caller's stream waits for TRT before reading _out
        cur.wait_stream(self.stream)
        return self._out


# --------------------------------------------------------------------------
# High-level orchestration
# --------------------------------------------------------------------------


@dataclass
class ExportResult:
    label: str
    ckpt_dir: str
    image_size: int
    num_classes: int
    manifest: List[dict] = field(default_factory=list)


def export_and_build(
    ckpt_dir,
    out_dir,
    precisions: Sequence[Precision] = PRECISIONS,
    label: Optional[str] = None,
    device: str = "cuda:0",
    min_batch: int = 1,
    opt_batch: int = 16,
    max_batch: int = 16,
    micro_batch: int = 4,
    workspace_mib: int = 4096,
    n_calib_batches: int = 10,
    calib_images: Optional[torch.Tensor] = None,
    calib_seed: int = 42,
) -> ExportResult:
    """Export a HybridSegmenter checkpoint's encoder+decoder to ONNX and build
    FP32 / mixed(FP16) / INT8 TensorRT engines for each, writing everything
    under `out_dir` plus a `manifest.json` in the same schema as
    `models/final/trt_pipeline_sizes/manifest.json`.

    `calib_images`: optional [N,3,H,W] real preprocessed images (N >=
    n_calib_batches * opt_batch) used to seed INT8 calibration with
    representative data instead of random noise; omit to use random noise
    (the setup all size/pipeline TRT benchmarks were validated with).

    `label` defaults to the checkpoint's parent directory name (e.g.
    "dinov3-l_pyrafuse_dice" for a checkpoint at
    ".../dinov3-l_pyrafuse_dice/best").
    """
    ckpt_dir = Path(ckpt_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = label or ckpt_dir.parent.name

    model, img, n_cls = load_checkpoint(ckpt_dir, device=device)
    enc_w = EncoderWrap(model).to(device).eval()
    dec_w = DecoderWrap(model).to(device).eval()

    if not (1 <= min_batch <= opt_batch <= max_batch):
        raise ValueError(
            "Require 1 <= min_batch <= opt_batch <= max_batch; got "
            f"{min_batch}, {opt_batch}, {max_batch}."
        )

    x = torch.randn(micro_batch, 3, img, img, device=device)
    with torch.no_grad():
        feats = enc_w(x)
    C, ph, pw = feats.shape[2], feats.shape[3], feats.shape[4]

    enc_onnx = out_dir / f"{label}_encoder.onnx"
    dec_onnx = out_dir / f"{label}_decoder.onnx"
    # A fixed batch-one deployment profile must export a static graph.  Passing
    # an unconstrained `Dim("B")` for a sample whose batch is one makes the
    # current torch.export path infer a contradictory [2:1] value range.
    # Dynamic batch export remains enabled whenever the TRT profile genuinely
    # has a range.
    enc_dynamic = (
        {"pixel_values": {0: torch.export.Dim("B", min=min_batch, max=max_batch)}}
        if min_batch != max_batch
        else None
    )
    dec_dynamic = (
        {"feats": {0: torch.export.Dim("B", min=min_batch, max=max_batch)}}
        if min_batch != max_batch
        else None
    )
    export_onnx(
        enc_w,
        (x,),
        ["pixel_values"],
        ["feats"],
        enc_onnx,
        dynamic_shapes=enc_dynamic,
    )
    export_onnx(
        dec_w,
        (feats,),
        ["feats"],
        ["logits"],
        dec_onnx,
        dynamic_shapes=dec_dynamic,
    )
    print(f"[{label}] exported ONNX -> {enc_onnx.name}, {dec_onnx.name}")

    enc_prof = {
        "pixel_values": (
            (min_batch, 3, img, img),
            (opt_batch, 3, img, img),
            (max_batch, 3, img, img),
        )
    }
    dec_prof = {
        "feats": (
            (min_batch, 4, C, ph, pw),
            (opt_batch, 4, C, ph, pw),
            (max_batch, 4, C, ph, pw),
        )
    }

    enc_calib = dec_calib = None
    if calib_images is not None:
        n_need = n_calib_batches * opt_batch
        if calib_images.shape[0] < n_need:
            raise ValueError(
                f"calib_images has {calib_images.shape[0]} images, need >= "
                f"{n_need} (n_calib_batches * opt_batch)"
            )
        enc_calib, dec_calib = [], []
        with torch.no_grad():
            for i in range(n_calib_batches):
                chunk = calib_images[i * opt_batch : (i + 1) * opt_batch].to(device)
                enc_calib.append([chunk])
                dec_calib.append([enc_w(chunk)])

    manifest = []
    for prec in precisions:
        t0 = time.perf_counter()
        enc_eng = build_engine(
            enc_onnx,
            out_dir / f"{label}_encoder_{prec}.engine",
            enc_prof,
            prec,
            device=device,
            workspace_mib=workspace_mib,
            calib_batches=enc_calib,
            n_calib_batches=n_calib_batches,
            calib_seed=calib_seed,
        )
        dec_eng = build_engine(
            dec_onnx,
            out_dir / f"{label}_decoder_{prec}.engine",
            dec_prof,
            prec,
            device=device,
            workspace_mib=workspace_mib,
            calib_batches=dec_calib,
            n_calib_batches=n_calib_batches,
            calib_seed=calib_seed,
        )
        build_s = time.perf_counter() - t0
        manifest.append(
            dict(
                label=label,
                precision=prec,
                encoder_onnx=str(enc_onnx),
                decoder_onnx=str(dec_onnx),
                encoder_engine=str(enc_eng),
                decoder_engine=str(dec_eng),
                image_size=img,
                num_classes=n_cls,
                min_batch=min_batch,
                opt_batch=opt_batch,
                max_batch=max_batch,
                build_time_s=round(build_s, 1),
            )
        )
        print(
            f"[{label}/{prec}] built in {build_s:.1f}s -> "
            f"{enc_eng.name}, {dec_eng.name}"
        )

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    del model, enc_w, dec_w, x, feats
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return ExportResult(
        label=label,
        ckpt_dir=str(ckpt_dir),
        image_size=img,
        num_classes=n_cls,
        manifest=manifest,
    )


@torch.no_grad()
def verify_engines(
    ckpt_dir,
    manifest_entry: dict,
    device: str = "cuda:0",
    n_check_batches: int = 3,
    batch: Optional[int] = None,
) -> Dict[str, float]:
    """Cosine-similarity / max-abs sanity check of one manifest entry's TRT
    engines against the eager checkpoint, on random inputs.

    This is a cheap smoke test, not a substitute for a real accuracy
    evaluation on held-out data (see the wrapper notebook for that) -- but it
    is exactly the check that caught both the stream-race bug and L's FP16
    NaN collapse (`has_nan=True`, or `cos` catastrophically low, e.g. <0.5,
    means don't trust this engine's accuracy numbers).

    Caution when gating on `cos` alone: INT8 quantization *inherently*
    produces a lower logit-level cosine similarity than FP32/mixed even on a
    correctly-built engine -- confirmed cos~0.86-0.92 on real images across
    S/S+/B/L in this repo's benchmarks, with real-accuracy cost of well under
    1 mIoU point. Only `has_nan` or a very low `cos` (roughly <0.5, the range
    seen for L's actual pre-fix collapse) indicates a broken engine at int8;
    treat `cos` in the 0.85-0.95 band at int8 as normal, not a failure.

    `batch`: input batch size for the smoke test. Defaults to a size that's
    guaranteed to sit inside the engine's built optimization profile
    (`manifest_entry["min_batch"/"max_batch"]` if present, else falls back to
    4) -- passing a batch outside the profile TensorRT was built with raises
    an opaque low-level error rather than a clear one.
    """
    if batch is None:
        lo = manifest_entry.get("min_batch", 1)
        hi = manifest_entry.get("max_batch", 16)
        batch = max(lo, min(4, hi))
    model, img, _ = load_checkpoint(ckpt_dir, device=device)
    enc_w = EncoderWrap(model).to(device).eval()
    dec_w = DecoderWrap(model).to(device).eval()
    enc_run = TRTRunner(
        manifest_entry["encoder_engine"], ["pixel_values"], "feats", device=device
    )
    dec_run = TRTRunner(
        manifest_entry["decoder_engine"], ["feats"], "logits", device=device
    )

    worst_cos, worst_max_abs, any_nan = 1.0, 0.0, False
    for _ in range(n_check_batches):
        x = torch.randn(batch, 3, img, img, device=device)
        eager_logits = dec_w(enc_w(x))
        trt_logits = dec_run(enc_run(x))
        if torch.isnan(trt_logits).any():
            any_nan = True
            continue
        cos = torch.nn.functional.cosine_similarity(
            eager_logits.flatten().float(), trt_logits.flatten().float(), dim=0
        ).item()
        max_abs = (eager_logits.float() - trt_logits.float()).abs().max().item()
        worst_cos = min(worst_cos, cos)
        worst_max_abs = max(worst_max_abs, max_abs)

    del model, enc_w, dec_w, enc_run, dec_run
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return dict(cos=worst_cos, max_abs=worst_max_abs, has_nan=any_nan)
