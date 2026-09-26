---
language:
- en
license: other
library_name: pytorch
pipeline_tag: image-segmentation
tags:
- semantic-segmentation
- skin-fabric-detection
- skin-segmentation
- fabric-segmentation
- dinov3
- tensorrt
- executorch
- on-device
- mobile
---

# PyraFuse: skin, fabric, and background segmentation

PyraFuse is a DINOv3-based semantic-segmentation framework for three classes:
`0 = background`, `1 = fabric`, and `2 = skin`. The associated paper has been
accepted at AIMLSystems 2026.

## Available checkpoints

| Variant | Encoder | Folder |
| --- | --- | --- |
| `small` | DINOv3 ViT-S/16 | `small/` |
| `small_plus` | DINOv3 ViT-S+/16 | `small_plus/` |
| `base` | DINOv3 ViT-B/16 | `base/` |
| `large` | DINOv3 ViT-L/16 | `large/` |

Each folder is self-contained: `config.json`, `decoder.pt`, `ema.pt`, and a
`backbone/` directory with the DINOv3 configuration, `model.safetensors`, and
adapter feature-normalisation weights. Use EMA weights for evaluation.

## Usage

Install the [PyraFuse code repository](https://github.com/jamal-saeedi/PyraFuse),
then load any variant. Pin a revision for reproducible experiments.

```python
from pyrafuse import load_pretrained

model = load_pretrained("base", revision="v1.0.0", device="cuda")
```

Inputs must be RGB, ImageNet-normalised, and have height and width divisible by
16. The default checkpoint configuration uses 448 × 448 inputs.

## Mobile and edge: ExecuTorch programs

The `mobile/` folder contains ready-to-run [ExecuTorch](https://executorch.ai)
programs for Android, iOS, React Native, Flutter, and embedded Linux. The same
`.pte` file runs in every framework that embeds the ExecuTorch runtime.

```text
mobile/
├── manifest.json                       # sizes, SHA-256, delegation, held-out mIoU
├── small/pyrafuse_small_<target>.pte
├── small_plus/pyrafuse_small_plus_<target>.pte
└── base/pyrafuse_base_<target>.pte
```

| Target | Runs on | Precision |
| --- | --- | --- |
| `xnnpack_fp32` | CPU: Android, iOS, macOS, Linux/ARM boards | FP32 |
| `xnnpack_int8` | same CPUs, about 4× smaller | INT8 weights, dynamic INT8 activations |
| `coreml_fp32` | iOS 17+ / macOS 14+ GPU and CPU via Core ML | FP32 |
| `vulkan_fp32` | Android GPU via Vulkan | FP32 |

| Variant | Program | Size | mIoU | Agreement with PyTorch |
| --- | --- | ---: | ---: | ---: |
| `small` | PyTorch reference | | 0.907 | |
| | `xnnpack_fp32` | 91 MB | 0.907 | 100.00% |
| | `xnnpack_int8` | 23 MB | 0.906 | 99.78% |
| | `coreml_fp32` | 92 MB | on device | on device |
| | `vulkan_fp32` | 91 MB | on device | on device |
| `small_plus` | PyTorch reference | | 0.908 | |
| | `xnnpack_fp32` | 119 MB | 0.908 | 100.00% |
| | `xnnpack_int8` | 31 MB | 0.907 | 99.74% |
| | `coreml_fp32` | 121 MB | on device | on device |
| | `vulkan_fp32` | 119 MB | on device | on device |
| `base` | PyTorch reference | | 0.912 | |
| | `xnnpack_fp32` | 348 MB | 0.912 | 100.00% |
| | `xnnpack_int8` | 88 MB | 0.912 | 99.80% |
| | `coreml_fp32` | 349 MB | on device | on device |
| | `vulkan_fp32` | 348 MB | on device | on device |

mIoU is measured on the first 200 images of the held-out test split at 448 px by executing each program with the ExecuTorch runtime; the Core ML and Vulkan programs need Apple or Android hardware and are not yet scored.

**Contract.** Input `float32 [1, 3, 448, 448]`: RGB, stretch-resized, divided by
255 (ImageNet normalisation is inside the program). Output `float32 [1, 3, 448, 448]`
logits for `background`, `fabric`, `skin`; take `argmax` over dim 1.

```python
from executorch.runtime import Runtime
from pyrafuse import download_mobile_model  # pip install "pyrafuse[mobile]"

forward = Runtime.get().load_program(
    download_mobile_model("small", "xnnpack_int8", revision="v1.2.0")).load_method("forward")
logits = forward.execute([image])[0]  # image: float32 [1, 3, 448, 448] in [0, 1]
```

React Native (`react-native-executorch` 0.10+) loads these files directly with
`useSemanticSegmenter` using labels `['background', 'fabric', 'skin']`,
`resizeMode: 'stretch'`, and `normalizeOpts: { alpha: 1 / 255, beta: 0 }`.
Copy-paste snippets for React Native, Flutter, Kotlin, Swift, and C++ are in the
[code repository README](https://github.com/jamal-saeedi/PyraFuse#mobile-and-edge-deployment).

There are no FP16 programs: DINOv3's first block produces attention logits up
to about 2.5 × 10⁶, beyond the FP16 range, so FP16 backends (including the Core
ML Neural Engine) output NaNs. Core ML and Vulkan programs are validated at
export but have not yet been benchmarked on devices.

## Release policy

These eager PyTorch checkpoint bundles are the portable source of truth; the
`mobile/` ExecuTorch programs are exported from them with the repository's
`scripts/export_mobile.py`.
TensorRT engines are deliberately not distributed here because they are tied to
the target GPU, CUDA, TensorRT version, precision, and optimization profile.
Build and validate TensorRT engines on the deployment host from the PyTorch
checkpoint using the repository's `scripts/export_trt.py` command.

## Licence

The PyraFuse decoder and repository code are Apache-2.0. Each checkpoint
bundle includes Meta DINOv3 backbone material; use and redistribution of those
bundles are additionally subject to the DINOv3 License. A verbatim copy is
provided at `LICENSES/DINOv3-LICENSE.md` in this model repository. The
underlying Fashionpedia and visuAAL datasets are not redistributed and retain
their own terms.

## Data, limitations, and responsible use

Training labels fuse Fashionpedia annotations with visuAAL skin masks. The
source data is not redistributed in this repository; users must comply with
the source datasets' terms. Model performance can vary by image quality,
lighting, occlusion, garment appearance, and representation in source data.
This model is for research and engineering use, not a medical or biometric
decision system.

## Citation

The accepted manuscript is included in the code repository. Formal citation
metadata will be added after the AIMLSystems 2026 camera-ready publication.
