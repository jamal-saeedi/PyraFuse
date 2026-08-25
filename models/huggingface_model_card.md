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

## Release policy

These eager PyTorch checkpoint bundles are the portable source of truth.
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
