# PyraFuse model release guide

The Git repository intentionally contains no pretrained weight binaries. The source-of-truth release is the public Hugging Face [jamal-one/PyraFuse](https://huggingface.co/jamal-one/PyraFuse) model repository, with one folder per variant:

```text
PyraFuse/
├── README.md                 # Hugging Face model card
├── small/
├── small_plus/
├── base/
├── large/
└── mobile/                   # ExecuTorch .pte programs + manifest.json
```

Every variant folder must include `config.json`, `decoder.pt`, `backbone/`, and—when evaluation used it—`ema.pt`. Keep `backbone/model.safetensors`; avoid serialising a whole `nn.Module` with `torch.save`, which is less safe, less portable, and couples consumers to implementation code.

## Publishing

1. Add a model card with the task, class mapping, preprocessing, source data, evaluation protocol, known limitations, licence, GitHub commit, and paper citation once published.
3. Upload checkpoint folders resumably with Git LFS through the official CLI:

   ```bash
   hf upload jamal-one/PyraFuse <staging-dir> .
   ```

   The staging directory must contain the four variant folders at its root. Do not upload TensorRT `.engine` files as the default model release.

4. Create a Hub tag matching the GitHub release:

   ```bash
   hf repos tag create jamal-one/PyraFuse v1.0.0 \
     --message "PyraFuse AIMLSystems 2026 release"
   ```

5. `load_pretrained` uses `jamal-one/PyraFuse` by default. Set `PYRAFUSE_MODEL_REPO` only to use a mirror or fork.

## TensorRT policy

TensorRT engines are hardware/runtime-specific. If engines are shared, keep them under a separate path such as `tensorrt/<gpu>-trt<version>/<variant>/`, and include a manifest recording GPU, CUDA, TensorRT, ONNX opset, precision, input size, batch profile, build command, and verification result. The eager PyTorch checkpoint remains the portable and archival artifact.

## ExecuTorch (mobile/edge) policy

Unlike TensorRT engines, ExecuTorch `.pte` programs are portable across devices
for a given backend, so they are published in the model repository under
`mobile/<variant>/pyrafuse_<variant>_<target>.pte`. Build and upload them with:

```bash
python scripts/export_mobile.py --variants small small_plus base --eval
hf upload jamal-one/PyraFuse models/mobile mobile
```

`models/mobile/manifest.json` is tracked in Git and uploaded alongside the
programs; it records the ExecuTorch/torch versions, I/O contract, sizes,
SHA-256 hashes, backend delegation, and held-out mIoU. Re-export when the
PyTorch weights change, and keep the programs' ExecuTorch version compatible
with the runtimes pinned by the consuming apps.
