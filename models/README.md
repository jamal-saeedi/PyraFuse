# PyraFuse model release guide

The Git repository intentionally contains no pretrained weight binaries. The source-of-truth release should be a Hugging Face **model** repository with one folder per variant:

```text
PyraFuse/
├── README.md                 # Hugging Face model card
├── small/
├── small_plus/
├── base/
└── large/
```

Every variant folder must include `config.json`, `decoder.pt`, `backbone/`, and—when evaluation used it—`ema.pt`. Keep `backbone/model.safetensors`; avoid serialising a whole `nn.Module` with `torch.save`, which is less safe, less portable, and couples consumers to implementation code.

## Publishing

1. Create the model repository, for example `hf repos create <namespace>/PyraFuse`.
2. Add a model card with the task, class mapping, preprocessing, source data, evaluation protocol, known limitations, licence, GitHub commit, and paper citation once published.
3. Upload checkpoint folders resumably with Git LFS through the official CLI:

   ```bash
   hf upload-large-folder <namespace>/PyraFuse <staging-dir>
   ```

   The staging directory must contain the four variant folders at its root. Do not upload TensorRT `.engine` files as the default model release.

4. Create a Hub tag matching the GitHub release:

   ```bash
   hf repos tag create <namespace>/PyraFuse v1.0.0 \
     --message "PyraFuse AIMLSystems 2026 release"
   ```

5. Set `PYRAFUSE_MODEL_REPO=<namespace>/PyraFuse` in user environments.

## TensorRT policy

TensorRT engines are hardware/runtime-specific. If engines are shared, keep them under a separate path such as `tensorrt/<gpu>-trt<version>/<variant>/`, and include a manifest recording GPU, CUDA, TensorRT, ONNX opset, precision, input size, batch profile, build command, and verification result. The eager PyTorch checkpoint remains the portable and archival artifact.
