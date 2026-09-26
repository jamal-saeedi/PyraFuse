# Changelog

All notable changes to PyraFuse are documented here.

## [1.2.0] - 2026-09-26

- Added ExecuTorch mobile and edge deployment: `pyrafuse.deploy.mobile_export`,
  the `scripts/export_mobile.py` CLI, and the `mobile` extra.
- Published `.pte` programs for `small`, `small_plus`, and `base` on Hugging Face
  (`mobile/`): XNNPACK FP32 and INT8 (CPU), Core ML FP32 (iOS/macOS), and
  Vulkan FP32 (Android GPU), with a manifest of sizes, hashes, and held-out mIoU.
- Added `download_mobile_model`, `MOBILE_VARIANTS`, and `MOBILE_TARGET_NAMES`.
- Added `notebooks/mobile_executorch.ipynb` and README usage for React Native,
  Flutter, Android, iOS, Python, and C++.
- `pyrafuse.deploy` now imports its TensorRT and ExecuTorch helpers lazily.
- Releases are now automated: pushing a `vX.Y.Z` tag verifies the version,
  publishes to PyPI through Trusted Publishing, and creates the GitHub Release.
- CI lints the deployment modules and exporter CLI.

## [1.1.1] - 2026-08-26

- Simplified the published PyPI documentation and installation guidance.
- Added the training section to the README's top navigation.
- Refreshed the PyPI badge to display the current published version reliably.

## [1.1.0] - 2026-08-26

- Added the configurable `scripts/train_segmenter.py` training CLI.
- Added reproducible split, class-weight estimation, smoke-test, dry-run, EMA,
  checkpoint, and metrics options to the training workflow.
- Added README training examples for CPU smoke tests, checkpoint fine-tuning,
  and fresh DINOv3-S training.
- Hardened the CLI's CUDA environment setup and validation, including safe
  device defaults and bounded training hyper-parameters.

## [1.0.0] - 2026-08-25

- First public research release accompanying the AIMLSystems 2026 accepted manuscript.
- Added PyTorch model-zoo loading for local checkpoints and the public Hugging Face release.
- Added reproducible TensorRT export manifests and deployment guidance.
- Added accepted manuscript, qualitative figures, data preparation, and inference notebooks.
- Released original PyraFuse code under Apache-2.0; documented third-party model and data terms.
