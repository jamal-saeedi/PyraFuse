# Changelog

All notable changes to PyraFuse are documented here.

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
