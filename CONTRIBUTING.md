# Contributing to PyraFuse

Thanks for helping improve PyraFuse. Bug reports, documentation improvements,
reproducible examples, and focused pull requests are welcome.

## Before opening an issue

- Search existing issues and include a minimal, reproducible example.
- Report the PyraFuse version, Python version, PyTorch version, operating
  system, GPU/CUDA/TensorRT versions when relevant, and the model variant.
- Do not attach private images, checkpoints, access tokens, or licensed source
  datasets.

## Development setup

```bash
git clone https://github.com/jamal-saeedi/PyraFuse.git
cd PyraFuse
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
pytest -q
ruff check pyrafuse scripts tests
```

Keep changes focused, add or update tests for behavioural changes, and explain
the motivation and validation in the pull request. The CI workflow must pass
before a change can be merged.

## Models and data

Do not commit model weights, TensorRT engines, raw datasets, credentials, or
experiment logs. See the [model release guide](models/README.md),
[NOTICE](NOTICE), and [licence](LICENSE) for redistribution requirements.
