import torch
import pytest
from nap_26.config import config, tm_config


def test_cuda_available():
    assert torch.cuda.is_available(), "CUDA not available"


def test_gpu_count():
    count = torch.cuda.device_count()
    assert count > 0, f"Expected at least 1 GPU, found {count}"
    print(f"\nGPUs found: {count}")
    for i in range(count):
        print(f"  [{i}] {torch.cuda.get_device_name(i)}")


def test_device_config():
    device = torch.device(config.device)
    # allocate a small tensor on the configured device
    t = torch.tensor([1.0, 2.0, 3.0]).to(device)
    assert str(t.device).startswith(device.type), f"Tensor not on {device}"
    print(f"\nTensor on: {t.device}")


def test_cuda_version():
    version = torch.version.cuda
    assert version is not None, "PyTorch not built with CUDA"
    print(f"\nPyTorch CUDA version: {version}")
    print(f"PyTorch version:      {torch.__version__}")


def test_tm_config_device_propagated():
    """tm_config.device must match config.device (propagated from top-level YAML)."""
    assert tm_config.device == config.device, (
        f"tm_config.device={tm_config.device!r} != config.device={config.device!r}; "
        "check _tm_yaml.setdefault('device', ...) in config.py"
    )
