from pathlib import Path

import pytest

from pyrafuse.zoo import (
    MODEL_VARIANTS,
    OFFICIAL_MODEL_REPO,
    ModelSpec,
    available_models,
    resolve_checkpoint,
)


def test_available_models_are_complete_and_ordered():
    assert [spec.name for spec in available_models()] == [
        "small", "small_plus", "base", "large"
    ]
    assert set(MODEL_VARIANTS) == {"small", "small_plus", "base", "large"}


def test_resolve_checkpoint_accepts_explicit_local_path(tmp_path: Path):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "decoder.pt").write_bytes(b"weights")
    assert resolve_checkpoint(tmp_path) == tmp_path.resolve()


def test_resolve_checkpoint_uses_official_repository_when_no_local_weights(monkeypatch):
    monkeypatch.delenv("PYRAFUSE_MODEL_REPO", raising=False)
    monkeypatch.setitem(
        MODEL_VARIANTS,
        "base",
        ModelSpec("base", "test", "a/checkpoint/that/does/not/exist"),
    )
    def fake_download(**kwargs):
        assert kwargs["repo_id"] == OFFICIAL_MODEL_REPO
        return "/not/a/complete/checkpoint"

    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_download)
    with pytest.raises(FileNotFoundError, match="does not contain"):
        resolve_checkpoint("base")


def test_mobile_model_path_matches_hub_layout():
    from pyrafuse.zoo import mobile_model_path

    assert (mobile_model_path("small", "xnnpack_int8")
            == "mobile/small/pyrafuse_small_xnnpack_int8.pte")
    with pytest.raises(ValueError, match="No mobile build"):
        mobile_model_path("large", "xnnpack_fp32")
    with pytest.raises(ValueError, match="Unknown mobile target"):
        mobile_model_path("small", "coreml_fp16")


def test_mobile_targets_match_exporter():
    pytest.importorskip("executorch")
    from pyrafuse.deploy.mobile_export import MOBILE_TARGETS
    from pyrafuse.zoo import MOBILE_TARGET_NAMES

    assert tuple(MOBILE_TARGETS) == MOBILE_TARGET_NAMES
