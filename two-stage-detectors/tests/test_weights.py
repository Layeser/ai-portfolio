"""The download path verifies SHA-256 before installing weights (remote storage is simulated)."""

import importlib.util
import json
import shutil
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "weights.py"


def load(tmp_path, backend):
    (tmp_path / "scripts").mkdir(exist_ok=True)
    shutil.copy(SCRIPT, tmp_path / "scripts" / "weights.py")
    (tmp_path / "artifacts" / "sub").mkdir(parents=True, exist_ok=True)
    (tmp_path / "artifacts" / "sub" / "model.bin").write_bytes(b"trained weights")
    config = {"backend": backend, "github": {"repo": "me/repo", "tag": "v1"}, "repo_id": "me/model",
              "files": {"sub/model.bin": "artifacts/sub/model.bin"}}
    (tmp_path / "weights.json").write_text(json.dumps(config))
    spec = importlib.util.spec_from_file_location(f"weights_{backend}", tmp_path / "scripts" / "weights.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("backend", ["github", "huggingface"])
def test_export_then_download_restores_verified_files(tmp_path, monkeypatch, backend):
    weights = load(tmp_path, backend)
    weights.export()
    released = tmp_path / "release" / weights.asset_name("sub/model.bin")
    assert released.exists() and ("/" not in released.name)
    (tmp_path / "artifacts" / "sub" / "model.bin").unlink()
    monkeypatch.setattr(weights, "fetch", lambda name: released)
    weights.download()
    assert (tmp_path / "artifacts" / "sub" / "model.bin").read_bytes() == b"trained weights"


def test_download_refuses_tampered_files(tmp_path, monkeypatch):
    weights = load(tmp_path, "github")
    weights.export()
    released = tmp_path / "release" / "sub__model.bin"
    released.write_bytes(b"tampered")
    monkeypatch.setattr(weights, "fetch", lambda name: released)
    with pytest.raises(SystemExit, match="checksum mismatch"):
        weights.download()
