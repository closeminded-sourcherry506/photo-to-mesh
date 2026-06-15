"""Unit tests for src/common.py — the dependency-free helpers (run in CI without torch)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import ensure_dir, list_images, sam3_repo_override  # noqa: E402


def test_list_images_sorted_absolute_and_filtered(tmp_path):
    for name in ("b.jpg", "a.png", "c.jpeg", "notes.txt", "mesh.obj"):
        (tmp_path / name).touch()
    paths = list_images(tmp_path)
    assert [Path(p).name for p in paths] == ["a.png", "b.jpg", "c.jpeg"]
    assert all(Path(p).is_absolute() for p in paths)


def test_list_images_empty_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list_images(tmp_path)


def test_ensure_dir_creates_nested_and_is_idempotent(tmp_path):
    target = tmp_path / "x" / "y" / "z"
    assert ensure_dir(target) == target
    assert target.is_dir()
    assert ensure_dir(target) == target  # second call must not raise


def test_sam3_repo_override_defaults_to_official(monkeypatch):
    monkeypatch.delenv("SAM3_HF_REPO", raising=False)
    assert sam3_repo_override() is None
    monkeypatch.setenv("SAM3_HF_REPO", "   ")   # blank/whitespace = unset
    assert sam3_repo_override() is None


def test_sam3_repo_override_reads_and_trims_env(monkeypatch):
    monkeypatch.setenv("SAM3_HF_REPO", "  1038lab/sam3  ")
    assert sam3_repo_override() == "1038lab/sam3"
