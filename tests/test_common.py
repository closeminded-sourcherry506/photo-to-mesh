"""Unit tests for src/common.py — the dependency-free helpers (run in CI without torch)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import list_images, ensure_dir  # noqa: E402


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
