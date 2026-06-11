"""Regression tests for the Route B mask matching — scene images are renamed to
img_%04d.png by pose_vggt while masks keep the original photo stems, so pure
stem matching silently applied zero masks for normal phone-photo filenames."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "routeB_2dgs"))
from apply_masks import find_mask  # noqa: E402


def test_stem_match_wins_when_names_align(tmp_path):
    (tmp_path / "img_0001.png").touch()
    (tmp_path / "img_0002.png").touch()
    ordered = sorted(tmp_path.iterdir())
    assert find_mask(tmp_path, "img_0001", 0, ordered) == tmp_path / "img_0001.png"


def test_positional_fallback_for_renamed_scene_images(tmp_path):
    # masks named after the original photos, in the same sorted order pose_vggt used
    for name in ("IMG_5301.png", "IMG_5304.png", "IMG_5310.png"):
        (tmp_path / name).touch()
    ordered = sorted(tmp_path.iterdir())
    assert find_mask(tmp_path, "img_0000", 0, ordered) == tmp_path / "IMG_5301.png"
    assert find_mask(tmp_path, "img_0002", 2, ordered) == tmp_path / "IMG_5310.png"


def test_no_match_returns_none(tmp_path):
    # empty ordered list = counts didn't line up, positional pairing disabled
    assert find_mask(tmp_path, "img_0000", 0, []) is None
