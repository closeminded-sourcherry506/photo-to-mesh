"""Regression tests for unit handling — including the reported "selected cm,
model came out 10x too big" bug (GLB written with cm numbers, which conforming
importers read as metres)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mesh_units import FROM_MM, TO_MM, echo_units, scale_factor, size_to_mm  # noqa: E402


def test_size_to_mm_accepts_user_units():
    assert size_to_mm(250, "mm") == 250.0
    assert size_to_mm(25, "cm") == 250.0
    assert size_to_mm(10, "in") == 254.0


def test_scale_factor_tallest_axis():
    # normalized Hunyuan output: longest axis ~2 units; 25 cm object -> x125
    assert scale_factor((1.0, 2.0, 0.8), size_to_mm(25, "cm")) == pytest.approx(125.0)


def test_scale_factor_explicit_axis():
    assert scale_factor((1.0, 2.0, 0.8), 100.0, axis="X") == pytest.approx(100.0)
    assert scale_factor((1.0, 2.0, 0.8), 100.0, axis="Z") == pytest.approx(125.0)


def test_scale_factor_none_when_unset_or_degenerate():
    assert scale_factor((1.0, 2.0, 1.0), 0.0) is None
    assert scale_factor((1.0, 2.0, 1.0), -5.0) is None
    assert scale_factor((0.0, 0.0, 0.0), 100.0) is None


def test_regression_25cm_object_is_0_25_in_the_metres_glb():
    """The reported bug: a 25 cm object must land at 0.25 glTF units (= metres),
    regardless of which unit the user typed the size in."""
    extents = (0.9, 2.0, 0.7)
    for value, unit in ((25, "cm"), (250, "mm")):
        f = scale_factor(extents, size_to_mm(value, unit))     # mesh -> mm
        tallest_m = max(extents) * f * FROM_MM["m"]            # mm -> metres GLB
        assert tallest_m == pytest.approx(0.25)


def test_to_mm_and_from_mm_round_trip():
    for unit in TO_MM:
        assert TO_MM[unit] * FROM_MM[unit] == pytest.approx(1.0)


def test_echo_units_spells_out_all_three():
    assert echo_units(250.0) == "250.0 mm = 25.00 cm = 0.250 m"
