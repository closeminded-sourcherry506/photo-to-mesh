"""Unit handling for mesh export — kept free of torch/gradio so it can be unit-tested.

NOTE: copy of src/mesh_units.py so the HF Space is self-contained — keep in sync.

The rules this module encodes (see README → Scale & resize):
  • glTF/GLB has NO unit field — the spec defines 1 unit = 1 metre, so GLBs
    must always be written in metres or conforming importers read them at the
    wrong size (a "cm GLB" opens 100x too big in Blender/Unity/three.js).
  • STL/OBJ are unitless by convention; slicers assume mm. Those formats are
    where a raw-number unit choice is meaningful.
"""
from __future__ import annotations

# multiply a value given in <unit> by this to get millimetres
TO_MM = {"mm": 1.0, "cm": 10.0, "in": 25.4}
# multiply a mm-sized mesh by this to write raw numbers in <unit>
FROM_MM = {"mm": 1.0, "cm": 0.1, "in": 1.0 / 25.4, "m": 0.001}

AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


def size_to_mm(value: float, unit: str) -> float:
    """Convert a user-entered size to millimetres."""
    return float(value) * TO_MM[unit]


def scale_factor(extents, real_mm: float, axis: str = "tallest") -> float | None:
    """Factor that scales a mesh with `extents` so `axis` measures `real_mm`.

    Returns None when no positive size was given, or the reference extent is
    degenerate — i.e. there is nothing meaningful to scale by.
    """
    if not real_mm or real_mm <= 0:
        return None
    ref = float(max(extents)) if axis == "tallest" else float(extents[AXIS_INDEX[axis]])
    if ref <= 0:
        return None
    return float(real_mm) / ref


def echo_units(mm: float) -> str:
    """One length spelled out in mm / cm / m, so unit mix-ups are visible at a glance."""
    return f"{mm:.1f} mm = {mm / 10.0:.2f} cm = {mm / 1000.0:.3f} m"
