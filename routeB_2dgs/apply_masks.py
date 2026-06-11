#!/usr/bin/env python
"""Route B helper: black out the background in a VGGT scene's images using
SAM 3 masks, so 2DGS concentrates Gaussians on the object.

Writes masked copies in-place over <scene>/images (keeps a one-time backup in
<scene>/images_raw). Idempotent: re-runs from the backup.

Usage: python routeB_2dgs/apply_masks.py --scene data/output/scene --masks data/masks
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

MASK_EXTS = (".png", ".jpg")


def find_mask(masks: Path, stem: str, index: int, ordered: list[Path]) -> Path | None:
    """Mask for scene image `stem` (e.g. img_0003): try the same stem first, then
    fall back to position — pose_vggt renames scene images to img_%04d.png while
    masks keep the ORIGINAL photo stems, but both follow the same sorted order."""
    for ext in MASK_EXTS:
        cand = masks / f"{stem}{ext}"
        if cand.exists():
            return cand
    if 0 <= index < len(ordered):
        return ordered[index]
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="data/output/scene")
    ap.add_argument("--masks", default="data/masks")
    ap.add_argument("--bg", choices=["black", "white"], default="black")
    args = ap.parse_args()

    img_dir = Path(args.scene) / "images"
    raw_dir = Path(args.scene) / "images_raw"
    if not raw_dir.exists():
        shutil.copytree(img_dir, raw_dir)

    masks = Path(args.masks)
    images = sorted(raw_dir.iterdir())
    ordered = sorted(p for p in masks.iterdir() if p.suffix.lower() in MASK_EXTS) if masks.is_dir() else []
    # positional pairing is only sound when the sets line up 1:1
    by_index = ordered if len(ordered) == len(images) else []

    fill = 0 if args.bg == "black" else 255
    n = 0
    for i, p in enumerate(images):
        img = np.array(Image.open(p).convert("RGB"))
        h, w = img.shape[:2]
        mp = find_mask(masks, p.stem, i, by_index)
        if mp is not None:
            m = np.array(Image.open(mp).convert("L").resize((w, h), Image.NEAREST)) > 127
            img[~m] = fill
            n += 1
        Image.fromarray(img).save(img_dir / p.name)
    print(f"Masked {n}/{len(images)} images (bg={args.bg}) -> {img_dir}")
    if n == 0:
        print("[warn] no masks matched — check --masks contents (expected one mask per input photo)")


if __name__ == "__main__":
    main()
