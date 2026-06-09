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
    fill = 0 if args.bg == "black" else 255
    n = 0
    for p in sorted(raw_dir.iterdir()):
        img = np.array(Image.open(p).convert("RGB"))
        h, w = img.shape[:2]
        # masks were computed on the ORIGINAL images; match by stem, resize to processed size
        mp = next((masks / f"{p.stem}{e}" for e in (".png", ".jpg") if (masks / f"{p.stem}{e}").exists()), None)
        if mp is None:
            # also try the original (pre-VGGT) stem mapping is 1:1 by index name img_XXXX
            mp = masks / f"{p.stem}.png"
        if mp.exists():
            m = np.array(Image.open(mp).convert("L").resize((w, h), Image.NEAREST)) > 127
            img[~m] = fill
            n += 1
        Image.fromarray(img).save(img_dir / p.name)
    print(f"Masked {n} images (bg={args.bg}) -> {img_dir}")


if __name__ == "__main__":
    main()
