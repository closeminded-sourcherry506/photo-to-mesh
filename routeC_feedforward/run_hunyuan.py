#!/usr/bin/env python
"""Route C — feed-forward image -> textured mesh with Hunyuan3D-2.1.

Fastest route: one clean, background-removed view of the object -> mesh in
seconds. Geometry on unseen sides is *generated* (plausible, not metric).
Great for assets / 3D print of decorative objects.

Prereq: ./setup.sh routeC   (clones the repo, builds the texture extensions)

Usage:
    # shape + PBR texture:
    python routeC_feedforward/run_hunyuan.py --image data/images/img_0000.png \
        --mask data/masks/img_0000.png --out data/output/mesh_C.glb

    # shape only (skips the texture build; most robust):
    python routeC_feedforward/run_hunyuan.py --image data/images/img_0000.png \
        --mask data/masks/img_0000.png --out data/output/mesh_C.glb --no-texture
"""
from __future__ import annotations
import argparse
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image

# The repo expects its two packages on sys.path (per the official README).
HY = Path(__file__).resolve().parent / "Hunyuan3D-2.1"
sys.path.insert(0, str(HY / "hy3dshape"))
sys.path.insert(0, str(HY / "hy3dpaint"))


def prep_image(image_path: str, mask_path: str | None) -> str:
    """Make a background-removed RGBA PNG (using our SAM 3 mask) and return its path."""
    img = Image.open(image_path).convert("RGB")
    tmp = Path(tempfile.mkdtemp()) / "input_rgba.png"
    if mask_path and Path(mask_path).exists():
        m = Image.open(mask_path).convert("L").resize(img.size, Image.NEAREST)
        rgba = img.convert("RGBA"); rgba.putalpha(m)
        rgba.save(tmp)
    else:
        img.save(tmp)  # Hunyuan has its own background remover as a fallback
    return str(tmp)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True, help="single best view of the object")
    ap.add_argument("--mask", default=None, help="optional SAM 3 mask -> background removal")
    ap.add_argument("--out", default="data/output/mesh_C.glb")
    ap.add_argument("--no-texture", action="store_true", help="shape only (skip the PBR texture stage)")
    ap.add_argument("--views", type=int, default=6, help="texture multiview count")
    ap.add_argument("--res", type=int, default=512, help="texture resolution")
    args = ap.parse_args()

    img_path = prep_image(args.image, args.mask)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)

    import torch
    # Low-RAM machines (this box has 30 GB): load the 7.4 GB fp16 ckpt memory-mapped
    # so it pages in lazily during .to('cuda') instead of doubling in CPU RAM (-> OOM).
    import os as _os
    _orig_load = torch.load
    def _mmap_load(*a, **k):
        f = a[0] if a else k.get("f")
        if isinstance(f, (str, _os.PathLike)):   # mmap only valid for file paths
            k.setdefault("mmap", True)
        return _orig_load(*a, **k)
    torch.load = _mmap_load

    from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

    print("loading Hunyuan3D-2.1 shape model (downloads on first run) ...")
    shape_pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2.1", device="cuda", dtype=torch.float16)
    mesh = shape_pipeline(image=img_path)[0]

    # export the untextured shape first (also the artifact for --no-texture)
    shape_path = str(out.with_suffix(".shape.glb"))
    mesh.export(shape_path)

    if args.no_texture:
        Path(shape_path).rename(out)
        print(f"Mesh (shape only) -> {out}")
        return

    try:
        from textureGenPipeline import Hunyuan3DPaintPipeline, Hunyuan3DPaintConfig
        print("texturing (PBR multiview) ...")
        paint = Hunyuan3DPaintPipeline(Hunyuan3DPaintConfig(max_num_view=args.views, resolution=args.res))
        textured = paint(shape_path, image_path=img_path)   # returns a path or mesh
        if isinstance(textured, (str, Path)):
            shutil.move(str(textured), str(out))   # rename() would fail across filesystems
        else:
            textured.export(out)
        print(f"Mesh (textured) -> {out}")
    except Exception as e:
        Path(shape_path).rename(out)
        print(f"[warn] texture stage failed ({e}); kept shape-only mesh -> {out}")


if __name__ == "__main__":
    main()
