#!/usr/bin/env python
"""VGGT pose backend — feed-forward camera poses + intrinsics + dense point map.

Replaces COLMAP. Given a folder of images, runs VGGT-1B once and writes a
COLMAP-format scene that both Route A (meshing) and Route B (2DGS) consume:

    <scene>/
        images/            processed RGB at the intrinsics' resolution
        sparse/0/
            cameras.txt    PINHOLE intrinsics per image
            images.txt     world->camera pose (qw qx qy qz tx ty tz), COLMAP/OpenCV convention
            points3D.txt   confident, (optionally mask-filtered) fused world points + RGB
        points.ply         same fused point cloud, for quick viewing / Route A

VGGT extrinsics are world->camera in the OpenCV convention, which is exactly
COLMAP's convention, so no axis flips are needed.

Usage:
    python src/pose_vggt.py --images data/images --masks data/masks --scene data/output/scene
    python src/pose_vggt.py --images data/images --scene data/output/scene --conf 50
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import list_images, pick_device, autocast_dtype, ensure_dir  # noqa: E402


def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
    """3x3 rotation -> COLMAP quaternion [qw, qx, qy, qz]."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0
    vals, vecs = np.linalg.eigh(K)
    q = vecs[[3, 0, 1, 2], np.argmax(vals)]
    if q[0] < 0:
        q = -q
    return q


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--masks", default=None, help="optional SAM2 mask folder -> object-only points")
    ap.add_argument("--scene", default="data/output/scene")
    ap.add_argument("--conf", type=float, default=50.0,
                    help="keep points above this confidence percentile (0-100)")
    ap.add_argument("--max-points", type=int, default=500_000)
    args = ap.parse_args()

    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    device = pick_device()
    paths = list_images(args.images)
    print(f"VGGT on {len(paths)} images ({device}) ...")

    model = VGGT.from_pretrained("facebook/VGGT-1B").to(device).eval()
    images = load_and_preprocess_images(paths).to(device)          # (S,3,H,W)
    with torch.no_grad(), torch.autocast(device, dtype=autocast_dtype()):
        preds = model(images[None])                                # add batch dim

    H, W = images.shape[-2:]
    extr, intr = pose_encoding_to_extri_intri(preds["pose_enc"], (H, W))
    extr = extr[0].float().cpu().numpy()                           # (S,3,4) world->cam
    intr = intr[0].float().cpu().numpy()                           # (S,3,3)
    world = preds["world_points"][0].float().cpu().numpy()         # (S,H,W,3)
    conf = preds["world_points_conf"][0].float().cpu().numpy()     # (S,H,W)
    rgb = (images.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)  # (S,H,W,3)

    scene = ensure_dir(args.scene)
    sparse = ensure_dir(scene / "sparse" / "0")
    img_out = ensure_dir(scene / "images")

    # --- write processed images (so intrinsics match exactly) ---
    names = [f"img_{i:04d}.png" for i in range(len(paths))]
    for i, name in enumerate(names):
        Image.fromarray(rgb[i]).save(img_out / name)

    # --- optional masks, resized to processed resolution ---
    masks = None
    if args.masks:
        masks = np.zeros((len(paths), H, W), bool)
        for i, p in enumerate(paths):
            mp = Path(args.masks) / (Path(p).stem + ".png")
            if mp.exists():
                m = np.array(Image.open(mp).convert("L").resize((W, H), Image.NEAREST))
                masks[i] = m > 127

    # --- cameras.txt (PINHOLE) + images.txt ---
    with open(sparse / "cameras.txt", "w") as fc, open(sparse / "images.txt", "w") as fi:
        fc.write("# Camera list\n")
        fi.write("# Image list\n")
        for i, name in enumerate(names):
            fx, fy, cx, cy = intr[i, 0, 0], intr[i, 1, 1], intr[i, 0, 2], intr[i, 1, 2]
            cam_id = i + 1
            fc.write(f"{cam_id} PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")
            R, t = extr[i, :, :3], extr[i, :, 3]
            qw, qx, qy, qz = rotmat_to_qvec(R)
            fi.write(f"{i+1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {cam_id} {name}\n\n")

    # --- fused point cloud (confidence + mask filtered) ---
    thr = np.percentile(conf, args.conf)
    keep = conf >= thr
    if masks is not None:
        keep &= masks
    pts = world[keep]
    cols = rgb[keep]
    if len(pts) > args.max_points:
        sel = np.random.default_rng(0).choice(len(pts), args.max_points, replace=False)
        pts, cols = pts[sel], cols[sel]

    _write_ply(scene / "points.ply", pts, cols)
    with open(sparse / "points3D.txt", "w") as fp:
        fp.write("# 3D point list\n")
        for j, (p, c) in enumerate(zip(pts, cols)):
            fp.write(f"{j+1} {p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]} 0\n")

    print(f"Scene written -> {scene}")
    print(f"  images: {len(names)}  |  fused points: {len(pts)}  (conf>={args.conf}pct"
          + (", masked)" if masks is not None else ")"))


def _write_ply(path, pts, cols):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(pts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for p, c in zip(pts, cols):
            f.write(f"{p[0]} {p[1]} {p[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")


if __name__ == "__main__":
    main()
