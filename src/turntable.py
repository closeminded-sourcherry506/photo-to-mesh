#!/usr/bin/env python
"""Headless turntable renderer for any mesh (.glb/.obj/.ply) -> GIF + MP4.

Decimates with Open3D (fast), renders an orbiting camera with matplotlib
(no display / GPU needed), then assembles GIF + MP4 via ffmpeg.

Usage:
    python src/turntable.py data/output/mesh_C.glb --out data/output/mesh_C_turntable --frames 48
"""
from __future__ import annotations
import argparse
import shutil
import subprocess
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def load_decimated(path: str, target_faces: int):
    tm = trimesh.load(path, force="mesh")
    m = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(tm.vertices)),
        o3d.utility.Vector3iVector(np.asarray(tm.faces)),
    )
    if len(tm.faces) > target_faces:
        m = m.simplify_quadric_decimation(target_faces)
    m.compute_vertex_normals()
    V = np.asarray(m.vertices)
    F = np.asarray(m.triangles)
    return V, F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mesh")
    ap.add_argument("--out", default="data/output/turntable")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--faces", type=int, default=8000, help="decimate target for fast rendering")
    ap.add_argument("--elev", type=float, default=15.0)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--color", default="0.45,0.6,0.95", help="base RGB 0-1")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found on PATH — install it first (e.g. apt install ffmpeg).")
    V, F = load_decimated(args.mesh, args.faces)

    # center, scale to unit, and stand the longest axis up (-> plot Z)
    V = V - V.mean(0)
    V = V / np.abs(V).max()
    up_axis = int(np.argmax(V.max(0) - V.min(0)))
    order = [a for a in range(3) if a != up_axis] + [up_axis]   # longest axis last == Z(up)
    V = V[:, order]

    tris = V[F]
    # flat shading from face normals against a fixed light
    e1 = tris[:, 1] - tris[:, 0]
    e2 = tris[:, 2] - tris[:, 0]
    n = np.cross(e1, e2)
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    light = np.array([0.4, 0.4, 0.9]); light /= np.linalg.norm(light)
    shade = np.clip(n @ light, 0.2, 1.0)[:, None]
    base = np.array([float(x) for x in args.color.split(",")])
    colors = np.clip(shade * base, 0, 1)

    frames_dir = Path(args.out + "_frames")
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")
    pc = Poly3DCollection(tris, facecolors=colors, edgecolors="none")
    ax.add_collection3d(pc)
    lim = 0.85
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(-lim, lim)
    ax.set_box_aspect((1, 1, 1)); ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    for i in range(args.frames):
        az = 360.0 * i / args.frames
        ax.view_init(elev=args.elev, azim=az)
        fig.savefig(frames_dir / f"frame_{i:03d}.png", dpi=120, facecolor="white")
    plt.close(fig)
    print(f"rendered {args.frames} frames -> {frames_dir}")

    mp4 = args.out + ".mp4"
    gif = args.out + ".gif"
    pat = str(frames_dir / "frame_%03d.png")
    # MP4 (H.264, widely playable)
    subprocess.run(["ffmpeg", "-y", "-framerate", str(args.fps), "-i", pat,
                    "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", mp4],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # GIF via two-pass palette for clean colors
    palette = str(frames_dir / "palette.png")
    subprocess.run(["ffmpeg", "-y", "-i", pat, "-vf", "palettegen", palette],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["ffmpeg", "-y", "-framerate", str(args.fps), "-i", pat, "-i", palette,
                    "-lavfi", "paletteuse", gif],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"wrote {mp4}\nwrote {gif}")


if __name__ == "__main__":
    main()
