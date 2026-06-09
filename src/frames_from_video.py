#!/usr/bin/env python
"""Extract sharp, well-spaced frames from a video (optional convenience step).

Usage:
    python src/frames_from_video.py input.mp4 data/images --fps 2 --max 60

Picks frames at a target fps, then keeps only the sharpest within each small
window so you don't feed motion-blurred frames into VGGT/COLMAP.
"""
from __future__ import annotations
import argparse
import subprocess
import tempfile
from pathlib import Path

import cv2  # type: ignore


def sharpness(img) -> float:
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("out_dir")
    ap.add_argument("--fps", type=float, default=2.0, help="sampling rate before sharpness filtering")
    ap.add_argument("--max", type=int, default=60, help="max frames to keep")
    ap.add_argument("--window", type=int, default=3, help="keep sharpest 1 of every N sampled frames")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["ffmpeg", "-y", "-i", args.video, "-vf", f"fps={args.fps}",
             "-qscale:v", "2", f"{tmp}/f_%05d.jpg"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        frames = sorted(Path(tmp).glob("f_*.jpg"))
        # keep sharpest within each window
        kept = []
        for i in range(0, len(frames), args.window):
            chunk = frames[i:i + args.window]
            best = max(chunk, key=lambda p: sharpness(cv2.imread(str(p))))
            kept.append(best)
        # cap to --max, evenly spaced
        if len(kept) > args.max:
            step = len(kept) / args.max
            kept = [kept[int(i * step)] for i in range(args.max)]
        for j, p in enumerate(kept):
            cv2.imwrite(str(out / f"img_{j:04d}.jpg"), cv2.imread(str(p)))

    print(f"Wrote {len(kept)} frames -> {out}")


if __name__ == "__main__":
    main()
