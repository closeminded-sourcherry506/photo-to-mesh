"""Shared helpers for the object-scan pipeline."""
from __future__ import annotations
import os
import glob
from pathlib import Path

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG", ".webp")


def list_images(folder: str | os.PathLike) -> list[str]:
    """Return sorted absolute image paths in a folder."""
    folder = str(folder)
    paths: list[str] = []
    for ext in IMAGE_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
    paths = sorted(set(os.path.abspath(p) for p in paths))
    if not paths:
        raise FileNotFoundError(f"No images found in {folder!r} (exts={IMAGE_EXTS})")
    return paths


def pick_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def autocast_dtype():
    """bf16 on Ampere+ (incl. Blackwell), else fp16."""
    import torch
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8:
        return torch.bfloat16
    return torch.float16


def ensure_dir(p: str | os.PathLike) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p
