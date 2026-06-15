"""Shared helpers for the photo-to-mesh pipeline."""
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


def sam3_repo_override() -> str | None:
    """Ungated mirror for SAM 3 weights, set via the SAM3_HF_REPO env var.

    The official facebook/sam3 weights are gated; users who can't get access can
    point this at an ungated community mirror they trust. Two caveats hold: Meta's
    SAM License still governs use (a mirror does not relicense it), and a .pt is a
    pickle that executes code on load, so the source must be trusted. Returns None
    — the official gated path — when the variable is unset or blank.
    """
    return os.environ.get("SAM3_HF_REPO", "").strip() or None


def build_sam3():
    """Build the SAM 3 image model.

    With SAM3_HF_REPO unset this is exactly the upstream (gated) loader. When set,
    the checkpoint is fetched from that repo and loaded explicitly — see
    sam3_repo_override() for the trust/license caveats. SAM3_CKPT overrides the
    checkpoint filename (default sam3.pt) for mirrors that rename it.
    """
    from sam3.model_builder import build_sam3_image_model

    repo = sam3_repo_override()
    if repo is None:
        return build_sam3_image_model()

    from huggingface_hub import hf_hub_download

    try:  # upstream also fetches config.json alongside the ckpt; non-fatal if absent
        hf_hub_download(repo_id=repo, filename="config.json")
    except Exception:
        pass
    ckpt = hf_hub_download(repo_id=repo, filename=os.environ.get("SAM3_CKPT", "sam3.pt"))
    return build_sam3_image_model(checkpoint_path=ckpt)
