#!/usr/bin/env python
"""Object masking with SAM 3 (shared 'object-only' step).

SAM 3 (Meta, Nov 2025) takes a *concept* prompt — a short noun phrase like
"the mug" or "a yellow toy" — and segments every matching instance, so you
usually don't need to click anything. We take the highest-scoring detection
per image (or the union of all detections with --union) as the object mask.

Outputs a binary mask PNG per image into --out (white = object).

Auth: SAM 3 checkpoints are gated on Hugging Face — run `huggingface-cli login`
once and request access on the SAM 3 model page.

Usage:
    python src/mask_sam3.py --images data/images --out data/masks --prompt "the mug"
    python src/mask_sam3.py --images data/images --out data/masks --prompt "toy" --union
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_sam3, ensure_dir, list_images  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/images")
    ap.add_argument("--out", default="data/masks")
    ap.add_argument("--prompt", required=True, help='concept to segment, e.g. "the mug"')
    ap.add_argument("--union", action="store_true",
                    help="union all matching instances instead of the top-scoring one")
    ap.add_argument("--min-score", type=float, default=0.3)
    args = ap.parse_args()

    import torch
    from sam3.model.sam3_image_processor import Sam3Processor

    processor = Sam3Processor(build_sam3())

    paths = list_images(args.images)
    out = ensure_dir(args.out)

    # SAM 3 internally casts features to bf16 expecting AMP to be active, so the
    # whole forward must run under a CUDA bf16 autocast context (otherwise the
    # bf16 inputs hit fp32 weights -> dtype mismatch).
    amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if torch.cuda.is_available() \
        else torch.autocast(device_type="cpu", dtype=torch.bfloat16)

    for p in paths:
        image = Image.open(p).convert("RGB")
        w, h = image.size
        with torch.inference_mode(), amp:
            state = processor.set_image(image)
            res = processor.set_text_prompt(state=state, prompt=args.prompt)

        def to_np(x):
            return x.float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
        scores = to_np(res["scores"]).reshape(-1)
        masks = to_np(res["masks"])                                # (N,H,W) bool/float
        if masks.ndim == 4:                                        # (N,1,H,W) -> (N,H,W)
            masks = masks[:, 0]
        good = scores >= args.min_score
        if not good.any():
            # nothing matched confidently -> empty mask (skip frame in downstream filters)
            mask = np.zeros((h, w), np.uint8)
        elif args.union:
            mask = (masks[good] > 0.5).any(0).astype(np.uint8) * 255
        else:
            best = int(np.argmax(np.where(good, scores, -1)))
            mask = (masks[best] > 0.5).astype(np.uint8) * 255

        Image.fromarray(mask).resize((w, h), Image.NEAREST).save(out / (Path(p).stem + ".png"))

    print(f'Wrote {len(paths)} masks for prompt "{args.prompt}" -> {out}')


if __name__ == "__main__":
    main()
