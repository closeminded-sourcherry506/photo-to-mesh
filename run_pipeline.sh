#!/usr/bin/env bash
# End-to-end driver.  Put photos in data/images/ first (or pass a video).
#
#   run_pipeline.sh --prompt "the mug"                 # A + B from data/images
#   run_pipeline.sh --prompt "the mug" --video clip.mp4
#   run_pipeline.sh --prompt "the mug" --routes A      # just Route A
#   run_pipeline.sh --prompt "the mug" --routes C --image data/images/img_0000.png
#
# Routes: A = VGGT point map -> Open3D mesh (fast, editable)
#         B = VGGT poses -> 2D Gaussian Splatting -> TSDF mesh (best geometry; needs setup.sh routeB)
#         C = Hunyuan3D feed-forward from one image (fastest; needs setup.sh routeC)
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
[ -f .venv/bin/activate ] || { echo "error: .venv missing — run ./setup.sh core first" >&2; exit 1; }
# shellcheck disable=SC1091
source .venv/bin/activate

PROMPT=""; VIDEO=""; ROUTES="AB"; IMAGE=""; SCENE="data/output/scene"
while [ $# -gt 0 ]; do case "$1" in
  --prompt) PROMPT="$2"; shift 2;;
  --video)  VIDEO="$2"; shift 2;;
  --routes) ROUTES="$2"; shift 2;;
  --image)  IMAGE="$2"; shift 2;;
  --scene)  SCENE="$2"; shift 2;;
  *) echo "unknown arg: $1"; exit 1;;
esac; done

[ -n "$VIDEO" ] && { echo "== frames =="; python src/frames_from_video.py "$VIDEO" data/images --fps 2 --max 60; }

# --- shared: SAM 3 masks (object-only) ---
if [ -n "$PROMPT" ]; then
  echo "== SAM 3 masks: \"$PROMPT\" =="
  python src/mask_sam3.py --images data/images --out data/masks --prompt "$PROMPT"
  MASK_ARG="--masks data/masks"
else
  echo "== no --prompt: skipping masks (whole-scene reconstruction) =="
  MASK_ARG=""
fi

# --- shared: VGGT poses + point map (Routes A/B only) ---
if [[ "$ROUTES" == *A* || "$ROUTES" == *B* ]]; then
  echo "== VGGT poses =="
  # shellcheck disable=SC2086  # MASK_ARG is intentionally word-split ("--masks data/masks" or empty)
  python src/pose_vggt.py --images data/images $MASK_ARG --scene "$SCENE"
fi

if [[ "$ROUTES" == *A* ]]; then
  echo "== Route A: Open3D mesh =="
  python routeA_photogrammetry/mesh_from_pointmap.py --scene "$SCENE" --out data/output/mesh_A.obj
fi

if [[ "$ROUTES" == *B* ]]; then
  echo "== Route B: 2D Gaussian Splatting =="
  routeB_2dgs/run_2dgs.sh "$SCENE" data/output/2dgs
fi

if [[ "$ROUTES" == *C* ]]; then
  IMG="$IMAGE"
  if [ -z "$IMG" ]; then
    # default to the first photo (frames_from_video writes .jpg, phones write either)
    IMG="$(find data/images -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) | sort | head -n1)"
  fi
  if [ -z "$IMG" ] || [ ! -f "$IMG" ]; then
    echo "Route C: no input image found in data/images (or pass --image)" >&2; exit 1
  fi
  STEM="$(basename "${IMG%.*}")"
  echo "== Route C: Hunyuan3D from $IMG =="
  python routeC_feedforward/run_hunyuan.py --image "$IMG" \
      --mask "data/masks/${STEM}.png" --out data/output/mesh_C.glb
fi

echo "== done. outputs in data/output/ =="
ls -la data/output 2>/dev/null || true
