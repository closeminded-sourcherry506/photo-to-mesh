#!/usr/bin/env bash
# Route B — 2D Gaussian Splatting from a VGGT scene -> TSDF mesh.
#
# Prereq: ./setup.sh core && ./setup.sh routeB   (builds the surfel rasterizer)
#         a VGGT scene at data/output/scene (run src/pose_vggt.py first)
#
# Usage:  routeB_2dgs/run_2dgs.sh data/output/scene data/output/2dgs
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
# shellcheck disable=SC1091
source .venv/bin/activate

# point the build/runtime at the pip-provided CUDA, if present
# (TORCH_CUDA_ARCH_LIST is only needed for source builds — export it yourself to override)
NVCC_BIN="$(python -c 'import os,nvidia.cuda_nvcc as m; print(os.path.join(os.path.dirname(m.__file__),"bin"))' 2>/dev/null || true)"
if [ -n "$NVCC_BIN" ]; then
  CUDA_HOME="$(dirname "$NVCC_BIN")"
  export CUDA_HOME
  export PATH="$NVCC_BIN:$PATH"
fi

SCENE="${1:-data/output/scene}"
OUT="${2:-data/output/2dgs}"
GS=routeB_2dgs/2d-gaussian-splatting

# Optional: black out background using SAM 3 masks so Gaussians focus on the object.
if [ -d data/masks ] && [ -z "${SKIP_MASK:-}" ]; then
  echo "==> applying masks (background -> black) for object-only training"
  python routeB_2dgs/apply_masks.py --scene "$SCENE" --masks data/masks
fi

echo "==> training 2DGS"
# depth_ratio 1.0 => sharper, bounded surfaces (ideal for a single object)
python "$GS/train.py" -s "$SCENE" -m "$OUT" --depth_ratio 1.0 -r 1

echo "==> extracting mesh via TSDF depth fusion"
python "$GS/render.py" -s "$SCENE" -m "$OUT" --depth_ratio 1.0 \
    --voxel_size 0.004 --sdf_trunc 0.02 --mesh_res 1024 --skip_test --skip_train

echo "==> mesh(es) in: $OUT/train/ours_*/fuse_post.ply (and fuse.ply)"
