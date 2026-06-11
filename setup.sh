#!/usr/bin/env bash
# =============================================================================
# photo-to-mesh setup — uv-based, no sudo required.
#
# Stages (run what you need):
#   ./setup.sh core      # uv venv + torch (CUDA) + common deps + VGGT + SAM 3
#   ./setup.sh routeC    # Hunyuan3D-2.1 (the GUI's core engine) + texture-stage build
#   ./setup.sh routeB    # extra: clone & build 2D Gaussian Splatting (needs nvcc; installed via pip)
#   ./setup.sh download [hunyuan-shape|hunyuan-full|vggt]   # pre-fetch model weights
#   ./setup.sh all       # core + routeB + routeC
#
# Hardware (auto-detected, override with env vars):
#   TORCH_CUDA_ARCH_LIST  arch for CUDA extension builds (default: read from
#                         nvidia-smi — 12.0 Blackwell / 8.9 Ada / 8.6 Ampere;
#                         if undetectable, PyTorch probes the GPU at build time)
#   CUDA_INDEX            PyTorch wheel index (default cu128 — required for
#                         Blackwell/RTX 50xx, fine on Ampere and newer;
#                         e.g. CUDA_INDEX=https://download.pytorch.org/whl/cu124)
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

CUDA_INDEX="${CUDA_INDEX:-https://download.pytorch.org/whl/cu128}"
STAGE="${1:-core}"

# Pinned upstream revisions — a known-good set, bump deliberately (and together).
VGGT_REF="${VGGT_REF:-a288dd0f14786c93483e45524328726ab7b1b4ce}"
SAM3_REF="${SAM3_REF:-8e451d5eb43c817b64ae7577fb7b9ae223db88a9}"
HUNYUAN_REF="${HUNYUAN_REF:-82920d643c0dc2f7bfd7255f45f62d386edfe60c}"
TWODGS_REF="${TWODGS_REF:-335ad612f2e783a4e57b9cbc4d1e167bd599fc98}"

log() { printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# GPU compute capability for CUDA source builds (honors user-set TORCH_CUDA_ARCH_LIST).
detect_arch() {
  if [ -n "${TORCH_CUDA_ARCH_LIST:-}" ]; then
    echo "$TORCH_CUDA_ARCH_LIST"
  else
    nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1 || true
  fi
}

export_arch() {
  local arch
  arch="$(detect_arch)"
  if [ -n "$arch" ]; then
    export TORCH_CUDA_ARCH_LIST="$arch"
    log "CUDA builds target sm arch: $arch"
  else
    log "could not detect GPU arch (no nvidia-smi?) — PyTorch will probe the GPU at build time"
  fi
}

pin_repo() {  # pin_repo <dir> <ref> — best-effort checkout of the pinned revision
  git -C "$1" fetch --quiet origin "$2" 2>/dev/null || true
  if git -C "$1" checkout --quiet "$2" 2>/dev/null; then
    git -C "$1" submodule update --init --recursive --quiet 2>/dev/null || true
  else
    log "[warn] could not pin $1 to $2 — using its current checkout"
  fi
}

ensure_uv() {
  command -v uv >/dev/null 2>&1 || { log "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; }
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
}

setup_core() {
  ensure_uv
  if [ -d .venv ]; then
    log "reusing existing venv (.venv)"
  else
    log "creating venv (.venv) with Python 3.12"
    uv venv --python 3.12 .venv
  fi
  # shellcheck disable=SC1091
  source .venv/bin/activate

  log "installing PyTorch from $CUDA_INDEX"
  uv pip install --index-url "$CUDA_INDEX" torch torchvision

  log "installing common deps (open3d, opencv, trimesh, ...)"
  uv pip install -e .

  log "installing VGGT (pose backend) from GitHub, pinned"
  uv pip install "git+https://github.com/facebookresearch/vggt.git@${VGGT_REF}"

  log "installing SAM 3 (concept-prompt object masking) from GitHub, pinned"
  uv pip install "git+https://github.com/facebookresearch/sam3.git@${SAM3_REF}"
  cat <<'NOTE'
  NOTE: SAM 3 checkpoints are gated on Hugging Face. One-time (optional —
        the GUI's "rembg auto" background removal works without it):
        huggingface-cli login           # paste an HF token
        # then request access on the SAM 3 model page.
        Weights download automatically on first run of src/mask_sam3.py.
NOTE
  log "core ready.  VGGT-1B + SAM 3 weights download automatically on first run."
}

setup_routeB() {
  # shellcheck disable=SC1091
  source .venv/bin/activate
  log "Route B: installing a pip CUDA toolkit so the 2DGS rasterizer can compile (no sudo)"
  uv pip install nvidia-cuda-nvcc-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-cccl-cu12
  # Point the build at the pip-provided nvcc.
  NVCC_BIN="$(python -c 'import os,nvidia.cuda_nvcc as m; print(os.path.join(os.path.dirname(m.__file__),"bin"))')"
  CUDA_HOME="$(dirname "$NVCC_BIN")"
  export CUDA_HOME
  export PATH="$NVCC_BIN:$PATH"
  export_arch
  log "CUDA_HOME=$CUDA_HOME"

  log "cloning hbb1/2d-gaussian-splatting (pinned)"
  [ -d routeB_2dgs/2d-gaussian-splatting ] || \
    git clone --recursive https://github.com/hbb1/2d-gaussian-splatting routeB_2dgs/2d-gaussian-splatting
  pin_repo routeB_2dgs/2d-gaussian-splatting "$TWODGS_REF"

  log "building 2DGS submodules (surfel rasterizer + simple-knn)"
  uv pip install routeB_2dgs/2d-gaussian-splatting/submodules/diff-surfel-rasterization
  uv pip install routeB_2dgs/2d-gaussian-splatting/submodules/simple-knn
  uv pip install plyfile mediapy lpips
  log "Route B ready."
}

setup_routeC() {
  # shellcheck disable=SC1091
  source .venv/bin/activate
  log "Route C: cloning Hunyuan3D-2.1 (feed-forward image->mesh, pinned)"
  [ -d routeC_feedforward/Hunyuan3D-2.1 ] || \
    git clone https://github.com/tencent-hunyuan/Hunyuan3D-2.1 routeC_feedforward/Hunyuan3D-2.1
  H=routeC_feedforward/Hunyuan3D-2.1
  pin_repo "$H" "$HUNYUAN_REF"

  # IMPORTANT: the repo pins torch 2.5.1+cu124, which does NOT run on Blackwell/RTX 5090.
  # We keep the torch from `setup.sh core` and install the rest WITHOUT touching torch.
  log "installing Hunyuan3D deps (keeping our torch)"
  grep -viE '^(torch|torchvision|torchaudio)\b' "$H/requirements.txt" > /tmp/hy_reqs.txt || true
  uv pip install -r /tmp/hy_reqs.txt || true

  log "building texture-stage extensions (custom_rasterizer + mesh painter)"
  export_arch
  NVCC_BIN="$(python -c 'import os,nvidia.cuda_nvcc as m; print(os.path.join(os.path.dirname(m.__file__),"bin"))' 2>/dev/null || true)"
  if [ -n "$NVCC_BIN" ]; then
    CUDA_HOME="$(dirname "$NVCC_BIN")"
    export CUDA_HOME
    export PATH="$NVCC_BIN:$PATH"
  else
    log "note: install routeB first (or a CUDA toolkit) if these builds need nvcc"
  fi
  ( cd "$H/hy3dpaint/custom_rasterizer" && uv pip install -e . ) || log "[warn] custom_rasterizer build failed (texture stage may be unavailable)"
  ( cd "$H/hy3dpaint/DifferentiableRenderer" && bash compile_mesh_painter.sh ) || log "[warn] mesh painter compile failed"

  log "downloading RealESRGAN upscaler checkpoint (texture stage)"
  mkdir -p "$H/hy3dpaint/ckpt"
  [ -f "$H/hy3dpaint/ckpt/RealESRGAN_x4plus.pth" ] || \
    wget -q --show-progress -P "$H/hy3dpaint/ckpt" \
      https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth || true

  log "Route C ready.  Model weights (tencent/Hunyuan3D-2.1) download from HF on first run."
  log "Shape-only works even if the texture build fails — use --no-texture."
}

setup_download() {
  # Pre-download model checkpoints into the local HF cache so first run is instant.
  # Everything is OPEN WEIGHTS that run locally — there is no paid API.
  # shellcheck disable=SC1091
  source .venv/bin/activate
  export HF_HUB_ENABLE_HF_TRANSFER=1
  WHAT="${2:-hunyuan-shape}"
  case "$WHAT" in
    hunyuan-shape)   # ~8 GB: shape DiT + VAE only (use --no-texture at runtime)
      log "downloading Hunyuan3D-2.1 SHAPE weights (~8 GB)"
      hf download tencent/Hunyuan3D-2.1 --include "hunyuan3d-dit-v2-1/*" "hunyuan3d-vae-v2-1/*" ;;
    hunyuan-full)    # ~15 GB: shape + PBR texture
      log "downloading Hunyuan3D-2.1 FULL weights (~15 GB)"
      hf download tencent/Hunyuan3D-2.1 ;;
    vggt)
      log "downloading VGGT-1B (~5 GB)"; hf download facebook/VGGT-1B ;;
    *) echo "usage: ./setup.sh download [hunyuan-shape|hunyuan-full|vggt]"; exit 1 ;;
  esac
  log "cached under ~/.cache/huggingface — from_pretrained() will use it offline."
}

case "$STAGE" in
  core)     setup_core ;;
  routeB)   setup_routeB ;;
  routeC)   setup_routeC ;;
  download) setup_download "$@" ;;
  all)      setup_core; setup_routeB; setup_routeC ;;
  *) echo "usage: ./setup.sh [core|routeB|routeC|download|all]"; exit 1 ;;
esac
log "DONE: '$STAGE'.  Activate with:  source .venv/bin/activate"
