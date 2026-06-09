# object-scan

Turn photos of a real object into a **clean, scaled, downloadable 3D mesh** — with
an interactive **web GUI** as the primary interface.

The core engine is **Hunyuan3D** (single-image *and* multi-view), front-ended by
**SAM 3** for object isolation. A second, photogrammetric path (**VGGT → mesh /
2D Gaussian Splatting**) is included for true multi-view reconstruction.

```
                         ┌─────────────────────────────────────────────┐
  photo(s) ──► SAM 3 ──► │  CORE: Hunyuan3D  (generative image → mesh)  │ ──► .glb / .stl
              (mask)     │   • Single image   (Hunyuan3D-2.1)           │
                         │   • Multi-view 1-4 (Hunyuan3D-2mv)           │
                         └─────────────────────────────────────────────┘
  many photos ─► SAM 3 ─► VGGT poses+pointmap ─► Open3D mesh  (Route A)  ──► .obj   (reconstruction)
                                              └─► 2D Gaussian Splatting  (Route B)  ──► .ply
```

---

## 1. Core methodology — Hunyuan3D

**Hunyuan3D is a *generative* 3D model, not a photogrammetric one.** It was trained
on `(image → 3D asset)` pairs and learns a prior `P(shape | image)`. Given a photo,
a flow-matching **DiT** denoises a 3D latent conditioned on image tokens; a VAE
decodes it to an occupancy/SDF field, and marching cubes extracts the surface. The
unseen sides of the object are **plausibly invented** from the learned prior.

Two variants are wired up:

| Mode | Checkpoint | Conditioner | Input |
|------|------------|-------------|-------|
| **Single image** | `tencent/Hunyuan3D-2.1` (`hunyuan3d-dit-v2-1`) | `SingleImageEncoder` | 1 image |
| **Multi-view** | `tencent/Hunyuan3D-2mv` (`hunyuan3d-dit-v2-mv`) | `DinoImageEncoderMV` (4 view embeddings) | 1–4 named views: front / left / back / right |

Multi-view still *generates*, but conditions on several views via per-view
embeddings (it does **not** triangulate — it has no camera poses). For true
geometry from many photos, use the VGGT routes (§5).

**Consequences worth knowing:**
- Output is in **normalized units** (object fit into a ~`[-1,1]` cube). There is no
  metric scale from images alone — you impose it (§4).
- Multi-view expects **canonical orthogonal views** (front, left = 90° CW, back,
  right = 270°). Feeding similar/oblique views degrades the result.
- Transparent/reflective parts (e.g. liquid) aren't physically reconstructed.

---

## 2. The GUI (primary interface)

```bash
./run_gui.sh
# then open http://localhost:7860 in a browser
```

Two tabs, both with the same controls:

- **Single image (2.1)** — upload one photo → mesh.
- **Multi-view (2mv)** — fill **1–4 slots** (Front / Left / Back / Right) → fused mesh.

**Controls**
| Control | What it does |
|---|---|
| Background removal | `SAM 3 prompt` (name the object, e.g. `"the blue spray bottle"`), `rembg auto`, or `none` |
| Diffusion steps | quality vs speed (default 50) |
| Octree resolution | mesh detail (128–512, default 384) |
| Guidance scale | faithfulness to the image (default 7.5) |
| Seed | change for different plausible back-sides |
| Keep largest component | drops stray floaters |
| **Known real size (mm)** + axis | impose metric scale (§4) |
| **GLB unit** (`mm`/`cm`/`m`) | unit the downloaded GLB is encoded in |
| **↺ Resize & re-export** | re-scale the *last* mesh instantly — **no GPU re-run** |

**Outputs** (in `data/output/`): an interactive orbit viewer, a downloadable
`gui_<mode>_<unit>.glb`, and a millimetre `gui_<mode>.stl` for slicers.

---

## 3. Setup (uv-based, no sudo)

```bash
./setup.sh core            # uv venv + torch(cu128) + VGGT + SAM 3 + common deps
huggingface-cli login      # one-time: SAM 3 weights are gated
./setup.sh routeC          # Hunyuan3D-2.1 repo + texture-stage build (for textures)
./setup.sh download hunyuan-full   # pre-fetch Hunyuan weights (~15 GB, optional)
# Multi-view weights (Hunyuan3D-2mv ~5 GB + its VAE) download on first GUI use.
```
`./setup.sh routeB` adds the 2D Gaussian Splatting build (only for Route B).

Model weights live in `~/.cache/huggingface` and `~/.cache/hy3dgen`; first run is
offline after that.

---

## 4. Scale & resize

Images carry **no absolute size**, so scale is *imposed*, not measured:

1. Measure one real dimension of the object (e.g. bottle height ≈ `255` mm).
2. Enter it in **Known real size (mm)** and pick the axis (`tallest` is robust).
3. The mesh is scaled so that dimension matches; the readout shows the bounding box.
4. Choose **GLB unit** so the file is physically correct where you open it:

| Opening in… | GLB unit | Why |
|---|---|---|
| Blender / Unity / three.js / web glTF viewers | **m** | glTF's native unit is metres |
| A tool where you type/expect mm | **mm** | numbers are literal millimetres |
| A 3D-print slicer | use the **STL** | STL is unitless; slicers assume mm |

> **Note on the bounding box readout:** the three numbers are `mesh.extents`
> (`max−min` per axis) in normalized units; one is always ≈2.0 because Hunyuan fits
> the longest axis to the `[-1,1]` cube. The `W×H×D` labels assume the object is
> upright (height along Y) and can be mislabeled if it was photographed lying down —
> the *scaling itself* (via `tallest`) is unaffected.

---

## 5. Other routes (photogrammetric / reconstruction)

For genuine multi-view geometry from many photos (not generation):

```bash
source .venv/bin/activate
# Route A — VGGT point map -> Open3D mesh (fast, no extra build)
python src/mask_sam3.py --images data/images --out data/masks --prompt "the mug"
python src/pose_vggt.py  --images data/images --masks data/masks --scene data/output/scene
python routeA_photogrammetry/mesh_from_pointmap.py --scene data/output/scene --out data/output/mesh_A.obj

# Route B — VGGT poses -> 2D Gaussian Splatting -> TSDF mesh (needs ./setup.sh routeB)
routeB_2dgs/run_2dgs.sh data/output/scene data/output/2dgs
```
**VGGT** (feed-forward) replaces COLMAP for camera poses — no sudo, robust on few
photos. See the route folders for details.

---

## 6. CLI for the core (no GUI)

```bash
source .venv/bin/activate
# single image
python routeC_feedforward/run_hunyuan.py \
    --image "data/images/your.jpg" --mask "data/masks/your.png" \
    --out data/output/mesh.glb --no-texture
# turntable preview of any mesh
python src/turntable.py data/output/mesh.glb --out data/output/mesh_turntable
```

---

## 7. Project layout

```
object-scan/
├── app.py                     # the GUI (single + multi-view tabs, scale, resize)
├── run_gui.sh                 # launch the GUI
├── setup.sh                   # uv env + deps + model downloads (core/routeB/routeC/download)
├── pyproject.toml             # uv-managed deps
├── src/
│   ├── mask_sam3.py           # SAM 3 object masking (text/concept prompt)
│   ├── pose_vggt.py           # VGGT poses + point map -> COLMAP-format scene
│   ├── turntable.py           # headless GIF/MP4 turntable for any mesh
│   ├── frames_from_video.py   # video -> sharp frames
│   └── common.py
├── routeC_feedforward/
│   ├── run_hunyuan.py         # CLI single-image Hunyuan
│   ├── hy3dgen_compat.py      # shim: aliases hy3dgen.shapegen -> hy3dshape (for 2mv ckpt)
│   └── Hunyuan3D-2.1/         # cloned repo (by setup.sh routeC)
├── routeA_photogrammetry/mesh_from_pointmap.py
├── routeB_2dgs/{run_2dgs.sh, apply_masks.py}
└── data/{images, masks, output}/
```

---

## 8. Hardware notes & gotchas (recorded from bring-up)

Tested on an **RTX 5090 (Blackwell, sm_120), 30 GB RAM, no passwordless sudo**.

- **Blackwell → CUDA 12.8**: torch is installed from the `cu128` index;
  `TORCH_CUDA_ARCH_LIST=12.0` for any CUDA builds.
- **30 GB RAM**: the 7.4 GB fp16 checkpoint OOMs a naive `torch.load`. We patch
  `torch.load` to use `mmap=True` **for file paths only** (SAM 3 loads from a file
  object, where mmap is invalid — hence the path guard).
- **SAM 3** must run inside a `torch.autocast(cuda, bfloat16)` context, and its
  outputs are CUDA tensors (`.cpu()` before numpy).
- **Hunyuan offline loading** uses `~/.cache/hy3dgen`, not the standard HF cache;
  don't set `HF_HUB_OFFLINE=1` (its loader needs to resolve the snapshot).
- **Hunyuan3D-2mv** config targets the old `hy3dgen.shapegen.*` package; the 2.1
  repo renamed it to `hy3dshape`. `hy3dgen_compat.py` aliases them in `sys.modules`.
- The 2mv DiT borrows its **VAE from `tencent/Hunyuan3D-2`** — that VAE is fetched too.

---

## 9. Status

| Piece | State |
|---|---|
| SAM 3 masking | ✅ working |
| Hunyuan single-image (GUI + CLI) | ✅ working, validated on real photos |
| Hunyuan multi-view (2mv) | ✅ working, validated (1–4 views) |
| Scale / resize / unit export / STL | ✅ working |
| Turntable GIF/MP4 | ✅ working |
| Hunyuan PBR **texture** stage | ⚠️ needs `custom_rasterizer` CUDA build; untested |
| Routes A/B (VGGT / 2DGS) | ⚠️ implemented, not yet run end-to-end here |

---

## 10. License

This project's **code** is released under the **MIT License** (see `LICENSE`).

The **models it downloads at runtime keep their own licenses** — review them before
any commercial or redistribution use:
- **Hunyuan3D-2.1 / 2mv** — Tencent Hunyuan Community License (has usage
  restrictions; not a permissive OSS license).
- **SAM 3** — Meta's SAM license (gated download, research-oriented terms).
- **VGGT** — see its repository for terms.

No model weights are vendored in this repo; `setup.sh` fetches them from their
official sources.

## Acknowledgements

Built on [Hunyuan3D](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1),
[SAM 3](https://github.com/facebookresearch/sam3),
[VGGT](https://github.com/facebookresearch/vggt),
[2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting), and
[Gradio](https://www.gradio.app/).
