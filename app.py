#!/usr/bin/env python
"""photo-to-mesh GUI (Hunyuan3D engine) — two modes:
  • Single image  (Hunyuan3D-2.1, generative from one view)
  • Multi-view     (Hunyuan3D-2mv, 1-4 named views: front/left/back/right)

View, tune sliders, generate, and orbit the mesh in the browser.

Run:  ./run_gui.sh     (or)   .venv/bin/python app.py
Then open the printed URL (e.g. http://localhost:7860).
"""
from __future__ import annotations
import os
import sys
import tempfile
from pathlib import Path

# Gradio stores uploaded files under GRADIO_TEMP_DIR (default /tmp/gradio). On a
# shared machine that directory is frequently already owned by another user (or
# root from a prior sudo run), so uploads die with PermissionError ([Errno 13])
# when Gradio mkdir's into it. Default it to a user-owned path (still overridable)
# BEFORE importing gradio, which captures the value at import time.
_gtmp = os.environ.get("GRADIO_TEMP_DIR") or str(Path.home() / ".cache" / "photo-to-mesh" / "gradio")
try:
    Path(_gtmp).mkdir(parents=True, exist_ok=True)
except OSError:
    _gtmp = tempfile.mkdtemp(prefix="photo-to-mesh-gradio-")
os.environ["GRADIO_TEMP_DIR"] = _gtmp

import numpy as np
import torch
from PIL import Image
import gradio as gr

ROOT = Path(__file__).resolve().parent
HY = ROOT / "routeC_feedforward" / "Hunyuan3D-2.1"
if not (HY / "hy3dshape").is_dir():
    sys.exit("Hunyuan3D-2.1 engine not found at routeC_feedforward/Hunyuan3D-2.1.\n"
             "Run:  ./setup.sh core && ./setup.sh routeC   (see README — Quickstart)")
sys.path.insert(0, str(HY / "hy3dshape"))
sys.path.insert(0, str(ROOT / "routeC_feedforward"))   # for hy3dgen_compat
sys.path.insert(0, str(ROOT / "src"))

from mesh_units import FROM_MM, echo_units, scale_factor, size_to_mm  # noqa: E402

# --- path-aware mmap load (avoids 30 GB-RAM OOM on the big ckpts; skips file objects) ---
_orig_load = torch.load
def _mmap_load(*a, **k):
    f = a[0] if a else k.get("f")
    if isinstance(f, (str, os.PathLike)):
        k.setdefault("mmap", True)
    return _orig_load(*a, **k)
torch.load = _mmap_load

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE = ROOT / "data" / "output"; SAVE.mkdir(parents=True, exist_ok=True)

# Pipelines load lazily so the GUI comes up instantly; the first generate of each
# mode pays the model-load (and, once ever, the weight-download) cost.
_PIPES: dict = {}

def single_pipe():
    if "single" not in _PIPES:
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        print("Loading Hunyuan3D-2.1 single-image shape model (downloads on first run) ...")
        _PIPES["single"] = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2.1", device=DEVICE, dtype=torch.float16)
        print("Single-image model ready.")
    return _PIPES["single"]

def mv_pipe():
    if "mv" not in _PIPES:
        import hy3dgen_compat  # noqa: F401  aliases hy3dgen.shapegen -> hy3dshape
        from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        print("Loading Hunyuan3D-2mv multi-view model (downloads on first run) ...")
        _PIPES["mv"] = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2mv", subfolder="hunyuan3d-dit-v2-mv",
            use_safetensors=True, device=DEVICE, dtype=torch.float16)
        print("Multi-view model ready.")
    return _PIPES["mv"]

_SAM = {"proc": None}     # lazy SAM 3
def sam3_mask(image: Image.Image, prompt: str) -> Image.Image:
    if _SAM["proc"] is None:
        from sam3.model.sam3_image_processor import Sam3Processor
        from common import build_sam3
        _SAM["proc"] = Sam3Processor(build_sam3())
    proc = _SAM["proc"]; w, h = image.size
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        st = proc.set_image(image)
        res = proc.set_text_prompt(state=st, prompt=prompt)
    to_np = lambda x: x.float().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    scores = to_np(res["scores"]).reshape(-1)
    masks = to_np(res["masks"])
    if masks.ndim == 4:
        masks = masks[:, 0]
    if len(scores) == 0 or scores.max() < 0.3:
        gr.Warning(f'SAM 3 found no confident match for "{prompt}" — keeping the full image. '
                   "Try a more specific prompt, or switch to rembg auto.")
        return Image.new("L", (w, h), 255)
    m = (masks[int(scores.argmax())] > 0.5).astype(np.uint8) * 255
    return Image.fromarray(m).resize((w, h), Image.NEAREST)


def prep_rgba(image: Image.Image, bg_mode: str, prompt: str):
    image = image.convert("RGB")
    if bg_mode == "SAM 3 prompt":
        mask = sam3_mask(image, prompt)
        rgba = image.convert("RGBA"); rgba.putalpha(mask)
        prev = Image.composite(image, Image.new("RGB", image.size, (255, 255, 255)), mask)
        return rgba, prev
    if bg_mode == "rembg auto":
        from rembg import remove
        rgba = remove(image); return rgba, rgba.convert("RGB")
    return image.convert("RGBA"), image


# cache the last generated NORMALIZED mesh per tab, so we can resize/re-export
# instantly (no GPU re-run).
_LAST: dict = {}


def _export(tag, real_size=0.0, size_unit="mm", axis="tallest", obj_unit="mm"):
    """Scale the cached normalized mesh so `axis` measures `real_size` (given in
    `size_unit`), then write each format in the unit its consumers assume:
      • GLB — always metres (glTF has no unit field; the spec defines 1 unit = 1 m,
        so any other choice imports at the wrong size in Blender/Unity/three.js)
      • STL — always mm (what slicers assume)
      • OBJ — raw numbers in `obj_unit`, for tools where you type/expect a unit
    Returns (viewer_glb, download_paths, info)."""
    base = _LAST.get(tag)
    if base is None:
        raise gr.Error("Generate a mesh first, then resize.")
    mesh = base.copy()
    real_mm = size_to_mm(real_size or 0.0, size_unit)
    factor = scale_factor(mesh.extents, real_mm, axis)
    if factor is not None:
        mesh.apply_scale(factor)   # mesh now in mm

    for old in SAVE.glob(f"gui_{tag}*"):   # drop stale exports from earlier unit choices
        old.unlink(missing_ok=True)

    e = mesh.extents
    glb_mesh = mesh.copy()
    if factor is not None:
        glb_mesh.apply_scale(FROM_MM["m"])    # mm -> metres, the glTF standard
    glb = SAVE / f"gui_{tag}.glb"; glb_mesh.export(glb)

    if factor is None:
        # No real size yet: GLB only (for viewing). Holding back STL/OBJ keeps a
        # "2 mm bottle" in normalized units from ever reaching a slicer.
        info = (f"size: **{e[0]:.2f} × {e[1]:.2f} × {e[2]:.2f}** (normalized — set a real size to scale) · "
                f"{len(mesh.vertices):,} verts · STL/OBJ are written once a real size is set")
        return str(glb), [str(glb)], info

    stl = SAVE / f"gui_{tag}.stl"; mesh.export(stl)
    obj_mesh = mesh.copy(); obj_mesh.apply_scale(FROM_MM[obj_unit])
    obj = SAVE / f"gui_{tag}_{obj_unit}.obj"; obj_mesh.export(obj)
    info = (f"size: **{e[0]:.1f} × {e[1]:.1f} × {e[2]:.1f} mm** (W×H×D) · "
            f"scaled axis: {echo_units(real_mm)} · {len(mesh.vertices):,} verts · "
            f"GLB in metres (glTF standard) · STL in mm · OBJ in {obj_unit}")
    return str(glb), [str(glb), str(stl), str(obj)], info


def _finish(mesh, keep_largest, tag, real_size=0.0, size_unit="mm", axis="tallest", obj_unit="mm"):
    if keep_largest:
        try:
            comps = mesh.split(only_watertight=False)
            if len(comps) > 1:
                mesh = max(comps, key=lambda c: len(c.faces))
        except Exception:
            pass
    _LAST[tag] = mesh.copy()              # cache normalized mesh for later resizes
    return _export(tag, real_size, size_unit, axis, obj_unit)


def gen_single(image, bg_mode, prompt, steps, octree, guidance, seed, keep_largest, real_size, size_unit, axis, obj_unit):
    if image is None:
        raise gr.Error("Upload an image first.")
    image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    rgba, preview = prep_rgba(image, bg_mode, prompt)
    gen = torch.Generator(device=DEVICE).manual_seed(int(seed or 0))   # field can be cleared -> None
    mesh = single_pipe()(image=rgba, num_inference_steps=int(steps), octree_resolution=int(octree),
                         guidance_scale=float(guidance), generator=gen, output_type="trimesh")[0]
    glb, dl, info = _finish(mesh, keep_largest, "single", real_size, size_unit, axis, obj_unit)
    return preview, glb, dl, info


def gen_mv(front, left, back, right, bg_mode, prompt, steps, octree, guidance, seed, keep_largest, real_size, size_unit, axis, obj_unit):
    raw = {"front": front, "left": left, "back": back, "right": right}
    raw = {k: v for k, v in raw.items() if v is not None}
    if not raw:
        raise gr.Error("Provide at least one view (front/left/back/right).")
    views, previews = {}, []
    for tag, im in raw.items():
        im = Image.fromarray(im) if isinstance(im, np.ndarray) else im
        rgba, prev = prep_rgba(im, bg_mode, prompt)
        views[tag] = rgba
        previews.append(np.array(prev.convert("RGB")))
    gen = torch.Generator(device=DEVICE).manual_seed(int(seed or 0))   # field can be cleared -> None
    mesh = mv_pipe()(image=views, num_inference_steps=int(steps), octree_resolution=int(octree),
                     guidance_scale=float(guidance), generator=gen, output_type="trimesh")[0]
    glb, dl, info = _finish(mesh, keep_largest, "mv", real_size, size_unit, axis, obj_unit)
    info = f"views used: {list(views.keys())} · " + info
    return previews, glb, dl, info


DEF = ROOT / "data" / "images"
_imgs = list(DEF.glob("*.jp*g")) + list(DEF.glob("*.png"))
default_img = str(_imgs[0]) if _imgs else None


def _controls():
    bg = gr.Radio(["SAM 3 prompt", "rembg auto", "none"], value="SAM 3 prompt", label="Background removal")
    prompt = gr.Textbox(value="the object", label='SAM 3 prompt — name the object to keep, e.g. "the blue mug"')
    steps = gr.Slider(20, 100, value=50, step=5, label="Diffusion steps")
    octree = gr.Slider(128, 512, value=384, step=64, label="Octree resolution (detail)")
    guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance scale")
    seed = gr.Number(value=42, label="Seed", precision=0)
    keep = gr.Checkbox(value=True, label="Keep largest component")
    with gr.Row():
        real_size = gr.Number(value=0, label="Known real size — 0 = leave normalized")
        size_unit = gr.Dropdown(["mm", "cm", "in"], value="mm", label="…size given in")
        axis = gr.Dropdown(["tallest", "X", "Y", "Z"], value="tallest", label="…measured along")
    obj_unit = gr.Dropdown(["mm", "cm", "in"], value="mm",
                           label="OBJ numbers unit (GLB is always metres, STL always mm)")
    return bg, prompt, steps, octree, guidance, seed, keep, real_size, size_unit, axis, obj_unit


with gr.Blocks(title="photo-to-mesh") as demo:
    gr.Markdown("## photo-to-mesh · Hunyuan3D\nGenerate a mesh from **one image** or **multiple views**, then orbit it on the right.")
    with gr.Tabs():
        # ---------------- Single image ----------------
        with gr.Tab("Single image (2.1)"):
            with gr.Row():
                with gr.Column():
                    s_in = gr.Image(label="Input image", type="pil", value=default_img)
                    s_bg, s_prompt, s_steps, s_octree, s_guid, s_seed, s_keep, s_real, s_sizeu, s_axis, s_obju = _controls()
                    s_btn = gr.Button("Generate mesh", variant="primary")
                    s_resize = gr.Button("↺ Resize & re-export (no re-generate)")
                with gr.Column():
                    s_prev = gr.Image(label="Masked input")
                    s_model = gr.Model3D(label="Result — drag to orbit")
                    s_info = gr.Markdown()
                    s_dl = gr.File(label="Download — .glb (metres) · .stl (mm) · .obj")
            s_btn.click(gen_single, [s_in, s_bg, s_prompt, s_steps, s_octree, s_guid, s_seed, s_keep, s_real, s_sizeu, s_axis, s_obju],
                        [s_prev, s_model, s_dl, s_info])
            s_resize.click(lambda r, su, a, ou: _export("single", r, su, a, ou), [s_real, s_sizeu, s_axis, s_obju],
                           [s_model, s_dl, s_info])

        # ---------------- Multi-view ----------------
        with gr.Tab("Multi-view (2mv)"):
            gr.Markdown("Provide **1–4 views**. Leave slots empty if you don't have them. "
                        "Canonical order: front, left (=90° CW), back, right (=270°).")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        m_front = gr.Image(label="Front", type="pil", value=default_img)
                        m_left = gr.Image(label="Left (90° CW)", type="pil")
                    with gr.Row():
                        m_back = gr.Image(label="Back", type="pil")
                        m_right = gr.Image(label="Right (270°)", type="pil")
                    m_bg, m_prompt, m_steps, m_octree, m_guid, m_seed, m_keep, m_real, m_sizeu, m_axis, m_obju = _controls()
                    m_btn = gr.Button("Generate mesh (multi-view)", variant="primary")
                    m_resize = gr.Button("↺ Resize & re-export (no re-generate)")
                with gr.Column():
                    m_prev = gr.Gallery(label="Masked views", columns=2, height=240)
                    m_model = gr.Model3D(label="Result — drag to orbit")
                    m_info = gr.Markdown()
                    m_dl = gr.File(label="Download — .glb (metres) · .stl (mm) · .obj")
            m_btn.click(gen_mv,
                        [m_front, m_left, m_back, m_right, m_bg, m_prompt, m_steps, m_octree, m_guid, m_seed, m_keep, m_real, m_sizeu, m_axis, m_obju],
                        [m_prev, m_model, m_dl, m_info])
            m_resize.click(lambda r, su, a, ou: _export("mv", r, su, a, ou), [m_real, m_sizeu, m_axis, m_obju],
                           [m_model, m_dl, m_info])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, inbrowser=False)
