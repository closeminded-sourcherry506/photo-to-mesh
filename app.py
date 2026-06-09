#!/usr/bin/env python
"""Interactive GUI for Route C (Hunyuan3D) — two modes:
  • Single image  (Hunyuan3D-2.1, generative from one view)
  • Multi-view     (Hunyuan3D-2mv, 1-4 named views: front/left/back/right)

View, tune sliders, generate, and orbit the mesh in the browser.

Run:  ./run_gui.sh     (or)   .venv/bin/python app.py
Then open the printed URL (e.g. http://localhost:7860).
"""
from __future__ import annotations
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import gradio as gr

ROOT = Path(__file__).resolve().parent
HY = ROOT / "routeC_feedforward" / "Hunyuan3D-2.1"
sys.path.insert(0, str(HY / "hy3dshape"))
sys.path.insert(0, str(ROOT / "routeC_feedforward"))   # for hy3dgen_compat
sys.path.insert(0, str(ROOT / "src"))

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

from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

print("Loading Hunyuan3D-2.1 single-image shape model ...")
SHAPE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2.1", device=DEVICE, dtype=torch.float16)
print("Single-image model ready.")

_MV = {"pipe": None}      # lazy: Hunyuan3D-2mv (loads on first multi-view generate)
def mv_pipe():
    if _MV["pipe"] is None:
        import hy3dgen_compat  # noqa: F401  aliases hy3dgen.shapegen -> hy3dshape
        print("Loading Hunyuan3D-2mv multi-view model ...")
        _MV["pipe"] = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            "tencent/Hunyuan3D-2mv", subfolder="hunyuan3d-dit-v2-mv",
            use_safetensors=True, device=DEVICE, dtype=torch.float16)
        print("Multi-view model ready.")
    return _MV["pipe"]

_SAM = {"proc": None}     # lazy SAM 3
def sam3_mask(image: Image.Image, prompt: str) -> Image.Image:
    if _SAM["proc"] is None:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
        _SAM["proc"] = Sam3Processor(build_sam3_image_model())
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
# multiply a mm-sized mesh by this to express it in the chosen GLB unit
_UNIT_FACTOR = {"mm": 1.0, "cm": 0.1, "m": 0.001}


def _export(tag, real_mm=0.0, axis="tallest", glb_unit="mm"):
    """Scale the cached normalized mesh to `real_mm` and write a GLB (in glb_unit)
    + an STL (always mm, for slicers). Returns (glb_path, dl_path, info)."""
    base = _LAST.get(tag)
    if base is None:
        raise gr.Error("Generate a mesh first, then resize.")
    mesh = base.copy()
    ext = mesh.extents  # normalized units
    scaled = False
    if real_mm and float(real_mm) > 0:
        ref = float(max(ext)) if axis == "tallest" else float(ext[{"X": 0, "Y": 1, "Z": 2}[axis]])
        if ref > 0:
            mesh.apply_scale(float(real_mm) / ref)   # mesh now in mm
            scaled = True

    e_mm = mesh.extents
    # GLB in the requested unit (glTF's native unit is metres -> use 'm' for Blender/Unity/web)
    glb_mesh = mesh.copy(); glb_mesh.apply_scale(_UNIT_FACTOR.get(glb_unit, 1.0))
    glb = SAVE / f"gui_{tag}_{glb_unit}.glb"; glb_mesh.export(glb)
    stl = SAVE / f"gui_{tag}.stl"; mesh.export(stl)   # STL stays in mm

    if scaled:
        dims = f"**{e_mm[0]:.1f} × {e_mm[1]:.1f} × {e_mm[2]:.1f} mm** (W×H×D)"
        extra = f" · GLB encoded in **{glb_unit}** · STL(mm): `{stl.name}`"
    else:
        dims = f"**{e_mm[0]:.2f} × {e_mm[1]:.2f} × {e_mm[2]:.2f}** (normalized — set a real size to scale)"
        extra = ""
    info = f"size: {dims} · {len(mesh.vertices):,} verts{extra}"
    return str(glb), str(glb), info


def _finish(mesh, keep_largest, tag, real_mm=0.0, axis="tallest", glb_unit="mm"):
    if keep_largest:
        try:
            comps = mesh.split(only_watertight=False)
            if len(comps) > 1:
                mesh = max(comps, key=lambda c: len(c.faces))
        except Exception:
            pass
    _LAST[tag] = mesh.copy()              # cache normalized mesh for later resizes
    return _export(tag, real_mm, axis, glb_unit)


def gen_single(image, bg_mode, prompt, steps, octree, guidance, seed, keep_largest, real_mm, axis, unit):
    if image is None:
        raise gr.Error("Upload an image first.")
    image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
    rgba, preview = prep_rgba(image, bg_mode, prompt)
    gen = torch.Generator(device=DEVICE).manual_seed(int(seed))
    mesh = SHAPE(image=rgba, num_inference_steps=int(steps), octree_resolution=int(octree),
                 guidance_scale=float(guidance), generator=gen, output_type="trimesh")[0]
    glb, dl, info = _finish(mesh, keep_largest, "single", real_mm, axis, unit)
    return preview, glb, dl, info


def gen_mv(front, left, back, right, bg_mode, prompt, steps, octree, guidance, seed, keep_largest, real_mm, axis, unit):
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
    gen = torch.Generator(device=DEVICE).manual_seed(int(seed))
    mesh = mv_pipe()(image=views, num_inference_steps=int(steps), octree_resolution=int(octree),
                     guidance_scale=float(guidance), generator=gen, output_type="trimesh")[0]
    glb, dl, info = _finish(mesh, keep_largest, "mv", real_mm, axis, unit)
    info = f"views used: {list(views.keys())} · " + info
    return previews, glb, dl, info


DEF = ROOT / "data" / "images"
_imgs = list(DEF.glob("*.jp*g")) + list(DEF.glob("*.png"))
default_img = str(_imgs[0]) if _imgs else None


def _controls():
    bg = gr.Radio(["SAM 3 prompt", "rembg auto", "none"], value="SAM 3 prompt", label="Background removal")
    prompt = gr.Textbox(value="the blue spray bottle", label="SAM 3 prompt (object to keep)")
    steps = gr.Slider(20, 100, value=50, step=5, label="Diffusion steps")
    octree = gr.Slider(128, 512, value=384, step=64, label="Octree resolution (detail)")
    guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance scale")
    seed = gr.Number(value=42, label="Seed", precision=0)
    keep = gr.Checkbox(value=True, label="Keep largest component")
    with gr.Row():
        real_size = gr.Number(value=0, label="Known real size (mm) — 0 = leave normalized")
        axis = gr.Dropdown(["tallest", "X", "Y", "Z"], value="tallest", label="…measured along")
        unit = gr.Dropdown(["mm", "cm", "m"], value="mm", label="GLB unit (use 'm' for Blender/Unity/web)")
    return bg, prompt, steps, octree, guidance, seed, keep, real_size, axis, unit


with gr.Blocks(title="Object Scan — Route C (Hunyuan3D)") as demo:
    gr.Markdown("## Object Scan · Route C (Hunyuan3D)\nGenerate a mesh from **one image** or **multiple views**, then orbit it on the right.")
    with gr.Tabs():
        # ---------------- Single image ----------------
        with gr.Tab("Single image (2.1)"):
            with gr.Row():
                with gr.Column():
                    s_in = gr.Image(label="Input image", type="pil", value=default_img)
                    s_bg, s_prompt, s_steps, s_octree, s_guid, s_seed, s_keep, s_real, s_axis, s_unit = _controls()
                    s_btn = gr.Button("Generate mesh", variant="primary")
                    s_resize = gr.Button("↺ Resize & re-export (no re-generate)")
                with gr.Column():
                    s_prev = gr.Image(label="Masked input")
                    s_model = gr.Model3D(label="Result — drag to orbit")
                    s_info = gr.Markdown()
                    s_dl = gr.File(label="Download .glb")
            s_btn.click(gen_single, [s_in, s_bg, s_prompt, s_steps, s_octree, s_guid, s_seed, s_keep, s_real, s_axis, s_unit],
                        [s_prev, s_model, s_dl, s_info])
            s_resize.click(lambda r, a, u: _export("single", r, a, u), [s_real, s_axis, s_unit],
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
                    m_bg, m_prompt, m_steps, m_octree, m_guid, m_seed, m_keep, m_real, m_axis, m_unit = _controls()
                    m_btn = gr.Button("Generate mesh (multi-view)", variant="primary")
                    m_resize = gr.Button("↺ Resize & re-export (no re-generate)")
                with gr.Column():
                    m_prev = gr.Gallery(label="Masked views", columns=2, height=240)
                    m_model = gr.Model3D(label="Result — drag to orbit")
                    m_info = gr.Markdown()
                    m_dl = gr.File(label="Download .glb")
            m_btn.click(gen_mv,
                        [m_front, m_left, m_back, m_right, m_bg, m_prompt, m_steps, m_octree, m_guid, m_seed, m_keep, m_real, m_axis, m_unit],
                        [m_prev, m_model, m_dl, m_info])
            m_resize.click(lambda r, a, u: _export("mv", r, a, u), [m_real, m_axis, m_unit],
                           [m_model, m_dl, m_info])

if __name__ == "__main__":
    demo.queue().launch(server_name="0.0.0.0", server_port=7860, inbrowser=False)
