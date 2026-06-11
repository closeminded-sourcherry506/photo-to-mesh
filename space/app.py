"""photo-to-mesh — Hugging Face Space (ZeroGPU) demo.

Single-image variant of https://github.com/Hasasasaki/photo-to-mesh :
photo -> Hunyuan3D-2.1 mesh -> scaled GLB (metres) / STL (mm) downloads.

Kept deliberately lean for a shared Space: rembg background removal only
(SAM 3 weights are gated), shape only (no texture stage), per-session
mesh state so concurrent users never collide.
"""
import spaces   # must be imported before torch (ZeroGPU requirement)

import subprocess
import sys
import tempfile
from pathlib import Path

# --- fetch the Hunyuan3D-2.1 engine (same pinned revision as the main repo) ---
HY_REF = "82920d643c0dc2f7bfd7255f45f62d386edfe60c"
ROOT = Path(__file__).resolve().parent
HY = ROOT / "Hunyuan3D-2.1"
if not (HY / "hy3dshape").is_dir():
    subprocess.run(["git", "clone", "https://github.com/tencent-hunyuan/Hunyuan3D-2.1", str(HY)], check=True)
    subprocess.run(["git", "-C", str(HY), "checkout", HY_REF], check=True)
sys.path.insert(0, str(HY / "hy3dshape"))

import gradio as gr
import torch

from mesh_units import FROM_MM, echo_units, scale_factor, size_to_mm
from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline

# ZeroGPU pattern: load to CUDA in global scope; `spaces` defers real allocation.
PIPE = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2.1", device="cuda", dtype=torch.float16)


@spaces.GPU(duration=120)
def _generate(rgba, steps, octree, guidance, seed):
    gen = torch.Generator(device="cuda").manual_seed(int(seed or 0))
    mesh = PIPE(image=rgba, num_inference_steps=int(steps), octree_resolution=int(octree),
                guidance_scale=float(guidance), generator=gen, output_type="trimesh")[0]
    comps = mesh.split(only_watertight=False)
    if len(comps) > 1:
        mesh = max(comps, key=lambda c: len(c.faces))
    return mesh


def _export(mesh, real_size, size_unit):
    """Write a GLB (always metres — glTF has no unit field) and, once a real
    size is set, an STL in mm. Returns (viewer_glb, downloads, info)."""
    out = Path(tempfile.mkdtemp())   # per-call dir: no cross-session collisions
    m = mesh.copy()
    real_mm = size_to_mm(real_size or 0.0, size_unit)
    factor = scale_factor(m.extents, real_mm)
    if factor is not None:
        m.apply_scale(factor)        # mesh now in mm
    e = m.extents

    glb_mesh = m.copy()
    if factor is not None:
        glb_mesh.apply_scale(FROM_MM["m"])
    glb = out / "photo-to-mesh.glb"; glb_mesh.export(glb)
    if factor is None:
        info = (f"size: **{e[0]:.2f} × {e[1]:.2f} × {e[2]:.2f}** (normalized — set a real size to scale) · "
                f"{len(m.vertices):,} verts · STL is written once a real size is set")
        return str(glb), [str(glb)], info

    stl = out / "photo-to-mesh.stl"; m.export(stl)   # mm, slicer-ready
    info = (f"size: **{e[0]:.1f} × {e[1]:.1f} × {e[2]:.1f} mm** (W×H×D) · "
            f"scaled axis: {echo_units(real_mm)} · {len(m.vertices):,} verts · "
            f"GLB in metres (glTF standard) · STL in mm")
    return str(glb), [str(glb), str(stl)], info


def generate(image, remove_bg, steps, octree, guidance, seed, real_size, size_unit):
    if image is None:
        raise gr.Error("Upload an image first.")
    image = image.convert("RGB")
    if remove_bg:
        from rembg import remove
        rgba = remove(image)
    else:
        rgba = image.convert("RGBA")
    mesh = _generate(rgba, steps, octree, guidance, seed)
    glb, files, info = _export(mesh, real_size, size_unit)
    return glb, files, info, mesh        # mesh -> gr.State for instant resizes


def resize(real_size, size_unit, mesh):
    if mesh is None:
        raise gr.Error("Generate a mesh first, then resize.")
    return _export(mesh, real_size, size_unit)


with gr.Blocks(title="photo-to-mesh") as demo:
    gr.Markdown(
        "## photo-to-mesh · Hunyuan3D-2.1\n"
        "One photo → a mesh you can actually print: set a real dimension and download a "
        "**GLB in metres** (glTF standard) or a **slicer-ready STL in mm**.\n\n"
        "Demo Space — the [full project](https://github.com/Hasasasaki/photo-to-mesh) adds "
        "multi-view fusion, SAM 3 text-prompt masking, and photogrammetric routes, locally."
    )
    state = gr.State(None)
    with gr.Row():
        with gr.Column():
            image = gr.Image(label="Photo of an object", type="pil")
            remove_bg = gr.Checkbox(value=True, label="Remove background (rembg)")
            steps = gr.Slider(20, 100, value=30, step=5, label="Diffusion steps (quality vs speed)")
            octree = gr.Slider(128, 512, value=384, step=64, label="Octree resolution (detail)")
            guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance scale")
            seed = gr.Number(value=42, label="Seed", precision=0)
            with gr.Row():
                real_size = gr.Number(value=0, label="Known real size — 0 = leave normalized")
                size_unit = gr.Dropdown(["mm", "cm", "in"], value="mm", label="…size given in")
            btn = gr.Button("Generate mesh", variant="primary")
            resize_btn = gr.Button("↺ Resize & re-export (no GPU re-run)")
        with gr.Column():
            model = gr.Model3D(label="Result — drag to orbit")
            info = gr.Markdown()
            dl = gr.File(label="Download — .glb (metres) · .stl (mm)")
    btn.click(generate, [image, remove_bg, steps, octree, guidance, seed, real_size, size_unit],
              [model, dl, info, state])
    resize_btn.click(resize, [real_size, size_unit, state], [model, dl, info])

demo.queue().launch()
