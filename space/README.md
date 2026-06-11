---
title: photo-to-mesh
emoji: 🧊
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.33.0
app_file: app.py
pinned: false
license: mit
short_description: Photo → scaled, 3D-printable mesh (Hunyuan3D-2.1)
---

# photo-to-mesh — demo Space

Single-image demo of [photo-to-mesh](https://github.com/Hasasasaki/photo-to-mesh):
upload a photo of an object, generate a mesh, set one real dimension, and download
a **GLB in metres** (the glTF standard) or a **slicer-ready STL in mm**.

The full project runs locally and adds multi-view fusion (Hunyuan3D-2mv), SAM 3
text-prompted object isolation, and photogrammetric reconstruction routes:
**https://github.com/Hasasasaki/photo-to-mesh**

Model: [tencent/Hunyuan3D-2.1](https://huggingface.co/tencent/Hunyuan3D-2.1).
The MIT license above covers this Space's code only — the model weights are under
the **Tencent Hunyuan Community License** (usage and territory restrictions apply;
review it before reusing outputs commercially).
