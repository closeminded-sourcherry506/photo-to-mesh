# Deploying the demo Space

Everything the Space needs is in this folder: `app.py`, `mesh_units.py`,
`requirements.txt`, and `README.md` (whose YAML frontmatter configures the
Space). The Hunyuan3D-2.1 engine is cloned automatically on first start, at
the same pinned revision as the main repo.

## Steps

1. **Create the Space**: huggingface.co → New Space → name `photo-to-mesh`,
   SDK **Gradio**, public.
2. **Hardware**: Settings → Hardware → **ZeroGPU**. ZeroGPU is free at
   runtime for public Spaces but creating one requires a PRO account (or an
   approved community grant). Fallback: any paid GPU tier works; CPU does not.
3. **Upload the files** — either drag the four files into the web editor, or:

   ```bash
   hf upload <your-username>/photo-to-mesh space/ . --repo-type=space
   ```

   (run from the repo root; requires `huggingface-cli login`)
4. **First run**: the build installs requirements (~minutes); the first
   generate downloads the ~8 GB shape weights into the Space's HF cache.
   Subsequent generates take roughly a minute at the default settings.
5. **Add the badge** to the main repo README once it works:

   ```markdown
   [![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/<your-username>/photo-to-mesh)
   ```

## Notes

- **Model license**: the weights are under the Tencent Hunyuan Community
  License (usage + territory restrictions). Tencent hosts public demos of
  these models themselves, but review the license before publishing yours.
- **Why no SAM 3 / multi-view here**: SAM 3 weights are gated (every visitor
  would hit the gate), and 2mv adds ~5 GB + load time. The Space's job is the
  30-second "wow", then it links visitors to the full local project.
- If the build fails on a requirement, check the pins against
  `Hunyuan3D-2.1/requirements.txt` at the pinned revision — the list here is
  that file minus torch and the texture-stage packages.
