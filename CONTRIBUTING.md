# Contributing

Thanks for helping! The single most valuable contribution right now is a
**[GPU compatibility report](../../issues/new?template=gpu_report.yml)** —
the stack has been validated on one card (RTX 5090) and needs data points
from other GPUs. It takes two minutes.

## Dev setup

```bash
./setup.sh core            # uv venv + deps (no sudo)
./setup.sh routeC          # only needed to run the GUI / Hunyuan engine
```

## Checks (same as CI)

```bash
uv run --group dev ruff check .
uv run --group dev pytest tests/ -q
shellcheck setup.sh run_gui.sh run_pipeline.sh routeB_2dgs/run_2dgs.sh
```

## Guidelines

- Match the existing style — compact, with comments explaining *why*, not
  *what*. The ruff config in `pyproject.toml` is the arbiter.
- Keep PRs small and focused. If a change touches GPU code paths, say which
  card you tested it on.
- For new features or new reconstruction routes, open an issue first so we
  can agree on scope before you build it.
