"""Compat shim: the Hunyuan3D-2mv checkpoint's config.yaml targets the OLD
package name `hy3dgen.shapegen.*` (Hunyuan3D-2.0 repo). Our installed repo is
2.1, which renamed that package to `hy3dshape`. The class names are identical,
so we alias the old module paths onto the new package in sys.modules.

Import this BEFORE loading the 2mv pipeline.
"""
import sys
import types
import importlib

import hy3dshape
import hy3dshape.models
import hy3dshape.schedulers
import hy3dshape.preprocessors
import hy3dshape.pipelines

_pkg = types.ModuleType("hy3dgen")
_pkg.__path__ = []  # mark as package
sys.modules.setdefault("hy3dgen", _pkg)
sys.modules["hy3dgen.shapegen"] = hy3dshape
sys.modules["hy3dgen.shapegen.models"] = hy3dshape.models
sys.modules["hy3dgen.shapegen.schedulers"] = hy3dshape.schedulers
sys.modules["hy3dgen.shapegen.preprocessors"] = hy3dshape.preprocessors
sys.modules["hy3dgen.shapegen.pipelines"] = hy3dshape.pipelines

# sanity: make sure the targeted attributes resolve
for _m, _c in [
    ("hy3dgen.shapegen.models", "Hunyuan3DDiT"),
    ("hy3dgen.shapegen.preprocessors", "MVImageProcessorV2"),
]:
    assert hasattr(importlib.import_module(_m), _c), f"shim failed for {_m}.{_c}"
