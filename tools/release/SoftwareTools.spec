# -*- mode: python ; coding: utf-8 -*-
"""Only application modules, runtime dependencies and explicit read-only resources."""
from pathlib import Path
from importlib.metadata import PackageNotFoundError

from PyInstaller.utils.hooks import copy_metadata

ROOT = Path(SPECPATH).resolve().parents[1]
EXCLUDES = [
    "tests", "debug", "experiments", "diagnostics", "tools", "pytest", "_pytest",
    "pywinauto", "comtypes", "IPython", "notebook", "jupyter", "matplotlib",
    "pandas", "sklearn", "tkinter", "PyQt5", "PySide2", "PySide6",
    "torch.testing", "torch.utils.benchmark", "torch.utils.tensorboard",
    "torch._dynamo.test_case", "torch._inductor.test_case", "torch._inductor.test_operators",
    "numpy.tests", "scipy.tests", "h5py.tests", "sympy.testing",
]
datas = []
for folder, extensions in (("config", {".json"}), ("resources", {".json", ".qss", ".svg", ".png", ".ico"})):
    for path in sorted((ROOT / folder).rglob("*")):
        if path.is_file() and path.suffix.lower() in extensions and "examples" not in path.parts:
            datas.append((str(path), str(path.parent.relative_to(ROOT))))
for package in ("PyQt6", "numpy", "scipy", "opencv-python", "opencv-python-headless", "h5py", "torch", "openpyxl"):
    try:
        datas += copy_metadata(package)
    except PackageNotFoundError:
        pass

a = Analysis(
    [str(ROOT / "main.py")], pathex=[str(ROOT)], binaries=[], datas=datas,
    hiddenimports=["scipy.special._cdflib", "openpyxl.cell._writer"],
    hookspath=[str(ROOT / "tools/release/hooks")], hooksconfig={},
    runtime_hooks=[], excludes=EXCLUDES, noarchive=False,
)

def runtime_file(entry):
    parts = entry[0].replace("\\", "/").lower().split("/")
    name = parts[-1]
    return not (set(parts) & {"tests", "test", "testing", "__pycache__", ".pytest_cache", ".git"}
                or name.startswith("test_") or name.endswith(("_test.py", ".pyi", ".h", ".hpp", ".cuh", ".lib")))

a.datas = [entry for entry in a.datas if runtime_file(entry)]
a.binaries = [entry for entry in a.binaries if runtime_file(entry)]
a.pure = [entry for entry in a.pure if entry[0] == "jinja2.tests" or (
          not set(entry[0].split(".")) & {"tests", "testing", "test"}
          and not entry[0].split(".")[-1].startswith("test_"))]
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name="SoftwareTools",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, disable_windowed_traceback=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="SoftwareTools")
