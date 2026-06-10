# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import site

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)


ROOT = Path.cwd()


def _collect_optional(
    package: str,
    *,
    include_submodules: bool = False,
) -> tuple[list[str], list[tuple[str, str]], list[tuple[str, str]]]:
    hiddenimports = []
    datas = []
    binaries = []
    if include_submodules:
        try:
            hiddenimports.extend(collect_submodules(package))
        except Exception:
            pass
    try:
        datas.extend(collect_data_files(package))
    except Exception:
        pass
    try:
        binaries.extend(collect_dynamic_libs(package))
    except Exception:
        pass
    return hiddenimports, datas, binaries


datas = [
    (str(ROOT / "config"), "config"),
    (str(ROOT / "models"), "models"),
    (str(ROOT / "app" / "resources"), "app/resources"),
]
for optional_dir in ("data", "storage"):
    path = ROOT / optional_dir
    if path.exists():
        datas.append((str(path), optional_dir))

binaries = []
hiddenimports = [
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
    "PySide6.QtOpenGL",
    "PySide6.QtOpenGLWidgets",
    "PySide6.QtSvg",
    "vtkmodules.all",
    "pyvista",
    "pyvistaqt",
    "yourdfpy",
    "trimesh",
    "collada",
    "xacro",
    "cv2",
]

for site_packages in site.getsitepackages():
    for mypyc_binary in sorted(Path(site_packages).glob("*__mypyc*.pyd")):
        binaries.append((str(mypyc_binary), "."))
        hiddenimports.append(mypyc_binary.name.split(".cp", 1)[0])

for package, include_submodules in (
    ("vtkmodules", True),
    ("pyvista", False),
    ("yourdfpy", False),
    ("trimesh", False),
    ("collada", False),
    ("sklearn", False),
    ("scipy", False),
    ("pandas", False),
    ("matplotlib", False),
    ("cv2", False),
):
    package_hiddenimports, package_datas, package_binaries = _collect_optional(
        package,
        include_submodules=include_submodules,
    )
    hiddenimports.extend(package_hiddenimports)
    datas.extend(package_datas)
    binaries.extend(package_binaries)


a = Analysis(
    ["main.py"],
    pathex=[str(ROOT)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[str(ROOT / "build_tools" / "hooks")],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "build_tools" / "pyinstaller_runtime_hook.py")],
    excludes=[
        "pytest",
        "tests",
        "scipy.tests",
        "scipy.stats.tests",
        "sklearn.tests",
        "matplotlib.tests",
        "pyvista.trame",
        "trame",
        "trame_vtk",
    ],
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="ZMSoftware",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
