"""Build the executable on the host OS and record the exact build dependencies."""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[2]
    if platform.machine().lower() not in ("amd64", "x86_64"):
        raise SystemExit("This release targets x86_64 only.")
    if sys.version_info[:2] != (3, 12):
        raise SystemExit("Build with Python 3.12.")
    import torch

    if torch.__version__ != "2.9.0+cpu":
        raise SystemExit("Use the pinned PyTorch 2.9.0+cpu runtime for this release.")
    target = "windows" if sys.platform == "win32" else "linux"
    output = root / "dist/verification"
    output.mkdir(parents=True, exist_ok=True)
    names = ["PyQt6", "PyQt6-Qt6", "PyQt6-sip", "numpy", "scipy", "opencv-python", "opencv-python-headless",
             "PyYAML", "h5py", "torch", "openpyxl", "et_xmlfile", "filelock", "fsspec", "Jinja2", "MarkupSafe",
             "networkx", "sympy", "mpmath", "typing_extensions", "setuptools", "packaging", "pyinstaller",
             "pyinstaller-hooks-contrib", "altgraph", "pefile", "pywin32-ctypes"]
    versions = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    record = {
        "platform": platform.platform(), "architecture": platform.machine(), "python": sys.version,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "working_tree_modified": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()),
        "dependencies": versions, "build_succeeded": False,
        "clean_target_environment_tested": False,
    }
    subprocess.run([
        sys.executable, "-B", "-m", "PyInstaller", "--noconfirm", "--clean",
        "--distpath", "dist", "--workpath", "build/pyinstaller", "tools/release/SoftwareTools.spec",
    ], cwd=root, check=True)
    record["build_succeeded"] = True
    (output / f"build-{target}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    subprocess.run([sys.executable, "-B", "tools/release/audit_bundle.py", "dist/SoftwareTools"], cwd=root, check=True)


if __name__ == "__main__":
    main()
