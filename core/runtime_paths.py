"""区分只读发布资源与可写业务数据；源码运行沿用工程目录。"""

import os
from pathlib import Path
import sys


def bundle_root() -> Path:
    """PyInstaller 的资源随程序分发，不受当前工作目录影响。"""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]


def data_root() -> Path:
    """发布版写入当前用户目录，可通过环境变量指定数据位置。"""
    override = os.environ.get("ZMSOFTWARE_DATA_DIR")
    if override:
        return Path(override).expanduser().resolve()
    if not getattr(sys, "frozen", False):
        return bundle_root()
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "SoftwareTools_PyQt"
