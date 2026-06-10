from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


if getattr(sys, "frozen", False):
    resource_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    if resource_root.name.startswith("_MEI"):
        runtime_root = Path(sys.executable).resolve().parent / "ZMSoftware_data"
        for relative in ("config", "models", "app/resources", "data", "storage"):
            source = resource_root / relative
            target = runtime_root / relative
            if not source.exists():
                continue
            if not target.exists():
                shutil.copytree(source, target)
                continue
            for item in source.rglob("*"):
                if item.is_dir():
                    continue
                destination = target / item.relative_to(source)
                if not destination.exists():
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, destination)
        os.chdir(runtime_root)
    else:
        os.chdir(resource_root)
