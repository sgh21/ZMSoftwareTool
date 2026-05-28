from __future__ import annotations

from datetime import datetime
from pathlib import Path
from shutil import copy2


def backup_param_file(param_path: str | Path, backup_dir: str | Path) -> Path | None:
    source = Path(param_path)
    if not source.exists():
        return None
    target_dir = Path(backup_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = target_dir / f"{source.stem}_{timestamp}{source.suffix}"
    copy2(source, target)
    return target
