from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppContext:
    root_dir: Path
    config_dir: Path
    data_dir: Path
    model_dir: Path
    storage_dir: Path

    @classmethod
    def from_root(cls, root_dir: Path | str) -> "AppContext":
        root = Path(root_dir).resolve()
        return cls(
            root_dir=root,
            config_dir=root / "config",
            data_dir=root / "data",
            model_dir=root / "models",
            storage_dir=root / "storage",
        )
