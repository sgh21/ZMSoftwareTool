from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    config: Path
    models: Path
    data: Path
    storage: Path

    @classmethod
    def from_root(cls, root: str | Path) -> "ProjectPaths":
        base = Path(root).resolve()
        return cls(
            root=base,
            config=base / "config",
            models=base / "models",
            data=base / "data",
            storage=base / "storage",
        )
