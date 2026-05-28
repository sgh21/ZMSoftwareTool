from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RobotModelFiles:
    urdf: Path | None
    xacro: Path | None
    mesh_dir: Path


def discover_robot_model(model_root: str | Path) -> RobotModelFiles:
    root = Path(model_root)
    urdf = next((path for path in sorted((root / "urdf").glob("*.urdf"))), None)
    xacro = next((path for path in sorted((root / "urdf").glob("*.xacro"))), None)
    return RobotModelFiles(urdf=urdf, xacro=xacro, mesh_dir=root / "meshes")
