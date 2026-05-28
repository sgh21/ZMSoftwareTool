from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VerificationPlan:
    pose_config: Path
    image_count: int
    output_dir: Path
