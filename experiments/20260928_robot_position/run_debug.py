"""Standalone entry point for the legacy robot-position debug manifests."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.robot_position_debug import main  # noqa: E402


if __name__ == "__main__":
    main()
