from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.calibration.bayesian_calibration_pipeline.core.data_io import load_dataset


def load_calibration_data(path: str | Path) -> dict[str, Any] | pd.DataFrame:
    data_path = Path(path)
    suffix = data_path.suffix.lower()
    if suffix == ".pkl":
        return load_dataset(data_path)
    if suffix == ".csv":
        return pd.read_csv(data_path)
    raise ValueError(f"Unsupported calibration data format: {data_path.suffix}")
