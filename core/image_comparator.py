from __future__ import annotations

from pathlib import Path

import numpy as np


def mean_absolute_image_difference(before_path: str | Path, after_path: str | Path) -> float:
    import cv2

    before = cv2.imread(str(before_path), cv2.IMREAD_GRAYSCALE)
    after = cv2.imread(str(after_path), cv2.IMREAD_GRAYSCALE)
    if before is None or after is None:
        raise ValueError("Unable to read one or both images")
    if before.shape != after.shape:
        after = cv2.resize(after, (before.shape[1], before.shape[0]))
    return float(np.mean(np.abs(before.astype(float) - after.astype(float))))
