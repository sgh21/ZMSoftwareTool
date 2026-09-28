"""独立统计工具以解析球面点验证，不调用生产统计函数。"""

import importlib.util
from pathlib import Path

import numpy as np


path = Path(__file__).resolve().parents[1] / "experiments/20260928_position_end_to_end/verify_results.py"
spec = importlib.util.spec_from_file_location("independent_verification", path)
verification = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verification)


def test_symmetric_directions_have_expected_scatter_and_degradation():
    directions = np.vstack([np.eye(3), -np.eye(3)])
    initial, current, targets = {}, {}, {}
    for point, center in (("P001", [5, 10, 15]), ("P002", [-20, 25, 30])):
        for index, direction in enumerate(directions, 1):
            key = (point, f"D{index:03d}")
            targets[key] = np.asarray(center)
            initial[key] = targets[key] + 0.2 * direction
            current[key] = targets[key] + 0.4 * direction
    report = verification.metrics(initial, current, targets)
    np.testing.assert_allclose(report["summary"]["repeatability"], [*[3 * 0.4 * np.sqrt(2 / 5)] * 3, 0.4])
    np.testing.assert_allclose(report["summary"]["absolute_change"], [0.2 / 3] * 3 + [0.2])
    np.testing.assert_allclose(report["summary"]["repeatability_change"], report["summary"]["baseline_scatter"])


def test_spatial_scatter_is_translation_and_rotation_invariant():
    points = np.array([[1, 3, 5], [2, 4, 6], [-3, 0, 5], [2, -2, 4]])
    rotation = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
    before = verification.scatter(points)
    after = verification.scatter(points @ rotation.T + [100, 200, -500])
    np.testing.assert_allclose(before[3], after[3])
    np.testing.assert_allclose(after[:3], before[[1, 2, 0]])
