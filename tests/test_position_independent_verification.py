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
    patent = report["patent_v6"]["summary"]
    np.testing.assert_allclose(patent["baseline_scatter"], [*[0.2 / np.sqrt(3)] * 3, 0.2])
    np.testing.assert_allclose(patent["repeatability"], [*[0.4 / np.sqrt(3)] * 3, 0.4])
    np.testing.assert_allclose(patent["absolute_change"], 0, atol=1e-13)
    np.testing.assert_allclose(patent["repeatability_change"], patent["baseline_scatter"])


def test_spatial_scatter_is_translation_and_rotation_invariant():
    points = np.array([[1, 3, 5], [2, 4, 6], [-3, 0, 5], [2, -2, 4]])
    rotation = np.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])
    before = verification.scatter(points)
    after = verification.scatter(points @ rotation.T + [100, 200, -500])
    np.testing.assert_allclose(before[3], after[3])
    np.testing.assert_allclose(after[:3], before[[1, 2, 0]])
    before_rms = verification.rms_scatter(points)
    after_rms = verification.rms_scatter(points @ rotation.T + [100, 200, -500])
    np.testing.assert_allclose(before_rms[3], after_rms[3])
    np.testing.assert_allclose(after_rms[:3], before_rms[[1, 2, 0]])


def test_fixed_point_bias_changes_absolute_error_but_not_repeatability():
    directions = np.vstack([np.eye(3), -np.eye(3)])
    initial, current, targets = {}, {}, {}
    biases = {"P001": np.array([0.3, 0, 0]), "P002": np.array([0, -0.2, 0.1])}
    for point, bias in biases.items():
        for index, direction in enumerate(directions, 1):
            key = (point, f"D{index:03d}")
            targets[key] = np.array([10, 20, 30])
            initial[key] = targets[key] + 0.4 * direction
            current[key] = initial[key] + bias
    report = verification.metrics(initial, current, targets)
    np.testing.assert_allclose(report["summary"]["repeatability_change"], 0, atol=1e-13)
    assert report["summary"]["absolute_change"][3] > 0
    for point, bias in biases.items():
        np.testing.assert_allclose(report["groups"][point]["centroid_change_xyz_mm"], bias, atol=1e-13)
        patent = report["patent_v6"]["groups"][point]
        np.testing.assert_allclose(patent["repeatability_change"], 0, atol=1e-13)
        np.testing.assert_allclose(patent["absolute_change"], [*np.abs(bias), np.linalg.norm(bias)])


def test_unbalanced_directions_follow_centroid_rms_identity():
    directions = np.eye(3)
    radius = 0.2
    measured = verification.rms_scatter(radius * directions + [10, 20, 30])
    # 三个非对称方向的质心不在生成球心；中心化 RMS = r sqrt(1 - |mean(d)|²)。
    expected = radius * np.sqrt(1 - np.dot(directions.mean(axis=0), directions.mean(axis=0)))
    np.testing.assert_allclose(measured[3], expected)
    assert measured[3] < radius
    np.testing.assert_allclose(measured[:3], radius * np.sqrt(2) / 3)


def test_point_centroid_shift_magnitudes_do_not_cancel_across_points():
    initial, current, targets = {}, {}, {}
    for point, bias in (("P001", np.array([0.3, 0, 0])), ("P002", np.array([-0.3, 0, 0]))):
        for index, direction in enumerate(np.vstack([np.eye(3), -np.eye(3)]), 1):
            key = (point, f"D{index:03d}")
            targets[key] = np.zeros(3)
            initial[key] = 0.2 * direction
            current[key] = initial[key] + bias
    summary = verification.metrics(initial, current, targets)["patent_v6"]["summary"]
    np.testing.assert_allclose(summary["centroid_change_xyz_mm"], 0, atol=1e-13)
    np.testing.assert_allclose(summary["absolute_change"], [0.3, 0, 0, 0.3], atol=1e-13)
