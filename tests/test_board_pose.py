"""Geometric recovery and frame-convention tests for the vision algorithms."""

import cv2
import numpy as np
import pytest

from core.algorithms.board_pose import (
    calibrate_hand_eye,
    checkerboard_points,
    solve_board_pose,
)


def transform(rotvec, translation):
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(np.asarray(rotvec, dtype=float))[0]
    pose[:3, 3] = translation
    return pose


@pytest.mark.parametrize("reverse_corners", [False, True])
def test_pnp_recovers_board_pose_in_mm_regardless_of_detector_order(reverse_corners):
    matrix = np.array([[1200, 0, 640], [0, 1200, 480], [0, 0, 1]], dtype=float)
    expected = transform([0.2, -0.3, 0.1], [12, -8, 450])
    corners = cv2.projectPoints(
        checkerboard_points((9, 6), 10), cv2.Rodrigues(expected[:3, :3])[0],
        expected[:3, 3], matrix, np.zeros(5),
    )[0].reshape(-1, 2)
    if reverse_corners:
        corners = corners[::-1]
    result = solve_board_pose(corners, matrix, np.zeros(5), (9, 6), 10)
    np.testing.assert_allclose(result["vision_pose"], expected, atol=1e-6)
    assert result["reprojection_error_px"] < 1e-6
    assert result["translation_unit"] == "mm"


def test_reference_rotation_keeps_board_identity_across_image_order_boundary():
    matrix = np.array([[1200, 0, 640], [0, 1200, 480], [0, 0, 1]], dtype=float)
    expected = transform([0.1, 0.2, 2.8], [0, 0, 450])
    corners = cv2.projectPoints(
        checkerboard_points((9, 6), 10), cv2.Rodrigues(expected[:3, :3])[0],
        expected[:3, 3], matrix, np.zeros(5),
    )[0].reshape(-1, 2)
    result = solve_board_pose(
        corners, matrix, np.zeros(5), (9, 6), 10,
        reference_rotation=expected[:3, :3],
    )
    np.testing.assert_allclose(result["vision_pose"], expected, atol=1e-6)
    assert result["corner_order"] == "reference_rotation_reversed"
    assert result["reprojection_error_px"] < 1e-6


@pytest.mark.parametrize("method", ["PARK", "TSAI"])
def test_hand_eye_recovers_camera_to_tool_not_its_inverse(method):
    rng = np.random.default_rng(12)
    hand_eye = transform([0.1, -0.2, 0.05], [20, -80, 70])
    target = transform([0.2, 0.1, 0.5], [300, 200, 100])
    robot = np.array([
        transform(rng.normal(0, 0.4, 3), rng.normal(0, 100, 3)) for _ in range(20)
    ])
    board = np.array([np.linalg.inv(pose @ hand_eye) @ target for pose in robot])
    result = calibrate_hand_eye(robot, board, method)
    np.testing.assert_allclose(result["hand_eye"], hand_eye, atol=1e-7)
    np.testing.assert_allclose(result["target_pose_base"], target, atol=1e-7)
    assert result["translation_rms_mm"] < 1e-7
    assert result["rotation_rms_deg"] < 1e-5
    assert result["sample_count"] == 20


def test_hand_eye_rejects_translation_only_samples():
    robot = np.array([transform([0, 0, 0], [i, 0, 0]) for i in range(4)])
    with pytest.raises(ValueError, match="非平行轴"):
        calibrate_hand_eye(robot, np.linalg.inv(robot))
