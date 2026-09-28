"""独立轴旋转和等价姿态检查参数表单的角度约定。"""

import numpy as np
import pytest

from core.algorithms.pose_fields import (
    rotation_from_rpy_degrees,
    rotation_to_rpy_degrees,
)


def test_mixed_rpy_rotates_about_fixed_xyz_axes():
    roll, pitch, yaw = np.deg2rad([30, -45, 60])
    rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)],
    ])
    ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)],
    ])
    rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1],
    ])
    expected = rz @ ry @ rx
    np.testing.assert_allclose(rotation_from_rpy_degrees([30, -45, 60]), expected)
    np.testing.assert_allclose(rotation_to_rpy_degrees(expected), [30, -45, 60])


@pytest.mark.parametrize("pitch", [90, -90, 89.999999, -89.999999, 90.000001, -90.000001])
def test_vertical_and_near_vertical_pitch_preserves_rotation(pitch):
    original = rotation_from_rpy_degrees([37, pitch, -124])
    rpy = rotation_to_rpy_degrees(original)
    assert np.isfinite(rpy).all()
    assert abs(rpy[1]) <= 90
    if abs(pitch) == 90:
        assert rpy[0] == 0
    np.testing.assert_allclose(rotation_from_rpy_degrees(rpy), original, atol=1e-12)


@pytest.mark.parametrize("rpy", [
    [0, 0, 0], [180, 0, 0], [0, 180, 0], [0, 0, 180],
    [-180, 0, -180], [180, 90, 180], [-180, -90, -180],
])
def test_half_turn_boundaries_preserve_rotation(rpy):
    original = rotation_from_rpy_degrees(rpy)
    restored = rotation_from_rpy_degrees(rotation_to_rpy_degrees(original))
    np.testing.assert_allclose(restored, original, atol=1e-12)


def test_random_rotations_round_trip_with_canonical_pitch():
    for angles in np.random.default_rng(24).uniform(-360, 360, (64, 3)):
        original = rotation_from_rpy_degrees(angles)
        restored_angles = rotation_to_rpy_degrees(original)
        assert abs(restored_angles[1]) <= 90
        np.testing.assert_allclose(
            rotation_from_rpy_degrees(restored_angles), original, atol=1e-12,
        )


@pytest.mark.parametrize("rpy", [
    [], [0, 0], [0, 0, 0, 0], [[0, 0, 0]],
    [0, np.nan, 0], [np.inf, 0, 0], [0, 0, -np.inf], ["roll", 0, 0],
])
def test_invalid_rpy_is_rejected_with_field_description(rpy):
    with pytest.raises(ValueError, match="RPY.*三个有限数值"):
        rotation_from_rpy_degrees(rpy)


@pytest.mark.parametrize("rotation, message", [
    (np.eye(4), "3×3"),
    ([[1, 0, 0]], "3×3"),
    (np.full((3, 3), np.nan), "有限数值"),
    (np.full((3, 3), np.inf), "有限数值"),
    ([["x"] * 3] * 3, "有限数值"),
    (np.zeros((3, 3)), "不正交"),
    (np.diag([1, 1, 2]), "不正交"),
    (np.diag([1, 1, -1]), "行列式"),
])
def test_invalid_rotation_is_rejected(rotation, message):
    with pytest.raises(ValueError, match=message):
        rotation_to_rpy_degrees(rotation)
