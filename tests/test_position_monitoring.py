"""独立刚体几何和解析统计样例，验证监控指标的含义。"""

import json

import numpy as np
import pytest

from core.algorithms.position_monitoring import (
    evaluate_position_monitoring,
    validate_transform,
)


def transform(position=(0, 0, 0), rotation=None):
    matrix = np.eye(4)
    matrix[:3, 3] = position
    if rotation is not None:
        matrix[:3, :3] = rotation
    return matrix


def rz(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def observations(poses, hand_eye=None, *, point="P1", direction="D1"):
    """由独立的基座真值 X 和固定靶标 A 生成 V=H^-1 X^-1 A。"""
    hand_eye = np.eye(4) if hand_eye is None else hand_eye
    target = transform((350, 200, 700), rz(-0.3))
    return [{
        "point_id": point,
        "direction_id": direction,
        "vision_pose": np.linalg.inv(hand_eye) @ np.linalg.inv(pose) @ target,
        "source": "synthetic ground truth",
    } for pose in poses]


def test_hand_eye_offset_does_not_turn_tcp_rotation_into_position_drift():
    hand_eye = transform((100, -50, 40), rz(0.4))
    initial = observations([transform((20, 30, 40), rz(0.3))], hand_eye)
    current = observations([transform((20, 30, 40), rz(0.9))], hand_eye)
    result = evaluate_position_monitoring(initial, current, hand_eye)
    assert result["groups"][0]["drift_local"] == pytest.approx([0, 0, 0], abs=1e-10)
    assert result["groups"][0]["drift_distance"] == pytest.approx(0, abs=1e-10)


def test_rp_uses_radial_sample_standard_deviation_and_rotates_axis_statistics():
    orientation = rz(np.pi / 2)
    # 参考末端 X 对应基座 Y；实际样本沿基座 Y 变化。
    initial = observations([
        transform((100, y, 200), orientation) for y in (-1, 0, 1)
    ])
    current = observations([
        transform((100, y, 200), orientation) for y in (-1, 1, 3)
    ])
    result = evaluate_position_monitoring(
        initial, current, np.eye(4), base_rotations={"P1": orientation}
    )
    group = result["groups"][0]
    expected_rp = 2 / 3 + np.sqrt(3)
    assert group["baseline"]["rp"] == pytest.approx(expected_rp)
    assert group["current"]["rp"] == pytest.approx(2 * expected_rp)
    assert group["rp_change"] == pytest.approx(expected_rp)
    assert group["baseline"]["mean_local"] == pytest.approx([1, 0, 0], abs=1e-10)
    assert group["baseline"]["mean_base"] == pytest.approx([0, 1, 0], abs=1e-10)
    assert group["current"]["mean_base"] == pytest.approx([0, 2, 0], abs=1e-10)
    assert group["baseline"]["axis_3sigma_local"] == pytest.approx([3, 0, 0], abs=1e-10)
    assert group["current"]["axis_3sigma_base"] == pytest.approx([0, 6, 0], abs=1e-10)
    assert group["axis_3sigma_change_base"] == pytest.approx([0, 3, 0], abs=1e-10)
    assert group["drift_local"] == pytest.approx([1, 0, 0], abs=1e-10)
    assert group["drift_base"] == pytest.approx([0, 1, 0], abs=1e-10)


def test_unknown_q_preserves_local_results_without_inventing_base_axes():
    initial = observations([transform(), transform((1, 0, 0))])
    current = observations([transform((0, 2, 0)), transform((1, 2, 0))])
    result = evaluate_position_monitoring(initial, current, np.eye(4))
    group = result["groups"][0]
    assert group["drift_local"] == pytest.approx([0, 2, 0], abs=1e-10)
    assert group["drift_distance"] == pytest.approx(2)
    assert group["drift_base"] is None
    assert group["baseline"]["mean_base"] is None
    assert group["current"]["axis_3sigma_base"] is None
    assert group["baseline"]["rp"] == pytest.approx(0.5)
    assert result["summary"]["mean_abs_drift_base"] is None
    assert any("缺少参考末端朝向 Q" in warning for warning in result["warnings"])


@pytest.mark.parametrize("shift,expected_ap,expected_change", [
    (-1, 1, -1),
    (1, 3, 1),
    (-4, 2, 0),
])
def test_absolute_error_change_requires_initial_vector(shift, expected_ap, expected_change):
    initial = observations([transform(), transform()])
    current = observations([transform((shift, 0, 0)), transform((shift, 0, 0))])
    result = evaluate_position_monitoring(
        initial, current, np.eye(4), base_rotations={"P1": np.eye(3)},
        initial_errors={"P1": {"D1": [2, 0, 0]}},
    )
    group = result["groups"][0]
    assert group["absolute_ap"] == pytest.approx(expected_ap)
    assert group["absolute_ap_change"] == pytest.approx(expected_change)
    assert group["absolute_axis_change"] == pytest.approx([expected_change, 0, 0])
    assert group["current_error_base"] == pytest.approx([2 + shift, 0, 0])
    assert group["drift_distance"] == pytest.approx(abs(shift))


def test_q_alone_does_not_establish_absolute_accuracy():
    initial = observations([transform()])
    current = observations([transform((1, 2, 3))])
    result = evaluate_position_monitoring(
        initial, current, np.eye(4), base_rotations={"P1": np.eye(3)}
    )
    group = result["groups"][0]
    assert group["drift_base"] == pytest.approx([1, 2, 3])
    assert group["absolute_ap"] is None
    assert group["absolute_ap_change"] is None
    assert result["summary"]["absolute_ap"] is None


def test_one_arrival_supports_drift_but_not_repeatability_and_json_has_no_nan():
    initial = observations([transform()])
    current = observations([transform((3, 4, 0))])
    result = evaluate_position_monitoring(initial, current, np.eye(4))
    group = result["groups"][0]
    assert group["drift_distance"] == pytest.approx(5)
    assert group["baseline"]["rp"] is None
    assert group["current"]["rp"] is None
    assert group["rp_change"] is None
    assert group["baseline"]["axis_3sigma_local"] is None
    assert result["summary"]["rp_current"] is None
    assert any("不能计算重复性" in warning for warning in result["warnings"])
    json.dumps(result, allow_nan=False)


def test_repeated_identical_arrivals_give_zero_rp_without_false_absolute_accuracy():
    initial = observations([transform()] * 30)
    result = evaluate_position_monitoring(initial, initial, np.eye(4))
    group = result["groups"][0]
    assert group["baseline"]["rp"] == pytest.approx(0, abs=1e-10)
    assert group["current"]["axis_3sigma_local"] == pytest.approx([0, 0, 0], abs=1e-10)
    assert group["absolute_ap"] is None
    assert not any("非完整国标样本" in warning for warning in result["warnings"])


def test_period_group_mismatch_is_rejected_instead_of_silently_dropping_data():
    initial = observations([transform()], direction="D1")
    current = observations([transform()], direction="D2")
    with pytest.raises(ValueError, match="测点/接近方向不匹配"):
        evaluate_position_monitoring(initial, current, np.eye(4))


def test_multidirection_vap_uses_distinct_direction_means_and_shared_point_reference():
    initial = []
    current = []
    for index, position in enumerate(((0, 0, 0), (2, 0, 0), (0, 3, 0))):
        initial += observations([transform(position)] * 2, direction=f"D{index}")
        current += observations([transform(np.asarray(position) * 2)] * 2, direction=f"D{index}")
    result = evaluate_position_monitoring(initial, current, np.eye(4))
    point = result["points"][0]
    assert point["baseline_vap"] == pytest.approx(np.sqrt(13))
    assert point["current_vap"] == pytest.approx(2 * np.sqrt(13))
    assert point["vap_change"] == pytest.approx(np.sqrt(13))
    assert all(group["baseline"]["rp"] == pytest.approx(0, abs=1e-10) for group in result["groups"])


def test_summary_weights_each_point_equally_and_averages_magnitudes_before_axes():
    initial = []
    current = []
    for point, direction, shift in (("P1", "D1", 1), ("P1", "D2", -1), ("P2", "D1", 3)):
        initial += observations([transform()] * 2, point=point, direction=direction)
        current += observations([transform((shift, 0, 0))] * 2, point=point, direction=direction)
    result = evaluate_position_monitoring(
        initial, current, np.eye(4), base_rotations={"P1": np.eye(3), "P2": np.eye(3)}
    )
    assert result["summary"]["drift_base"] == pytest.approx([1.5, 0, 0])
    assert result["summary"]["mean_abs_drift_base"] == pytest.approx([2, 0, 0])
    assert result["summary"]["drift_distance"] == pytest.approx(2)


def test_summary_missing_rp_or_q_does_not_change_weights():
    initial = observations([transform()] * 2, point="P1")
    initial += observations([transform()], point="P2")
    result = evaluate_position_monitoring(
        initial, initial, np.eye(4), base_rotations={"P1": np.eye(3)}
    )
    assert result["summary"]["rp_baseline"] is None
    assert result["summary"]["rp_current"] is None
    assert result["summary"]["drift_base"] is None
    assert result["summary"]["drift_distance"] == pytest.approx(0, abs=1e-10)


@pytest.mark.parametrize("invalid", [
    np.zeros((4, 4)),
    np.diag([-1, 1, 1, 1]),
    np.diag([2, 1, 1, 1]),
    np.full((4, 4), np.nan),
])
def test_invalid_rigid_transform_is_rejected(invalid):
    with pytest.raises(ValueError):
        validate_transform(invalid)


def test_empty_period_is_rejected():
    with pytest.raises(ValueError, match="没有观测数据"):
        evaluate_position_monitoring([], observations([transform()]), np.eye(4))
