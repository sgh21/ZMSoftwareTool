"""多方向各一次的工程散布：独立位置真值构造视觉观测并核对解析结果。"""

import json

import numpy as np
import pytest

from core.algorithms.position_monitoring import evaluate_multidirectional


def transform(position=(0, 0, 0), angle=0):
    cosine, sine = np.cos(angle), np.sin(angle)
    matrix = np.eye(4)
    matrix[:3, :3] = [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]]
    matrix[:3, 3] = position
    return matrix


TARGET = transform((350, 200, 700), -0.3)


def observations(poses, *, hand_eye=None, point="P001", ideal=None):
    hand_eye = np.eye(4) if hand_eye is None else hand_eye
    ideal = transform() if ideal is None else ideal
    return [{
        "point_id": point,
        "direction_id": f"D{index:03d}",
        "vision_pose": np.linalg.inv(hand_eye) @ np.linalg.inv(pose) @ TARGET,
        "ideal_pose": ideal.copy(),
    } for index, pose in enumerate(poses, 1)]


def symmetric_observations(radius, **kwargs):
    return observations([transform((-radius, 0, 0)), transform((radius, 0, 0))], **kwargs)


def test_two_period_scatter_and_absolute_degradation_preserve_direction_errors():
    baseline = symmetric_observations(0.2)
    current = symmetric_observations(0.4)
    result = evaluate_multidirectional(
        current, np.eye(4), baseline_samples=baseline, target_pose_base=TARGET
    )
    group, summary = result["groups"][0], result["summary"]
    assert result["sampling_protocol"] == "multidirectional"
    assert result["is_standard_repeatability"] is False
    assert group["direction_id"] == "多方向"
    assert group["baseline"]["count"] == group["current"]["count"] == 2
    assert summary["rp_baseline"] == pytest.approx(0.2)
    assert summary["rp_current"] == pytest.approx(0.4)
    assert summary["rp_change"] == pytest.approx(0.2)
    assert summary["axis_3sigma_base"] == pytest.approx([1.2 * np.sqrt(2), 0, 0], abs=1e-10)
    assert summary["axis_3sigma_change_base"] == pytest.approx([0.6 * np.sqrt(2), 0, 0], abs=1e-10)
    assert summary["absolute_ap"] == pytest.approx(0.4)
    assert summary["absolute_ap_change"] == pytest.approx(0.2)
    assert summary["absolute_axis_change"] == pytest.approx([0.2, 0, 0], abs=1e-10)
    assert group["current_error_base"] == pytest.approx([0, 0, 0], abs=1e-10)
    for detail, sign in zip(group["directions"], (-1, 1)):
        assert detail["initial_error_base"] == pytest.approx([sign * 0.2, 0, 0], abs=1e-10)
        assert detail["current_error_base"] == pytest.approx([sign * 0.4, 0, 0], abs=1e-10)
        assert detail["absolute_ap_change"] == pytest.approx(0.2)
    json.dumps(result, allow_nan=False)


def test_absolute_degradation_can_improve_and_is_not_displacement_magnitude():
    result = evaluate_multidirectional(
        symmetric_observations(0.2), np.eye(4),
        baseline_samples=symmetric_observations(0.4), target_pose_base=TARGET,
    )
    assert result["summary"]["absolute_ap_change"] == pytest.approx(-0.2)
    assert result["summary"]["absolute_axis_change"] == pytest.approx([-0.2, 0, 0], abs=1e-10)
    assert result["summary"]["drift_distance"] == pytest.approx(0.2)
    assert result["summary"]["rp_change"] == pytest.approx(-0.2)


def test_radial_scatter_uses_sample_standard_deviation_for_more_than_two_directions():
    current = observations([transform((x, 0, 0)) for x in (-1, 0, 1)])
    result = evaluate_multidirectional(current, np.eye(4))
    assert result["summary"]["rp_current"] == pytest.approx(2 / 3 + np.sqrt(3))
    assert result["groups"][0]["current"]["axis_3sigma_local"] == pytest.approx([3, 0, 0], abs=1e-10)


def test_rotating_tcp_and_hand_eye_lever_arm_do_not_create_false_position_scatter():
    hand_eye = transform((100, -50, 40), 0.4)
    ideal = transform((20, 30, 40), 0.3)
    baseline = observations([ideal, transform((20, 30, 40), 0.9)], hand_eye=hand_eye, ideal=ideal)
    current = observations([
        transform((20, 30, 40), 1.2), transform((20, 30, 40), -0.5)
    ], hand_eye=hand_eye, ideal=ideal)
    for target in (None, TARGET):
        result = evaluate_multidirectional(
            current, hand_eye, baseline_samples=baseline,
            base_rotations={"P001": ideal[:3, :3]}, target_pose_base=target,
        )
        assert result["summary"]["rp_current"] == pytest.approx(0, abs=1e-10)
        assert result["summary"]["rp_baseline"] == pytest.approx(0, abs=1e-10)
        assert result["summary"]["drift_distance"] == pytest.approx(0, abs=1e-10)
        assert result["summary"]["axis_3sigma_base"] == pytest.approx([0, 0, 0], abs=1e-10)


def test_q_rotates_all_samples_and_a_recovers_axes_without_q():
    angle = np.pi / 2
    hand_eye = transform((80, 20, 10), 0.2)
    ideal = transform((100, 0, 200), angle)
    baseline = observations([
        transform((100, y, 200), angle) for y in (-0.2, 0.2)
    ], hand_eye=hand_eye, ideal=ideal)
    current = observations([
        transform((100, y, 200), angle + 0.1) for y in (0.6, 1.4)
    ], hand_eye=hand_eye, ideal=ideal)
    relative = evaluate_multidirectional(
        current, hand_eye, baseline_samples=baseline,
        base_rotations={"P001": ideal[:3, :3]},
    )
    absolute = evaluate_multidirectional(
        current, hand_eye, baseline_samples=baseline, target_pose_base=TARGET,
    )
    for result in (relative, absolute):
        assert result["summary"]["drift_base"] == pytest.approx([0, 1, 0], abs=1e-10)
        assert result["summary"]["axis_3sigma_base"] == pytest.approx([0, 1.2 * np.sqrt(2), 0], abs=1e-10)
        assert result["summary"]["rp_current"] == pytest.approx(0.4)
    assert relative["summary"]["absolute_ap"] is None
    assert relative["summary"]["absolute_ap_change"] is None
    assert relative["groups"][0]["current"]["mean_position_base"] is None
    assert absolute["groups"][0]["current"]["mean_position_base"] == pytest.approx([100, 1, 200])
    assert absolute["summary"]["absolute_ap_change"] == pytest.approx(0.8)


def test_missing_q_keeps_spatial_scatter_without_claiming_base_axes():
    result = evaluate_multidirectional(
        symmetric_observations(0.4), np.eye(4), baseline_samples=symmetric_observations(0.2)
    )
    assert result["summary"]["rp_current"] == pytest.approx(0.4)
    assert result["summary"]["rp_change"] == pytest.approx(0.2)
    assert result["summary"]["axis_3sigma_base"] is None
    assert result["summary"]["drift_base"] is None
    assert result["summary"]["absolute_ap_change"] is None
    assert any("缺少参考末端朝向 Q" in warning for warning in result["warnings"])


def test_current_only_does_not_fabricate_a_zero_baseline_or_zero_degradation():
    result = evaluate_multidirectional(symmetric_observations(0.2), np.eye(4), target_pose_base=TARGET)
    group, summary = result["groups"][0], result["summary"]
    assert group["baseline"] is None
    assert summary["rp_current"] == pytest.approx(0.2)
    assert summary["absolute_ap"] == pytest.approx(0.2)
    for name in ("rp_baseline", "rp_change", "absolute_ap_change", "absolute_axis_change"):
        assert summary[name] is None
    assert group["directions"][0]["initial_error_base"] is None
    json.dumps(result, allow_nan=False)


def test_one_direction_has_no_scatter_but_can_have_absolute_error():
    result = evaluate_multidirectional(
        observations([transform((3, 4, 0))]), np.eye(4), target_pose_base=TARGET
    )
    assert result["summary"]["rp_current"] is None
    assert result["summary"]["axis_3sigma_base"] is None
    assert result["summary"]["absolute_ap"] == pytest.approx(5)
    assert any("少于 2 个接近方向" in warning for warning in result["warnings"])
    json.dumps(result, allow_nan=False)


def test_absolute_summary_averages_directions_then_weights_points_equally():
    baseline = observations([transform(), transform()])
    baseline += observations([transform()] * 3, point="P002")
    current = symmetric_observations(1)
    current += observations([transform((3, 0, 0))] * 3, point="P002")
    result = evaluate_multidirectional(
        current, np.eye(4), baseline_samples=baseline, target_pose_base=TARGET
    )
    assert len(result["groups"]) == 2
    assert result["summary"]["absolute_ap_change"] == pytest.approx(2)
    assert result["summary"]["absolute_axis_change"] == pytest.approx([2, 0, 0], abs=1e-10)
    assert result["summary"]["absolute_axis"] == pytest.approx([2, 0, 0], abs=1e-10)


def test_duplicate_direction_and_cross_period_missing_direction_are_rejected():
    current = symmetric_observations(0.2)
    with pytest.raises(ValueError, match="每方向只允许一条"):
        evaluate_multidirectional(current + [current[0]], np.eye(4))
    with pytest.raises(ValueError, match="测点/接近方向不匹配"):
        evaluate_multidirectional(current, np.eye(4), baseline_samples=current[:1])
    with pytest.raises(ValueError, match="没有观测数据"):
        evaluate_multidirectional([], np.eye(4))


@pytest.mark.parametrize("changed_pose", [transform((1, 0, 0)), transform(angle=0.1)])
def test_fixed_point_ideal_pose_is_checked_within_and_across_periods(changed_pose):
    baseline, current = symmetric_observations(0.2), symmetric_observations(0.4)
    current[1]["ideal_pose"] = changed_pose
    with pytest.raises(ValueError, match="各方向的理想位姿不一致"):
        evaluate_multidirectional(current, np.eye(4))
    current[0]["ideal_pose"] = changed_pose
    with pytest.raises(ValueError, match="两期理想位姿不一致"):
        evaluate_multidirectional(current, np.eye(4), baseline_samples=baseline)


def test_simulation_actual_and_truth_fields_are_never_inputs_to_metrics():
    current = symmetric_observations(0.4)
    expected = evaluate_multidirectional(current, np.eye(4), target_pose_base=TARGET)
    for sample in current:
        sample["actual_pose"] = "not a valid transform"
        sample["actual"] = object()
        sample["truth"] = {"position": [1e9, -1e9, 0]}
    actual = evaluate_multidirectional(current, np.eye(4), target_pose_base=TARGET)
    assert actual == expected
    for sample in current:
        sample.pop("ideal_pose")
    missing_ideal = evaluate_multidirectional(current, np.eye(4), target_pose_base=TARGET)
    assert missing_ideal["summary"]["absolute_ap"] is None
    assert missing_ideal["summary"]["rp_current"] == pytest.approx(0.4)
