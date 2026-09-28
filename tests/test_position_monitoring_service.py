"""服务层业务与持久化测试；输入、输出均隔离在 pytest 的 tmp_path。"""

from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import core.services.position_monitoring_service as service_module
from core.services.position_monitoring_service import (
    METRIC_LABELS,
    PositionMonitoringService,
    assess_metric,
    metric_values,
    read_document,
)


def transform(position=(0, 0, 0), rotvec=(0, 0, 0)):
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(np.asarray(rotvec, dtype=float))[0]
    pose[:3, 3] = position
    return pose


HAND_EYE = transform((40, -30, 80), (0.1, -0.2, 0.3))
TARGET = transform((500, 300, 900), (-0.2, 0.15, 0.3))
REFERENCE_ROTVEC = (0.1, 0.2, 0.5)


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def batch_file(path, positions, *, unit="mm", include_q=True, **metadata):
    """positions: [(point, direction, actual_base_position), ...]，一次记录一次到达。"""
    samples = []
    counts = defaultdict(int)
    rotations = {}
    for point, direction, position in positions:
        counts[point, direction] += 1
        robot = transform(position, REFERENCE_ROTVEC)
        vision = np.linalg.inv(HAND_EYE) @ np.linalg.inv(robot) @ TARGET
        rotations.setdefault(point, robot[:3, :3].tolist())
        if unit == "m":
            vision[:3, 3] /= 1000
            robot[:3, 3] /= 1000
        samples.append({
            "point_id": point,
            "direction_id": direction,
            "sample_id": str(counts[point, direction]),
            "vision_pose": vision.tolist(),
            "robot_pose": robot.tolist(),
        })
    document = {
        "length_unit": unit,
        "program_id": "same-target-program",
        "target_id": "fixed-target",
        "samples": samples,
        "base_rotations": rotations if include_q else {},
        **metadata,
    }
    return write_json(path, document)


def repeated(position=(100, 200, 300), *, point="P1", direction="D1", count=2):
    return [(point, direction, position)] * count


@pytest.fixture
def service(tmp_path):
    instance = PositionMonitoringService(root=tmp_path / "project")
    instance.save_parameters({"hand_eye": HAND_EYE.tolist(), "length_unit": "mm"})
    return instance


def establish_baseline(service, path, positions=None, **metadata):
    source = batch_file(path, repeated() if positions is None else positions, **metadata)
    service.load_observations(source)
    return service.create_baseline("initial reference")


def test_baseline_parameters_and_settings_survive_reopen_without_changing_source(service, tmp_path):
    source = batch_file(tmp_path / "sources" / "baseline.json", repeated())
    original_bytes = source.read_bytes()
    service.save_settings({
        "thresholds": {"X": 0.2, "Y": None, "Z": None, "distance": 0.4},
        "processing_points": [{"id": "work-point", "x": 1, "y": 2, "z": 3}],
    })
    imported = service.load_observations(source)
    baseline = service.create_baseline("initial reference")
    reopened = PositionMonitoringService(root=service.root)
    assert source.read_bytes() == original_bytes
    assert reopened.baseline["id"] == baseline["id"]
    assert reopened.baseline["batch"]["samples"] == imported["samples"]
    assert reopened.baseline["label"] == "initial reference"
    assert reopened.parameters["version"] == baseline["parameters"]["version"]
    assert reopened.settings == service.settings
    assert reopened.current_batch is None
    assert reopened.list_baselines()[0]["id"] == baseline["id"]
    assert Path(baseline["path"]).is_file()


def test_evaluation_saves_history_and_actual_observation_snapshot(service, tmp_path):
    baseline = establish_baseline(service, tmp_path / "sources" / "initial.json")
    source = batch_file(tmp_path / "sources" / "current.json", repeated((100.2, 199.7, 300.4)))
    original_bytes = source.read_bytes()
    imported = service.load_observations(source)
    result = service.evaluate()
    assert result["groups"][0]["drift_base"] == pytest.approx([0.2, -0.3, 0.4], abs=1e-9)
    assert result["summary"]["mean_abs_drift_base"] == pytest.approx([0.2, 0.3, 0.4], abs=1e-9)
    assert result["summary"]["absolute_ap"] is None
    assert result["status"] == "未设置阈值"
    assert result["baseline_id"] == baseline["id"]
    assert result["baseline_created_at"] == baseline["created_at"]
    assert source.read_bytes() == original_bytes
    snapshot = read_document(service.storage / "observations" / f"{result['id']}.json")
    assert snapshot["samples"] == imported["samples"]
    source.write_text("source changed after evaluation", encoding="utf-8")
    reopened = PositionMonitoringService(root=service.root)
    assert reopened.list_history(baseline_id=baseline["id"])[0] == result
    assert reopened.list_history(baseline_id="other-baseline") == []
    assert read_document(service.storage / "observations" / f"{result['id']}.json") == snapshot


def test_alarm_checks_each_group_even_when_overall_average_is_below_limit(service, tmp_path):
    positions = sum((repeated(point=point) for point in ("P1", "P2", "P3")), [])
    establish_baseline(service, tmp_path / "initial.json", positions, initial_errors={
        "P1": {"D1": [1, 0, 0]}, "P2": {"D1": [-1, 0, 0]}, "P3": {"D1": [1, 0, 0]},
    })
    current = (
        repeated((103, 200, 300), point="P1")
        + repeated((97, 200, 300), point="P2")
        + repeated((100, 200, 300), point="P3")
    )
    service.save_settings({"metric_thresholds": {"absolute_change": {"X": 2.5}}})
    service.load_observations(batch_file(tmp_path / "current.json", current))
    result = service.evaluate()
    assert result["summary"]["mean_abs_drift_base"][0] == pytest.approx(2)
    assert result["status"] == "超限"
    assert {(alarm["point_id"], alarm["axis"]) for alarm in result["alarms"]} == {
        ("P1", "X"), ("P2", "X"),
    }
    assert sorted(alarm["value"] for alarm in result["alarms"]) == pytest.approx([3, 3])
    assert {alarm["metric"] for alarm in result["alarms"]} == {"absolute_change"}


def test_parameter_update_preserves_previous_version_and_requires_matching_baseline(service, tmp_path):
    baseline = establish_baseline(service, tmp_path / "initial.json")
    old_version = service.parameters["version"]
    old_parameter_path = service.storage / "parameters" / f"{old_version}.json"
    old_parameter_bytes = old_parameter_path.read_bytes()
    current_path = batch_file(tmp_path / "current.json", repeated((101, 200, 300)))
    service.load_observations(current_path)
    changed = HAND_EYE.copy()
    changed[0, 3] += 1
    service.save_parameters({"hand_eye": changed.tolist()})
    assert service.current_batch is None
    assert service.parameters["version"] != old_version
    assert old_parameter_path.read_bytes() == old_parameter_bytes
    service.load_observations(current_path)
    with pytest.raises(ValueError, match="参数版本已变化"):
        service.evaluate()
    assert service.list_history() == []
    service.select_baseline(baseline["path"])
    assert service.parameters["version"] == old_version
    service.load_observations(current_path)
    assert service.evaluate()["groups"][0]["drift_base"] == pytest.approx([1, 0, 0], abs=1e-9)


@pytest.mark.parametrize("field", ["program_id", "target_id"])
def test_different_program_or_target_is_not_comparable(service, tmp_path, field):
    establish_baseline(service, tmp_path / "initial.json")
    path = batch_file(tmp_path / "current.json", repeated(), **{field: "different"})
    service.load_observations(path)
    with pytest.raises(ValueError, match=field):
        service.evaluate()
    assert service.list_history() == []


def test_missing_q_does_not_substitute_controller_rotation_or_claim_absolute_ap(service, tmp_path):
    establish_baseline(service, tmp_path / "initial.json", include_q=False)
    service.save_settings({"metric_thresholds": {"repeatability": {"X": 1}}})
    # robot_pose 在样本中存在，但没有明确 Q 时不能拿它自动充当实测基座朝向。
    service.load_observations(batch_file(tmp_path / "current.json", repeated((100, 202, 300))))
    result = service.evaluate()
    assert result["groups"][0]["drift_base"] is None
    assert result["groups"][0]["drift_distance"] == pytest.approx(2)
    assert result["summary"]["absolute_ap"] is None
    assert result["status"] == "部分指标不可判定"
    assert assess_metric(result, "repeatability")["status"] == "部分指标不可判定"
    assert result["alarms"] == []


def test_metre_observation_boundary_normalizes_all_translations_once(service, tmp_path):
    source = batch_file(
        tmp_path / "initial_m.json", repeated(), unit="m",
        initial_errors={"P1": {"D1": [0.002, 0, 0]}},
    )
    original = read_document(source)
    imported = service.load_observations(source)
    assert imported["length_unit"] == "mm"
    assert imported["initial_errors"]["P1"]["D1"] == [2, 0, 0]
    assert np.asarray(imported["samples"][0]["robot_pose"])[:3, 3] == pytest.approx([100, 200, 300])
    assert np.asarray(imported["samples"][0]["vision_pose"])[:3, 3] == pytest.approx(
        1000 * np.asarray(original["samples"][0]["vision_pose"])[:3, 3]
    )
    service.create_baseline()
    service.load_observations(batch_file(tmp_path / "current_m.json", repeated((99, 200, 300)), unit="m"))
    result = service.evaluate()
    assert result["summary"]["drift_base"] == pytest.approx([-1, 0, 0], abs=1e-9)
    assert result["summary"]["absolute_ap"] == pytest.approx(1)
    assert result["summary"]["absolute_ap_change"] == pytest.approx(-1)


def test_metre_hand_eye_parameters_and_legacy_import_do_not_change_sources(service, tmp_path):
    in_metres = HAND_EYE.copy()
    in_metres[:3, 3] /= 1000
    generic = write_json(tmp_path / "sources" / "parameters.json", {
        "hand_eye": in_metres.tolist(), "length_unit": "m", "transform_convention": "E_T_C",
    })
    legacy = write_json(tmp_path / "sources" / "legacy.json", {"T_tool_cam": in_metres.tolist()})
    original_bytes = {path: path.read_bytes() for path in (generic, legacy)}
    for path in (generic, legacy):
        parameters = service.load_parameters(path)
        assert parameters["length_unit"] == "mm"
        assert np.asarray(parameters["hand_eye"]) == pytest.approx(HAND_EYE)
        assert path.read_bytes() == original_bytes[path]


def test_partial_parameter_update_does_not_scale_previous_hand_eye_again(service):
    parameters = service.save_parameters({"length_unit": "m", "square_size_mm": 0.6})
    assert parameters["length_unit"] == "mm"
    assert np.asarray(parameters["hand_eye"]) == pytest.approx(HAND_EYE)


def test_duplicate_arrival_id_is_rejected(service, tmp_path):
    path = batch_file(tmp_path / "duplicate.json", repeated())
    document = read_document(path)
    document["samples"][1]["sample_id"] = document["samples"][0]["sample_id"]
    write_json(path, document)
    with pytest.raises(ValueError, match="重复样本编号"):
        service.load_observations(path)


@pytest.mark.parametrize("with_pnp_metadata", [False, True])
def test_hand_eye_calibration_is_serializable_and_does_not_replace_active_parameters(
    service, tmp_path, monkeypatch, with_pnp_metadata,
):
    rotvecs = [
        (0, 0, 0), (0.3, 0, 0), (0, -0.4, 0), (0, 0, 0.5),
        (0.2, -0.3, 0.1), (-0.25, 0.1, -0.4),
    ]
    samples = []
    for index, rotvec in enumerate(rotvecs):
        robot = transform((80 + index * 25, 200 - index * 10, 300 + index * 8), rotvec)
        vision = np.linalg.inv(HAND_EYE) @ np.linalg.inv(robot) @ TARGET
        samples.append({
            "point_id": str(index), "direction_id": "calibration", "sample_id": "1",
            "robot_pose": robot.tolist(), "vision_pose": vision.tolist(),
        })
        if with_pnp_metadata:
            samples[-1].update({
                "image_path": f"capture_{index}.png", "reprojection_error_px": 0.1 + index * 0.01,
                "corner_count": 100, "corner_order": "reference_rotation_original",
            })
    accepted_count = len(samples)
    if with_pnp_metadata:
        image = tmp_path / "rejected.bin"
        image.write_bytes(b"unclear board")
        samples.append({
            "point_id": "rejected", "direction_id": "calibration", "sample_id": "1",
            "robot_pose": transform().tolist(), "image_path": image.name,
        })
        service.save_parameters({"camera_matrix": [[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]})
        monkeypatch.setattr(service_module.cv2, "imdecode", lambda *_: np.zeros((20, 20), dtype=np.uint8))

        def reject_board(*args, **kwargs):
            raise ValueError("board not detected")

        monkeypatch.setattr(service_module, "estimate_board_pose", reject_board)
    path = write_json(tmp_path / "calibration.json", {
        "length_unit": "mm", "program_id": "calibration", "target_id": "fixed-target",
        "comparison_status": "calibration_only", "samples": samples,
    })
    parameters_before = deepcopy(service.parameters)
    source_before = path.read_bytes()
    imported = service.load_observations(path)
    result = service.calibrate_hand_eye()
    saved = read_document(result["path"])
    assert np.asarray(saved["hand_eye"]) == pytest.approx(HAND_EYE, abs=1e-7)
    assert saved["translation_rms_mm"] < 1e-7
    assert saved["sample_count"] == accepted_count
    snapshot = read_document(saved["input_batch_path"])
    assert snapshot == imported
    assert saved["parameters"] == parameters_before
    assert saved["rejected_samples"] == imported["rejected_samples"]
    if with_pnp_metadata:
        assert len(saved["pnp_summary"]) == accepted_count
        assert saved["pnp_summary"][0]["reprojection_error_px"] == pytest.approx(0.1)
        assert saved["rejected_samples"][0]["point_id"] == "rejected"
        assert saved["rejected_samples"][0]["reason"] == "board not detected"
    else:
        assert saved["pnp_summary"] is None
    assert service.parameters == parameters_before
    assert path.read_bytes() == source_before
    with pytest.raises(ValueError, match="手眼标定数据"):
        service.create_baseline()


def test_image_observation_arrays_are_serializable_at_service_boundary(service, tmp_path, monkeypatch):
    image = tmp_path / "sources" / "raw.bin"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"original image bytes")
    raw_before = image.read_bytes()
    source = write_json(image.parent / "images.json", {
        "length_unit": "mm", "program_id": "fixed", "target_id": "fixed-target",
        "samples": [{
            "point_id": "P1", "direction_id": "D1", "sample_id": "1", "image_path": image.name,
        }],
    })
    service.save_parameters({"camera_matrix": [[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]})
    monkeypatch.setattr(service_module.cv2, "imdecode", lambda *_: np.zeros((20, 20), dtype=np.uint8))
    monkeypatch.setattr(service_module, "estimate_board_pose", lambda *args, **kwargs: {
        "vision_pose": transform((10, 20, 300)), "reprojection_error_px": 0.1,
        "image_points": np.zeros((4, 2)), "projected_points": np.zeros((4, 2)),
    })
    service.load_observations(source)
    baseline = service.create_baseline()
    saved = read_document(baseline["path"])
    assert isinstance(saved["batch"]["samples"][0]["vision_pose"], list)
    assert "image_points" not in saved["batch"]["samples"][0]
    assert image.read_bytes() == raw_before


def test_mixed_batch_images_use_first_matrix_observation_as_board_reference(service, tmp_path, monkeypatch):
    image = tmp_path / "second.bin"
    image.write_bytes(b"image bytes")
    first_pose = transform((10, 20, 300), (0.1, -0.2, 1.7))
    source = write_json(tmp_path / "mixed.json", {
        "length_unit": "mm", "program_id": "fixed", "target_id": "fixed-target",
        "samples": [
            {"point_id": "P1", "direction_id": "D1", "sample_id": "1", "vision_pose": first_pose.tolist()},
            {"point_id": "P1", "direction_id": "D1", "sample_id": "2", "image_path": image.name},
        ],
    })
    service.save_parameters({"camera_matrix": [[1000, 0, 320], [0, 1000, 240], [0, 0, 1]]})
    monkeypatch.setattr(service_module.cv2, "imdecode", lambda *_: np.zeros((20, 20), dtype=np.uint8))
    used_references = []

    def estimate(*args, reference_rotation=None, **kwargs):
        used_references.append(reference_rotation)
        return {"vision_pose": first_pose.copy(), "reprojection_error_px": 0.1}

    monkeypatch.setattr(service_module, "estimate_board_pose", estimate)
    imported = service.load_observations(source)
    assert len(imported["samples"]) == 2
    assert len(used_references) == 1
    assert used_references[0] == pytest.approx(first_pose[:3, :3])


def test_failed_import_clears_old_batch_before_evaluation(service, tmp_path):
    establish_baseline(service, tmp_path / "initial.json")
    service.load_observations(batch_file(tmp_path / "valid.json", repeated((101, 200, 300))))
    assert service.current_batch is not None
    invalid = write_json(tmp_path / "invalid.json", {"length_unit": "mm"})
    with pytest.raises(ValueError, match="program_id"):
        service.load_observations(invalid)
    assert service.current_batch is None
    with pytest.raises(ValueError, match="本次复测观测"):
        service.evaluate()
    assert service.list_history() == []


def test_debug_batches_remain_explicitly_unverified(service, tmp_path):
    establish_baseline(service, tmp_path / "initial.json", comparison_status="debug_unverified")
    service.load_observations(batch_file(tmp_path / "current.json", repeated()))
    result = service.evaluate()
    assert result["status"] == "调试比较 · 采样对应待确认"
    assert any("不能将比较结果认定为真实精度退化" in warning for warning in result["warnings"])
    assert result["comparison_status"] == "debug_unverified"
    assert all(assess_metric(result, mode)["status"].startswith("调试比较") for mode in METRIC_LABELS)


def test_metric_thresholds_are_independent_and_old_drift_limits_are_not_migrated(service):
    service.save_settings({"thresholds": {"X": 99}})
    assert all(value is None for thresholds in service.settings["metric_thresholds"].values()
               for value in thresholds.values())
    service.save_settings({"metric_thresholds": {"repeatability": {"X": 0.5, "distance": 0.7}}})
    service.save_settings({"metric_thresholds": {"absolute_change": {"X": 0.2}}})
    reopened = PositionMonitoringService(root=service.root)
    assert reopened.settings["metric_thresholds"]["repeatability"]["X"] == 0.5
    assert reopened.settings["metric_thresholds"]["absolute_change"]["X"] == 0.2
    assert reopened.settings["metric_thresholds"]["repeatability_change"]["X"] is None
    before = deepcopy(service.settings)
    with pytest.raises(ValueError, match="非负数"):
        service.save_settings({"metric_thresholds": {"repeatability": {"X": -1}}})
    assert service.settings == before
    state_path = service.storage / "state.json"
    state = read_document(state_path)
    del state["settings"]["metric_thresholds"]
    write_json(state_path, state)
    old_state = PositionMonitoringService(root=service.root)
    assert old_state.settings["thresholds"]["X"] == 99
    assert all(value is None for thresholds in old_state.settings["metric_thresholds"].values()
               for value in thresholds.values())


def test_current_repeatability_without_baseline_saves_real_statistics_and_snapshot(service, tmp_path):
    positions = [("P1", "D1", (99, 200, 300)), ("P1", "D1", (101, 200, 300))]
    imported = service.load_observations(batch_file(tmp_path / "current.json", positions))
    result = service.evaluate_current()
    assert result["baseline_id"] is None
    assert result["baseline_path"] is None
    assert result["baseline_created_at"] is None
    assert result["program_id"] == imported["program_id"]
    assert result["target_id"] == imported["target_id"]
    assert result["groups"][0]["baseline"] is None
    assert metric_values(result, "repeatability") == pytest.approx([3 * np.sqrt(2), 0, 0, 1], abs=1e-9)
    assert metric_values(result, "repeatability_change") == [None] * 4
    assert metric_values(result, "absolute_change") == [None] * 4
    assert result["summary"]["drift_distance"] is None
    assert result["status"] == "未设置阈值"
    assert assess_metric(result, "absolute_change")["status"] == "指标不可计算"
    assert read_document(result["current_batch_path"])["samples"] == imported["samples"]
    assert service.list_history() == [result]


def test_one_arrival_per_direction_does_not_create_repeatability_samples(service, tmp_path):
    positions = [("P1", "D1", (99, 200, 300)), ("P1", "D2", (101, 200, 300))]
    service.load_observations(batch_file(tmp_path / "current.json", positions))
    service.save_settings({"metric_thresholds": {"repeatability": {"distance": 1}}})
    result = service.evaluate_current()
    assert len(result["groups"]) == 2
    assert metric_values(result, "repeatability") == [None] * 4
    assert result["status"] == "指标不可计算"
    assert assess_metric(result, "repeatability")["alarms"] == []


def test_current_repeatability_does_not_borrow_selected_baseline_rotation(service, tmp_path):
    establish_baseline(service, tmp_path / "baseline.json")
    positions = [("P1", "D1", (99, 200, 300)), ("P1", "D1", (101, 200, 300))]
    service.load_observations(batch_file(tmp_path / "current.json", positions, include_q=False))
    result = service.evaluate_current()
    assert result["summary"]["axis_3sigma_base"] is None
    assert result["summary"]["rp_current"] == pytest.approx(1)
    assert result["baseline_id"] is None


def test_negative_degradation_is_improvement_and_threshold_snapshots_stay_independent(service, tmp_path):
    initial = [("P1", "D1", (98, 200, 300)), ("P1", "D1", (102, 200, 300))]
    establish_baseline(service, tmp_path / "initial.json", initial,
                       initial_errors={"P1": {"D1": [2, 0, 0]}})
    current = [("P1", "D1", (98, 200, 300)), ("P1", "D1", (100, 200, 300))]
    service.load_observations(batch_file(tmp_path / "current.json", current))
    service.save_settings({"metric_thresholds": {
        "absolute_change": dict.fromkeys(("X", "Y", "Z", "distance"), 0.5),
        "repeatability_change": dict.fromkeys(("X", "Y", "Z", "distance"), 0.5),
        "repeatability": {"distance": 0.5},
    }})
    result = service.evaluate()
    assert result["summary"]["absolute_ap_change"] == pytest.approx(-1)
    assert result["summary"]["rp_change"] == pytest.approx(-1)
    assert assess_metric(result, "absolute_change")["status"] == "阈值内"
    assert assess_metric(result, "repeatability_change")["status"] == "阈值内"
    assert {alarm["metric"] for alarm in result["alarms"]} == {"repeatability"}
    assert assess_metric(result, "repeatability")["status"] == "超限"
    service.save_settings({"metric_thresholds": {"repeatability": {"distance": 9}}})
    assert assess_metric(result, "repeatability")["thresholds"]["distance"] == 0.5
    assert service.list_history()[0]["metric_thresholds"] == result["metric_thresholds"]


def test_old_history_never_borrows_drift_threshold_or_alarm_for_new_metric(service, tmp_path):
    service.load_observations(batch_file(tmp_path / "current.json", repeated()))
    result = service.evaluate_current()
    del result["metric_thresholds"]
    result.update({"thresholds": dict.fromkeys(("X", "Y", "Z", "distance"), 0.01),
                   "status": "超限", "alarms": [{"axis": "X", "value": 3}]})
    assessment = assess_metric(result, "repeatability")
    assert assessment["status"] == "未设置阈值"
    assert assessment["alarms"] == []
    assert set(assessment["thresholds"].values()) == {None}


@pytest.mark.parametrize("change, message", [
    ({"parameter_version": "old-version"}, "参数版本已变化"),
    ({"comparison_status": "calibration_only"}, "手眼标定数据"),
])
def test_current_repeatability_rejects_changed_parameters_and_calibration_data(service, tmp_path, change, message):
    service.load_observations(batch_file(tmp_path / "current.json", repeated()))
    service.current_batch.update(change)
    with pytest.raises(ValueError, match=message):
        service.evaluate_current()
    assert service.list_history() == []


@pytest.mark.parametrize("incompatibility", ["parameters", "program", "target", "groups"])
def test_current_repeatability_can_bypass_an_incompatible_baseline(service, tmp_path, incompatibility):
    baseline = establish_baseline(service, tmp_path / "initial.json")
    metadata = {}
    if incompatibility == "parameters":
        service.save_parameters({"hand_eye": HAND_EYE.tolist()})
    elif incompatibility == "program":
        metadata["program_id"] = "another-program"
    elif incompatibility == "target":
        metadata["target_id"] = "another-target"
    point = "P2" if incompatibility == "groups" else "P1"
    positions = [(point, "D1", (99, 200, 300)), (point, "D1", (101, 200, 300))]
    service.load_observations(batch_file(tmp_path / "current.json", positions, **metadata))
    with pytest.raises(ValueError):
        service.evaluate()
    result = service.evaluate(allow_current_only=True)
    assert result["baseline_id"] is None
    assert result["baseline_created_at"] is None
    assert result["summary"]["rp_current"] == pytest.approx(1)
    assert result["summary"]["rp_change"] is None
    assert service.baseline["id"] == baseline["id"]
    service.current_batch["parameter_version"] = "expired-current-parameters"
    with pytest.raises(ValueError, match="参数版本已变化"):
        service.evaluate(allow_current_only=True)


def test_current_mode_still_compares_when_the_baseline_is_compatible(service, tmp_path):
    baseline = establish_baseline(service, tmp_path / "initial.json")
    service.load_observations(batch_file(tmp_path / "current.json", repeated()))
    result = service.evaluate(allow_current_only=True)
    assert result["baseline_id"] == baseline["id"]
    assert result["baseline_created_at"] == baseline["created_at"]
    assert result["summary"]["rp_change"] == pytest.approx(0)
