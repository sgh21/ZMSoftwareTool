"""当前参数、上一版备份及旧状态迁移；所有写入隔离在 tmp_path。"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import core.services.position_monitoring_service as service_module
from core.services.position_monitoring_service import (
    PositionMonitoringService,
    read_document,
    write_document,
)


def template():
    return read_document(Path(__file__).resolve().parents[1] / "config" / "robot_position.json")


def observation_file(root):
    path = root / "observations.json"
    write_document(path, {
        "length_unit": "mm", "program_id": "program", "target_id": "target",
        "samples": [{
            "point_id": "P001", "direction_id": "D001", "sample_id": "1",
            "vision_pose": np.eye(4).tolist(),
        }],
    })
    return path


def test_first_start_creates_current_and_later_starts_ignore_template_changes(tmp_path):
    configured = tmp_path / "config" / "robot_position.json"
    initial = template()
    initial["square_size_mm"] = 2.5
    write_document(configured, initial)
    service = PositionMonitoringService(root=tmp_path)
    assert service.parameters == initial
    assert read_document(service.parameter_path) == initial
    assert not service.previous_parameter_path.exists()
    assert set(read_document(service.storage / "state.json")) == {"settings", "baseline_path"}
    current_bytes = service.parameter_path.read_bytes()
    current_mtime = service.parameter_path.stat().st_mtime_ns
    write_document(configured, {**initial, "square_size_mm": 9})
    reopened = PositionMonitoringService(root=tmp_path)
    assert reopened.parameters == initial
    assert reopened.parameter_path.read_bytes() == current_bytes
    assert reopened.parameter_path.stat().st_mtime_ns == current_mtime


def test_actual_changes_keep_only_current_and_one_rolling_backup(tmp_path):
    service = PositionMonitoringService(root=tmp_path)
    original = deepcopy(service.parameters)
    first = service.save_parameters({"hand_eye": np.eye(4).tolist()})
    assert first["version"] != original["version"]
    assert read_document(service.previous_parameter_path) == original
    assert read_document(service.parameter_path) == first
    second = service.save_parameters({"square_size_mm": first["square_size_mm"] + 1})
    assert second["version"] != first["version"]
    assert read_document(service.previous_parameter_path) == first
    assert read_document(service.parameter_path) == second
    assert {path.name for path in service.parameter_path.parent.glob("*.json")} == {
        "current.json", "previous.json",
    }
    assert PositionMonitoringService(root=tmp_path).parameters == second
    assert "parameters" not in read_document(service.storage / "state.json")


def test_same_effective_parameters_do_not_write_or_clear_observation(tmp_path, monkeypatch):
    service = PositionMonitoringService(root=tmp_path)
    original = service.save_parameters({"hand_eye": np.eye(4).tolist()})
    observations = observation_file(tmp_path)
    service.load_observations(observations)
    baseline = service.create_baseline("initial")
    service.load_observations(observations)
    batch = service.current_batch
    external = tmp_path / "another-source.json"
    write_document(external, {
        **original, "version": "external-version", "source_path": "different-source",
        "source_note": "copied file", "updated_at": "later", "notes": "same calibration",
    })
    source_bytes = external.read_bytes()

    def unexpected_write(*args, **kwargs):
        raise AssertionError("相同有效参数不应写文件或生成版本")

    monkeypatch.setattr(service_module, "write_document", unexpected_write)
    monkeypatch.setattr(service_module, "_stamp", unexpected_write)
    assert service.load_parameters(external) == original
    assert service.save_parameters(deepcopy(original)) == original
    assert service.current_batch is batch
    assert service.baseline["id"] == baseline["id"]
    assert service.current_batch["parameter_version"] == baseline["parameters"]["version"]
    assert external.read_bytes() == source_bytes


def test_legacy_state_migrates_once_without_deleting_old_versions(tmp_path):
    storage = tmp_path / "storage" / "position_monitoring"
    legacy = {**template(), "version": "legacy-version", "hand_eye": np.eye(4).tolist()}
    old_version = storage / "parameters" / "legacy-version.json"
    write_document(old_version, legacy)
    old_bytes = old_version.read_bytes()
    state_path = storage / "state.json"
    write_document(state_path, {
        "parameters": legacy, "settings": {"processing_points": [{"id": "work-point"}]},
        "baseline_path": None,
    })
    service = PositionMonitoringService(root=tmp_path)
    assert service.parameters == legacy
    assert read_document(service.parameter_path) == legacy
    assert not service.previous_parameter_path.exists()
    assert old_version.read_bytes() == old_bytes
    migrated_state = read_document(state_path)
    assert "parameters" not in migrated_state
    assert service.settings["processing_points"] == [{"id": "work-point"}]
    write_document(state_path, {**migrated_state, "parameters": {**legacy, "version": "obsolete"}})
    reopened = PositionMonitoringService(root=tmp_path)
    assert reopened.parameters == legacy
    assert old_version.read_bytes() == old_bytes
    assert "parameters" not in read_document(state_path)


def test_select_baseline_restores_exact_version_and_rolls_previous_parameters(tmp_path):
    service = PositionMonitoringService(root=tmp_path)
    first = service.save_parameters({"hand_eye": np.eye(4).tolist()})
    service.load_observations(observation_file(tmp_path))
    baseline = service.create_baseline("initial")
    baseline_bytes = Path(baseline["path"]).read_bytes()
    second = service.save_parameters({"square_size_mm": first["square_size_mm"] + 1})
    service.select_baseline(baseline["path"])
    assert service.parameters == first
    assert read_document(service.parameter_path) == first
    assert read_document(service.previous_parameter_path) == second
    assert Path(baseline["path"]).read_bytes() == baseline_bytes
    reopened = PositionMonitoringService(root=tmp_path)
    assert reopened.parameters == first
    assert reopened.baseline["parameters"]["version"] == first["version"]
    backup_bytes = reopened.previous_parameter_path.read_bytes()
    reopened.select_baseline(baseline["path"])
    assert reopened.previous_parameter_path.read_bytes() == backup_bytes


def test_invalid_parameter_change_preserves_current_backup_and_batch(tmp_path):
    service = PositionMonitoringService(root=tmp_path)
    parameters = service.save_parameters({"hand_eye": np.eye(4).tolist()})
    service.load_observations(observation_file(tmp_path))
    batch = service.current_batch
    files = (service.parameter_path, service.previous_parameter_path, service.storage / "state.json")
    before = {path: path.read_bytes() for path in files}
    with pytest.raises(ValueError, match="棋盘格长"):
        service.save_parameters({"square_size_mm": -1})
    assert service.parameters == parameters
    assert service.current_batch is batch
    assert {path: path.read_bytes() for path in files} == before


def test_external_metre_parameters_are_read_only_and_repeated_import_keeps_version(tmp_path):
    service = PositionMonitoringService(root=tmp_path)
    pose = np.eye(4)
    pose[:3, 3] = [0.04, -0.03, 0.08]
    source = tmp_path / "hand-eye-metres.json"
    write_document(source, {"hand_eye": pose.tolist(), "length_unit": "m"})
    before = source.read_bytes()
    first = service.load_parameters(source)
    assert np.asarray(first["hand_eye"])[:3, 3] == pytest.approx([40, -30, 80])
    current_bytes = service.parameter_path.read_bytes()
    previous_bytes = service.previous_parameter_path.read_bytes()
    assert service.load_parameters(source)["version"] == first["version"]
    assert service.parameter_path.read_bytes() == current_bytes
    assert service.previous_parameter_path.read_bytes() == previous_bytes
    assert source.read_bytes() == before


def simulation_parameter_file(path):
    write_document(path, {
        "schema": "ur10_simulated_camera_parameters_v1",
        "camera": {"camera_matrix": [[5000, 0, 1536], [0, 5000, 1024], [0, 0, 1]],
                   "dist_coeffs": [0] * 5, "image_size_px": [3072, 2048]},
        "charuco": {"dictionary": "DICT_5X5_1000", "squares_xy": [9, 7],
                    "square_length_mm": 20, "marker_length_mm": 14},
        "system_transforms": {"E_T_C_hand_eye": np.eye(4).tolist(),
                              "B_T_M_charuco_top_left": np.eye(4).tolist()},
    })
    return path


def test_full_checkerboard_file_clears_previous_simulation_measurement_conditions(tmp_path):
    service = PositionMonitoringService(root=tmp_path)
    service.load_parameters(simulation_parameter_file(tmp_path / "simulation.json"))
    previous = deepcopy(service.parameters)
    assert previous["board_type"] == "charuco"
    assert previous["target_pose_base"] is not None
    source = tmp_path / "legacy-complete.json"
    complete = {**template(), "hand_eye": np.eye(4).tolist(),
                "camera_matrix": [[1200, 0, 640], [0, 1200, 480], [0, 0, 1]]}
    write_document(source, complete)
    source_bytes = source.read_bytes()
    parameters = service.load_parameters(source)
    assert parameters["board_type"] == "checkerboard"
    assert parameters["charuco"] == {}
    assert parameters["target_pose_base"] is None
    assert parameters["image_size_px"] is None
    assert parameters["end_frame"] is None
    assert parameters["target_id"] == "configured_target"
    assert parameters["source_note"] == ""
    assert parameters["board_grid"] == complete["board_grid"]
    assert parameters["camera_matrix"] == complete["camera_matrix"]
    assert read_document(service.previous_parameter_path) == previous
    assert source.read_bytes() == source_bytes
    assert service.load_parameters(source)["version"] == parameters["version"]


def test_complete_configuration_keeps_explicit_new_fields(tmp_path):
    service = PositionMonitoringService(root=tmp_path)
    service.load_parameters(simulation_parameter_file(tmp_path / "simulation.json"))
    source = tmp_path / "complete.json"
    target = np.eye(4)
    target[:3, 3] = [10, 20, 300]
    complete = {
        **template(), "camera_matrix": [[1000, 0, 320], [0, 1000, 240], [0, 0, 1]],
        "hand_eye": np.eye(4).tolist(), "board_type": "charuco", "square_size_mm": 20,
        "charuco": {"dictionary": "DICT_6X6_250", "squares_xy": [9, 7],
                    "marker_length_mm": 12, "legacy_pattern": True},
        "image_size_px": [640, 480], "target_pose_base": target.tolist(),
        "end_frame": "TCP", "target_id": "actual-board", "source_note": "实际标定参数",
    }
    write_document(source, complete)
    parameters = service.load_parameters(source)
    for key in ("board_type", "charuco", "image_size_px", "target_pose_base", "end_frame",
                "target_id", "source_note"):
        assert parameters[key] == complete[key]


def test_legacy_hand_eye_import_only_changes_hand_eye(tmp_path):
    service = PositionMonitoringService(root=tmp_path)
    service.load_parameters(simulation_parameter_file(tmp_path / "simulation.json"))
    before = deepcopy(service.parameters)
    source = tmp_path / "legacy-hand-eye.json"
    transform = np.eye(4)
    transform[:3, 3] = [0.04, 0.05, 0.06]
    write_document(source, {"T_tool_cam": transform.tolist()})
    source_bytes = source.read_bytes()
    parameters = service.load_parameters(source)
    for key in ("board_type", "charuco", "target_pose_base", "image_size_px", "end_frame",
                "target_id", "camera_matrix", "dist_coeffs"):
        assert parameters[key] == before[key]
    np.testing.assert_array_equal(np.asarray(parameters["hand_eye"])[:3, 3], [40, 50, 60])
    assert source.read_bytes() == source_bytes
