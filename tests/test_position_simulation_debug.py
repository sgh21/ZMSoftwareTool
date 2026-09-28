"""隔离仿真核验经过图片/PnP，仿真真值只用于测量完成后的对照。"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

import core.services.position_monitoring_service as service_module
from core.services.position_monitoring_service import PositionMonitoringService, read_document
from core.services.position_simulation_debug import run_simulation_check


def transform(position=(0, 0, 0), angle=0):
    cosine, sine = np.cos(angle), np.sin(angle)
    matrix = np.eye(4)
    matrix[:3, :3] = [[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]]
    matrix[:3, 3] = position
    return matrix


HAND_EYE = transform((30, -40, 50), 0.1)
TARGET = transform((10, 20, 1000), -0.2)
IDEAL = transform((0, 0, 300))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def snapshot(folder):
    return {str(path.relative_to(folder)): path.read_bytes() for path in folder.rglob("*") if path.is_file()}


@pytest.fixture
def simulation(tmp_path, monkeypatch):
    dataset = tmp_path / "source_data"
    write_json(dataset / "parameters.json", {
        "schema": "ur10_simulated_camera_parameters_v1",
        "camera": {
            "camera_matrix": [[100, 0, 10], [0, 100, 8], [0, 0, 1]],
            "dist_coeffs": [0, 0, 0, 0, 0], "image_size_px": [20, 16],
        },
        "charuco": {
            "dictionary": "DICT_5X5_1000", "squares_xy": [9, 7],
            "square_length_mm": 20, "marker_length_mm": 14,
        },
        "system_transforms": {
            "E_T_C_hand_eye": HAND_EYE.tolist(), "B_T_M_charuco_top_left": TARGET.tolist(),
        },
    })
    measured_poses = {}
    for batch_index, (batch_id, radius) in enumerate((("B001", 0.2), ("B002", 0.4))):
        frames = []
        for direction, sign in enumerate((-1, 1), 1):
            pixel_value = 10 * (2 * batch_index + direction)
            name = f"{batch_id}_P001_D{direction:03d}.png"
            path = dataset / batch_id / "calibration_images" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            image = np.full((16, 20), pixel_value, dtype=np.uint8)
            success, encoded = cv2.imencode(".png", image)
            assert success
            path.write_bytes(encoded.tobytes())
            frames.append({
                "image": f"calibration_images/{name}", "ideal": IDEAL.tolist(),
                "actual": transform((sign * radius, 0, 300)).tolist(),
                "vision_pose": "This record field must not bypass image measurement",
            })
            measured_poses[pixel_value] = transform((1.25 * sign * radius, 0, 300), 0.02)
        write_json(dataset / batch_id / "record.json", {"batch": batch_id, "frames": frames})
    calls = []

    def estimate_image(image, camera_matrix, distortion, board, square_size):
        pixel_value = int(image[0, 0])
        calls.append(pixel_value)
        assert image.shape == (16, 20)
        assert board["dictionary"] == "DICT_5X5_1000"
        assert square_size == 20
        measured_pose = measured_poses[pixel_value]
        return {
            "vision_pose": np.linalg.inv(HAND_EYE) @ np.linalg.inv(measured_pose) @ TARGET,
            "reprojection_error_px": 0.05,
        }

    monkeypatch.setattr(service_module, "estimate_charuco_pose", estimate_image)
    return dataset, calls


def test_image_measurement_and_truth_report_are_isolated_and_source_files_unchanged(simulation, tmp_path):
    dataset, calls = simulation
    production = PositionMonitoringService(root=tmp_path / "production")
    production_before = snapshot(production.root)
    source_before = snapshot(dataset)
    progress = []
    output = tmp_path / "verification"
    delivered = run_simulation_check(dataset, output, progress=lambda value, text: progress.append((value, text)))
    report = delivered["report"]
    assert calls == [10, 20, 30, 40]
    assert snapshot(dataset) == source_before
    assert snapshot(production.root) == production_before
    assert production.baseline is None
    assert production.current_batch is None
    assert report["measured_result"]["summary"]["rp_current"] == pytest.approx(0.5)
    assert report["truth_result"]["summary"]["rp_current"] == pytest.approx(0.4)
    assert report["measured_result"]["summary"]["absolute_ap_change"] == pytest.approx(0.25)
    assert report["truth_result"]["summary"]["absolute_ap_change"] == pytest.approx(0.2)
    comparison = {row["metric"]: row for row in report["metric_comparison"]}
    assert comparison["absolute_change"]["difference"] == pytest.approx([0.05, 0, 0, 0.05], abs=1e-10)
    assert comparison["repeatability"]["difference"][3] == pytest.approx(0.1)
    assert report["point_metric_comparison"][0]["point_id"] == "P001"
    summary = report["measurement_summary"]
    assert summary["sample_count"] == 4
    assert summary["mean_abs_xyz_mm"] == pytest.approx([0.075, 0, 0], abs=1e-10)
    assert summary["mean_signed_xyz_mm"] == pytest.approx([0, 0, 0], abs=1e-10)
    assert summary["mean_distance_mm"] == pytest.approx(0.075)
    assert summary["p95_distance_mm"] == pytest.approx(0.1)
    assert summary["max_distance_mm"] == pytest.approx(0.1)
    assert summary["mean_rotation_deg"] == pytest.approx(np.degrees(0.02))
    assert summary["p95_rotation_deg"] == pytest.approx(np.degrees(0.02))
    assert summary["max_rotation_deg"] == pytest.approx(np.degrees(0.02))
    first = report["batches"][0]["samples"][0]
    assert first["measured_position_xyz_mm"] == pytest.approx([-0.25, 0, 300], abs=1e-10)
    assert first["truth_position_xyz_mm"] == pytest.approx([-0.2, 0, 300], abs=1e-10)
    assert first["position_error_xyz_mm"] == pytest.approx([-0.05, 0, 0], abs=1e-10)
    assert first["rotation_error_deg"] == pytest.approx(np.degrees(0.02))
    assert read_document(delivered["report_path"]) == report
    assert Path(delivered["report_path"]) == output / "verification.json"
    assert Path(report["measured_result"]["baseline_path"]).is_relative_to(output)
    assert Path(report["measured_result"]["current_batch_path"]).is_relative_to(output)
    assert [value for value, _ in progress] == sorted(value for value, _ in progress)
    assert progress[0][0] == 0 and progress[-1][0] == 100
    assert all(value <= 45 for value, text in progress if text.startswith("B001"))
    assert all(45 <= value <= 90 for value, text in progress if text.startswith("B002"))


def test_changing_simulation_truth_changes_only_the_check_not_image_measurement(simulation, tmp_path):
    dataset, calls = simulation
    first = run_simulation_check(dataset, tmp_path / "first")["report"]
    for batch_id in ("B001", "B002"):
        path = dataset / batch_id / "record.json"
        document = read_document(path)
        for frame in document["frames"]:
            frame["actual"][0][3] *= 3
        write_json(path, document)
    second = run_simulation_check(dataset, tmp_path / "second")["report"]
    assert calls == [10, 20, 30, 40] * 2
    assert second["measured_result"]["summary"] == first["measured_result"]["summary"]
    assert second["truth_result"]["summary"]["rp_current"] == pytest.approx(1.2)
    assert second["measurement_summary"]["mean_distance_mm"] > first["measurement_summary"]["mean_distance_mm"]


def test_failed_image_measurement_never_falls_back_to_valid_actual_poses(simulation, tmp_path, monkeypatch):
    dataset, _ = simulation

    def reject_image(*args, **kwargs):
        raise ValueError("图像无法检测标定板")

    monkeypatch.setattr(service_module, "estimate_charuco_pose", reject_image)
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="图像无法检测标定板"):
        run_simulation_check(dataset, output)
    assert not (output / "verification.json").exists()


@pytest.mark.parametrize("folder", [".", "checks"])
def test_output_inside_source_is_rejected_without_touching_images(simulation, folder):
    dataset, calls = simulation
    before = snapshot(dataset)
    with pytest.raises(ValueError, match="不能位于原始仿真数据目录内"):
        run_simulation_check(dataset, dataset / folder)
    assert calls == []
    assert snapshot(dataset) == before


def test_missing_batch_is_explicit_and_does_not_search_incomplete_work(simulation, tmp_path):
    dataset, calls = simulation
    missing_id = "B003"
    write_json(dataset / "_work" / missing_id / "record.json", {"batch": missing_id, "frames": []})
    output = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="缺少复测批次 B003"):
        run_simulation_check(dataset, output, current_id=missing_id)
    assert calls == []
    assert not output.exists()


def test_missing_actual_reports_missing_truth_after_images_are_measured(simulation, tmp_path):
    dataset, calls = simulation
    path = dataset / "B001" / "record.json"
    document = read_document(path)
    document["frames"][0].pop("actual")
    write_json(path, document)
    output = tmp_path / "missing_truth"
    with pytest.raises(ValueError, match="缺少仿真末端真值 actual"):
        run_simulation_check(dataset, output)
    assert calls == [10, 20, 30, 40]
    assert not (output / "verification.json").exists()
