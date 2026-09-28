"""真实文件边界：图片身份、仿真参数适配、观测持久化和测量/真值隔离。"""

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import core.services.position_monitoring_service as service_module
from core.services.position_monitoring_service import PositionMonitoringService, read_document


def pose(position, rotvec=(0.1, -0.2, 0.3)):
    result = np.eye(4)
    result[:3, :3] = cv2.Rodrigues(np.asarray(rotvec, dtype=float))[0]
    result[:3, 3] = position
    return result


HAND_EYE = pose([40, -30, 80])
TARGET = pose([500, 300, 900], [-0.2, 0.15, 0.3])
IDEAL = pose([100, 200, 300], [0.05, -0.1, 0.2])


def write_json(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    dataset = tmp_path / "fixed_dataset"
    native = {
        "schema": "ur10_simulated_camera_parameters_v1", "length_unit": "mm",
        "camera": {"camera_matrix": [[200, 0, 4], [0, 200, 3], [0, 0, 1]],
                   "dist_coeffs": [0] * 5, "image_size_px": [8, 6]},
        "charuco": {"dictionary": "DICT_5X5_1000", "squares_xy": [9, 7],
                    "square_length_mm": 20, "marker_length_mm": 14,
                    "marker_ids": list(range(31)), "legacy_pattern": False},
        "system_transforms": {"E_T_C_hand_eye": HAND_EYE.tolist(),
                              "B_T_M_charuco_top_left": TARGET.tolist()},
        "extrinsics_by_capture": {"B001/P001/D001": {"C_T_M": pose([999, 999, 999]).tolist()}},
    }
    parameter_file = write_json(dataset / "parameters.json", native)
    service = PositionMonitoringService(root=tmp_path / "application")
    service.load_parameters(parameter_file)
    measurements, calls = {}, []

    def estimate(image, camera, distortion, charuco, square_size):
        token = int(image[0, 0])
        calls.append(token)
        assert image.shape == (6, 8)
        assert charuco["dictionary"] == "DICT_5X5_1000"
        assert square_size == 20
        return {"vision_pose": measurements[token].copy(), "reprojection_error_px": 0.05,
                "reprojection_mean_px": 0.04, "corner_count": 8,
                "corner_order": "charuco_ids", "charuco_corner_ids": list(range(8))}

    monkeypatch.setattr(service_module, "estimate_charuco_pose", estimate)

    def batch(batch_id="B001", offsets=(-0.2, 0.2), *, with_record=True):
        folder = dataset / batch_id
        frames, images = [], []
        for index, offset in enumerate(offsets, 1):
            token = len(measurements) + 1
            endpoint = IDEAL.copy()
            endpoint[0, 3] += offset
            measurements[token] = np.linalg.inv(HAND_EYE) @ np.linalg.inv(endpoint) @ TARGET
            image = folder / "calibration_images" / f"{batch_id}_P001_D{index:03d}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            cv2.imencode(".png", np.full((6, 8), token, dtype=np.uint8))[1].tofile(image)
            images.append(image)
            frames.append({"image": image.relative_to(folder).as_posix(),
                           "ideal": IDEAL.tolist(), "actual": pose([999, 888, 777]).tolist()})
        record = folder / "record.json"
        if with_record:
            write_json(record, {"batch": batch_id, "inertia": 0.1, "frames": frames})
        return SimpleNamespace(folder=folder, images=images, record=record, frames=frames)

    return SimpleNamespace(service=service, native=native, parameters=parameter_file,
                           batch=batch, measurements=measurements, calls=calls)


def test_native_parameters_are_adapted_without_modifying_source_or_importing_truth(inputs):
    original = inputs.parameters.read_bytes()
    service = inputs.service
    version = service.parameters["version"]
    service.load_parameters(inputs.parameters)
    assert service.parameters["version"] == version
    assert inputs.parameters.read_bytes() == original
    assert service.parameters["camera_matrix"] == inputs.native["camera"]["camera_matrix"]
    assert service.parameters["hand_eye"] == HAND_EYE.tolist()
    assert service.parameters["target_pose_base"] == TARGET.tolist()
    assert service.parameters["charuco"]["squares_xy"] == [9, 7]
    assert service.parameters["board_type"] == "charuco"
    assert service.parameters["end_frame"] == "tool0"
    assert "extrinsics_by_capture" not in service.parameters


@pytest.mark.parametrize("source_kind", ["record", "folder", "images", "single"])
def test_input_forms_decode_images_and_save_complete_observations(inputs, source_kind):
    batch = inputs.batch()
    original = {path: path.read_bytes() for path in [batch.record, *batch.images]}
    sources = {"record": batch.record, "folder": batch.folder,
               "images": batch.images, "single": batch.images[0]}
    imported = inputs.service.load_observations(sources[source_kind])
    expected_count = 1 if source_kind == "single" else 2
    assert len(inputs.calls) == expected_count
    assert len(imported["samples"]) == expected_count
    assert imported["sampling_protocol"] == "multidirectional"
    assert imported["batch_id"] == "B001"
    assert imported["program_id"] == "fixed_dataset"
    sample = imported["samples"][0]
    assert sample["point_id"] == "P001"
    assert sample["direction_id"] == "D001"
    assert sample["sample_id"] == "B001_P001_D001"
    assert len(sample["vision_xyz_mm"]) == len(sample["vision_rpy_deg"]) == 3
    np.testing.assert_allclose(sample["end_pose"], pose([99.8, 200, 300], [0.05, -0.1, 0.2]), atol=1e-9)
    np.testing.assert_allclose(sample["error_base_mm"], [-0.2, 0, 0], atol=1e-9)
    assert sample["simulation_truth"]["end_pose"] == batch.frames[0]["actual"]
    assert Path(imported["saved_path"]).is_file()
    assert read_document(imported["saved_path"])["samples"] == imported["samples"]
    assert imported["parameters"]["version"] == inputs.service.parameters["version"]
    assert {path: path.read_bytes() for path in original} == original


@pytest.mark.parametrize("problem, message", [("duplicate", "只能有一张"), ("mixed", "一个批次"),
                                              ("bad_name", "图片名应为")])
def test_invalid_image_identity_is_rejected_before_pnp(inputs, problem, message):
    first = inputs.batch("B001")
    if problem == "duplicate":
        selected = [first.images[0], first.images[0]]
    elif problem == "mixed":
        second = inputs.batch("B002")
        selected = [first.images[0], second.images[0]]
    else:
        bad = first.images[0].with_name("unnamed.png")
        bad.write_bytes(first.images[0].read_bytes())
        selected = [bad]
    with pytest.raises(ValueError, match=message):
        inputs.service.load_observations(selected)
    assert not inputs.calls
    assert inputs.service.current_batch is None


@pytest.mark.parametrize("problem, message", [("duplicate", "图片重复"), ("batch", "批次.*不一致")])
def test_record_identity_must_match_its_images(inputs, problem, message):
    batch = inputs.batch()
    document = read_document(batch.record)
    if problem == "duplicate":
        document["frames"].append(document["frames"][0])
    else:
        document["batch"] = "B009"
    write_json(batch.record, document)
    with pytest.raises(ValueError, match=message):
        inputs.service.load_observations(batch.record)
    assert not inputs.calls


def test_saved_observations_reload_without_pnp_and_reject_different_parameter_version(inputs, monkeypatch):
    imported = inputs.service.load_observations(inputs.batch().record)
    original = Path(imported["saved_path"]).read_bytes()

    def unexpected_pnp(*args, **kwargs):
        raise AssertionError("已有观测结果不应再次执行 PnP")

    monkeypatch.setattr(service_module, "estimate_charuco_pose", unexpected_pnp)
    restored = inputs.service.load_observations(imported["saved_path"])
    assert restored["samples"] == imported["samples"]
    assert Path(imported["saved_path"]).read_bytes() == original
    inputs.service.save_parameters({"square_size_mm": 21})
    with pytest.raises(ValueError, match="参数版本.*不一致"):
        inputs.service.load_observations(imported["saved_path"])


def test_truth_changes_do_not_change_image_measurements_or_evaluation(inputs):
    batch = inputs.batch()
    first = inputs.service.load_observations(batch.record)
    first_result = inputs.service.evaluate_current()
    record = read_document(batch.record)
    for frame in record["frames"]:
        frame["actual"] = pose([-9999, 7777, 2222], [1, 2, 1]).tolist()
    write_json(batch.record, record)
    second = inputs.service.load_observations(batch.record)
    second_result = inputs.service.evaluate_current()
    assert len(inputs.calls) == 4
    for before, after in zip(first["samples"], second["samples"]):
        assert before["simulation_truth"] != after["simulation_truth"]
        for field in ("vision_pose", "vision_xyz_mm", "vision_rpy_deg", "end_pose", "error_base_mm"):
            assert before[field] == after[field]
    assert first_result["summary"] == second_result["summary"]
    assert first_result["groups"] == second_result["groups"]


def test_single_and_two_period_evaluation_use_multidirectional_scatter(inputs):
    service = inputs.service
    initial = inputs.batch("B001", (-0.2, 0.2))
    service.load_observations(initial.folder)
    single = service.evaluate_current()
    assert single["sampling_protocol"] == "multidirectional"
    assert single["is_standard_repeatability"] is False
    assert single["summary"]["rp_current"] == pytest.approx(0.2, abs=1e-9)
    assert single["summary"]["absolute_ap"] == pytest.approx(0.2, abs=1e-9)
    assert single["summary"]["rp_change"] is None
    baseline = service.create_baseline()
    current = inputs.batch("B002", (-0.4, 0.4))
    service.load_observations(current.images)
    compared = service.evaluate()
    assert compared["baseline_id"] == baseline["id"]
    assert compared["sampling_protocol"] == "multidirectional"
    assert compared["summary"]["rp_change"] == pytest.approx(0.2, abs=1e-9)
    assert compared["summary"]["absolute_ap_change"] == pytest.approx(0.2, abs=1e-9)
    assert len(compared["groups"][0]["directions"]) == 2


def test_single_direction_reference_is_not_duplicated_to_match_twenty_directions(inputs):
    service = inputs.service
    service.load_observations(inputs.batch("B000", (0,)).record)
    service.create_baseline()
    service.load_observations(inputs.batch("B001", (-0.2, 0.2)).record)
    with pytest.raises(ValueError, match="接近方向不匹配"):
        service.evaluate()
    single = service.evaluate(allow_current_only=True)
    assert single["baseline_id"] is None
    assert single["summary"]["rp_change"] is None


def test_png_list_without_record_still_measures_but_does_not_invent_ideal_pose(inputs):
    batch = inputs.batch(with_record=False)
    imported = inputs.service.load_observations(batch.images)
    assert imported["comparison_status"] == "observed"
    assert all("ideal_pose" not in sample and "simulation_truth" not in sample
               for sample in imported["samples"])
    result = inputs.service.evaluate_current()
    assert result["summary"]["rp_current"] == pytest.approx(0.2, abs=1e-9)
    assert result["summary"]["absolute_ap"] is None


def test_image_subfolder_keeps_the_same_batch_record_and_program(inputs):
    batch = inputs.batch()
    from_record = inputs.service.load_observations(batch.record)
    from_images = inputs.service.load_observations(batch.images[0].parent)
    assert from_images["program_id"] == from_record["program_id"]
    assert from_images["source_path"] == from_record["source_path"]
    assert from_images["samples"] == from_record["samples"]
