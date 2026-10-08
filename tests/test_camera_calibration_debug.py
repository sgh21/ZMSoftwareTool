"""独立标定的合成几何、参数来源与原始数据隔离回归。"""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.algorithms.board_pose import make_charuco_board
from core.services.position_monitoring_service import PositionMonitoringService
from debug.diagnostics import camera_calibration as calibration


def _transform(rotvec, translation):
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(np.asarray(rotvec, dtype=float))[0]
    pose[:3, 3] = translation
    return pose


@pytest.fixture
def synthetic_dataset(tmp_path, monkeypatch):
    dataset = tmp_path / "input"
    batch = dataset / "B000"
    batch.mkdir(parents=True)
    matrix = np.array([[1200, 0, 640], [0, 1180, 480], [0, 0, 1]], dtype=float)
    hand_eye = _transform([0.2, -0.1, 0.3], [25, -60, 80])
    target = _transform([0.1, 0.2, 0.3], [600, 200, 100])
    specification = {"dictionary": "DICT_5X5_1000", "squares_xy": [9, 7],
                     "square_length_mm": 20.0, "marker_length_mm": 14.0}
    board = make_charuco_board(specification, 20)
    rng = np.random.default_rng(91)
    frames, detections = [], {}
    for index in range(14):
        vision = _transform(rng.uniform(-0.45, 0.45, 3),
                            [rng.uniform(-120, -30), rng.uniform(-100, -20), rng.uniform(450, 700)])
        robot = target @ np.linalg.inv(vision) @ np.linalg.inv(hand_eye)
        # 部分角点也必须按 ID 正确配对，不能假设每张都检测完整棋盘。
        ids = np.arange(index % 5, len(board.getChessboardCorners()), dtype=np.int32)
        points = cv2.projectPoints(
            board.getChessboardCorners()[ids], cv2.Rodrigues(vision[:3, :3])[0],
            vision[:3, 3], matrix, np.zeros(5),
        )[0].reshape(-1, 2)
        name = f"B000_P{index + 1:03d}_D001.png"
        frames.append({"image": name, "actual": robot.tolist(), "ideal": np.eye(4).tolist()})
        detections[name] = (points, ids, (1280, 960))
    document = {
        "schema": "ur10_simulated_camera_parameters_v1", "length_unit": "mm",
        "camera": {"camera_matrix": matrix.tolist(), "dist_coeffs": [0] * 5,
                   "image_size_px": [1280, 960]},
        "charuco": specification,
        "system_transforms": {"E_T_C_hand_eye": hand_eye.tolist(),
                              "B_T_M_charuco_top_left": target.tolist()},
        "extrinsics_by_capture": {
            str(index): {"image_filename": f"B000/{frame['image']}", "C_T_M": np.eye(4).tolist()}
            for index, frame in enumerate(frames)
        },
    }
    (dataset / "parameters.json").write_text(json.dumps(document), encoding="utf-8")
    (batch / "record.json").write_text(json.dumps({"batch": "B000", "frames": frames}), encoding="utf-8")
    monkeypatch.setattr(calibration, "_detect_image", lambda path, board: detections[path.name])
    return dataset, matrix, hand_eye, detections


@pytest.mark.parametrize("distortion_mode", ["estimate", "zero"])
def test_image_calibration_recovers_parameters_without_reference_initialization(synthetic_dataset, tmp_path, distortion_mode):
    dataset, matrix, hand_eye, _ = synthetic_dataset
    parameter_path = dataset / "parameters.json"
    document = json.loads(parameter_path.read_text())
    document["camera"]["camera_matrix"] = [[3500, 0, 10], [0, 2500, 20], [0, 0, 1]]
    document["system_transforms"]["E_T_C_hand_eye"] = np.eye(4).tolist()
    parameter_path.write_text(json.dumps(document), encoding="utf-8")
    before = parameter_path.read_bytes()
    events = []
    result = calibration.calibrate_dataset(
        dataset, tmp_path / "result", distortion_mode=distortion_mode,
        progress=lambda value, text: events.append(value),
    )
    report = result["report"]
    np.testing.assert_allclose(report["camera_calibration"]["camera_matrix"], matrix, atol=0.003)
    np.testing.assert_allclose(report["hand_eye_calibration"]["hand_eye"], hand_eye, atol=0.004)
    assert report["camera_calibration"]["rms_px"] < 1e-4
    assert report["hand_eye_calibration"]["translation_rms_mm"] < 0.003
    assert report["sample_count"] == 14
    assert all(pose["robot_pose_source"] == "actual" for pose in report["poses"])
    assert parameter_path.read_bytes() == before
    assert events[0] == 0 and events[-1] == 100 and events == sorted(events)
    # 导出完整参数经正式服务导入可用；不携带逐图真值作为主流程输入。
    service = PositionMonitoringService(root=tmp_path / "isolated_app")
    parameters = service.load_parameters(result["parameters_path"])
    assert parameters["board_type"] == "charuco"
    assert parameters["end_frame"] == "tool0"
    assert "extrinsics_by_capture" not in parameters
    assert Path(result["hand_eye_path"]).is_file()
    assert json.loads(Path(result["report_path"]).read_text(encoding="utf-8"))["batch_id"] == "B000"


@pytest.mark.parametrize("intrinsic_mode,hand_eye_mode", [
    ("reference", "calibrate"), ("calibrate", "reference"), ("reference", "reference"),
])
def test_reference_modes_are_independent_and_declared(synthetic_dataset, tmp_path, intrinsic_mode, hand_eye_mode):
    dataset, matrix, hand_eye, _ = synthetic_dataset
    result = calibration.calibrate_dataset(
        dataset, tmp_path / "result", intrinsic_mode=intrinsic_mode, hand_eye_mode=hand_eye_mode,
    )
    report = result["report"]
    assert report["intrinsic_mode"] == intrinsic_mode
    assert report["hand_eye_mode"] == hand_eye_mode
    np.testing.assert_allclose(report["camera_calibration"]["camera_matrix"], matrix, atol=0.003)
    np.testing.assert_allclose(report["hand_eye_calibration"]["hand_eye"], hand_eye, atol=0.004)
    if hand_eye_mode == "reference":
        assert report["hand_eye_calibration"]["hand_eye"] == hand_eye.tolist()
        assert report["hand_eye_calibration"]["method"] == "reference"
    if intrinsic_mode == "reference":
        assert report["camera_calibration"]["camera_matrix"] == matrix.tolist()
        assert report["camera_calibration"]["distortion_mode"] == "reference"


def test_explicit_robot_reading_takes_priority_and_ideal_is_never_used(synthetic_dataset, tmp_path):
    dataset, _, hand_eye, _ = synthetic_dataset
    record_path = dataset / "B000" / "record.json"
    record = json.loads(record_path.read_text())
    for frame in record["frames"]:
        frame["robot_pose"] = frame["actual"]
        frame["actual"] = np.eye(4).tolist()
    record_path.write_text(json.dumps(record), encoding="utf-8")
    report = calibration.calibrate_dataset(dataset, tmp_path / "result", intrinsic_mode="reference")["report"]
    np.testing.assert_allclose(report["hand_eye_calibration"]["hand_eye"], hand_eye, atol=0.004)
    assert all(pose["robot_pose_source"] == "robot_pose" for pose in report["poses"])
    for frame in record["frames"]:
        del frame["robot_pose"]
        del frame["actual"]
    record_path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="缺少 robot_pose 或 actual"):
        calibration.calibrate_dataset(dataset, tmp_path / "missing_robot")


def test_input_and_existing_outputs_cannot_be_overwritten(synthetic_dataset, tmp_path):
    dataset, _, _, _ = synthetic_dataset
    for output in (dataset, dataset / "derived"):
        with pytest.raises(ValueError, match="原始数据"):
            calibration.calibrate_dataset(dataset, output)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "hand_eye.json").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="保留旧参数"):
        calibration.calibrate_dataset(dataset, output)
    assert (output / "hand_eye.json").read_text() == "keep"


def test_real_charuco_detection_uses_image_pixels_and_accepts_unicode_path(tmp_path):
    specification = {"dictionary": "DICT_5X5_1000", "squares_xy": [9, 7], "marker_length_mm": 14}
    board = make_charuco_board(specification, 20)
    image = cv2.copyMakeBorder(board.generateImage((900, 700)), 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    path = tmp_path / "标定图.png"
    cv2.imencode(".png", image)[1].tofile(path)
    points, ids, size = calibration._detect_image(path, board)
    assert points.shape == (48, 2)
    assert len(np.unique(ids)) == 48
    assert size == (980, 780)
