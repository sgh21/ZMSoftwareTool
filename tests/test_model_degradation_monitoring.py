from __future__ import annotations

import importlib
import math
from pathlib import Path

import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from core.accuracy_evaluator import confidence_from_uncertainty
from core.calibration_persistence import save_identification_result
from core.calibration_service import CalibrationService
from core.model_degradation_monitoring import (
    ModelDegradationMonitoringService,
    rigid_inverse,
    se3_log,
)
from core.nominal_parameter_service import nominal_after_applying_error_parameters
from core.parameter_repository import ParameterFileRepository


def make_transform(translation, rotvec=(0.0, 0.0, 0.0)) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rotation.from_rotvec(np.asarray(rotvec, dtype=np.float64)).as_matrix()
    T[:3, 3] = np.asarray(translation, dtype=np.float64)
    return T


def write_observation_npz(path: Path, names: list[str], transforms: list[np.ndarray]) -> Path:
    np.savez(
        path,
        filenames=np.asarray(names),
        T_cam_board=np.stack(transforms, axis=0),
        reproj_mean_px=np.asarray([0.05] * len(names), dtype=np.float64),
    )
    return path


def write_model_state_yaml(
    path: Path,
    *,
    confidence: float = 90.0,
    position_uncertainty_rmse_mm: float = 0.2,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "identification": {
                    "confidence": confidence,
                    "metrics": {
                        "position_uncertainty_rmse_mm": position_uncertainty_rmse_mm
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def write_monitoring_config_yaml(
    path: Path,
    *,
    board_grid: tuple[int, int] = (39, 34),
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "model_monitoring": {
                    "hand_eye": {
                        "convention": "E_T_C",
                        "T_tool_camera": np.eye(4).tolist(),
                    },
                    "camera": {
                        "K": np.eye(3).tolist(),
                        "D": [0, 0, 0, 0, 0],
                        "board_grid": list(board_grid),
                        "square_size_mm": 0.5,
                    },
                    "evaluation": {"orientation_weight_mm_per_rad": 100.0},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def write_legacy_monitoring_yaml(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "identification": {
                    "confidence": 90.0,
                    "metrics": {"position_uncertainty_rmse_mm": 0.2},
                    "monitoring": {
                        "hand_eye": {
                            "convention": "E_T_C",
                            "T_tool_camera": np.eye(4).tolist(),
                        },
                        "camera": {
                            "K": np.eye(3).tolist(),
                            "D": [0, 0, 0, 0, 0],
                            "board_grid": [39, 34],
                            "square_size_mm": 0.5,
                        },
                        "evaluation": {"orientation_weight_mm_per_rad": 100.0},
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_pnp_module_no_longer_depends_on_external_controller_package() -> None:
    pnp = importlib.import_module("core.vision.pnp")
    T = pnp.makeT(np.eye(3), [1.0, 2.0, 3.0])

    assert T.shape == (4, 4)
    assert np.allclose(pnp.invT(T) @ T, np.eye(4))


def test_c1_monitoring_images_have_consistent_pnp_orientation() -> None:
    cv2 = pytest.importorskip("cv2")
    from core.vision.pnp import estimate_pose_from_image, relative_transform, rotation_angle_deg

    image_paths = [
        Path("data/calibration/c1/normal/calib_00.jpg"),
        Path("data/calibration/c1/abnormal/calib_00.jpg"),
    ]
    if not all(path.exists() for path in image_paths):
        pytest.skip("data/calibration/c1 monitoring images are not available.")

    repo = ParameterFileRepository(Path.cwd())
    camera_path = repo.active_path_for("camera_monitoring")
    if camera_path is None:
        pytest.skip("active camera_monitoring parameters are not available.")
    camera_document = yaml.safe_load(camera_path.read_text(encoding="utf-8"))
    camera = camera_document["payload"]["model_monitoring"]["camera"]
    K = np.asarray(camera["K"], dtype=np.float64)
    D = np.asarray(camera["D"], dtype=np.float64)
    board_grid = tuple(camera["board_grid"])
    square_size_mm = float(camera["square_size_mm"])

    poses = []
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        assert image is not None
        pose = estimate_pose_from_image(
            image,
            board_grid,
            square_size_mm,
            K,
            D,
            r_diag_preference=camera.get("r_diag_preference", "none"),
        )
        assert pose.success, pose.message
        poses.append(pose)

    relative = relative_transform(poses[0].T_cam_board, poses[1].T_cam_board)
    translation_delta_mm = float(np.linalg.norm(relative[:3, 3]) * 1000.0)
    rotation_delta_deg = rotation_angle_deg(relative[:3, :3])

    assert translation_delta_mm < 2.0
    assert rotation_delta_deg < 1.0


def test_se3_log_recovers_translation_and_rotation_norms() -> None:
    T = make_transform([0.001, 0.0, 0.0], [0.0, 0.0, math.radians(2.0)])
    xi = se3_log(T)

    assert np.linalg.norm(xi[:3]) * 1000.0 == pytest.approx(1.0001015, rel=1.0e-4)
    assert np.linalg.norm(xi[3:]) == pytest.approx(math.radians(2.0))


def test_degradation_evaluation_recovers_known_npz_relative_drift(tmp_path: Path) -> None:
    model_path = write_model_state_yaml(tmp_path / "config" / "calibration_result.yaml")
    write_monitoring_config_yaml(tmp_path / "config" / "model_monitoring.yaml")
    drift = make_transform([0.001, 0.0, 0.0])
    reference = write_observation_npz(tmp_path / "reference.npz", ["pose_a"], [np.eye(4)])
    current = write_observation_npz(tmp_path / "current.npz", ["pose_a"], [rigid_inverse(drift)])

    progress = []
    result = ModelDegradationMonitoringService(tmp_path).evaluate(
        reference,
        current,
        calibration_result_path=model_path,
        progress_callback=progress.append,
    )

    assert result.sample_count == 1
    assert result.position_drift_rms_mm == pytest.approx(1.0)
    assert result.orientation_drift_rms_deg == pytest.approx(0.0)
    assert result.position_uncertainty_after_mm == pytest.approx(math.sqrt(1.04))
    assert result.confidence_after == pytest.approx(
        confidence_from_uncertainty(result.position_uncertainty_after_mm, 0.5)
    )
    assert "整体位置漂移 RMS: 1.000000 mm" in result.format_log()
    assert progress[-1].current == progress[-1].total == 3
    assert any("已加载参考观测 PnP 结果" in item.message for item in progress)
    assert any(item.stage == "完成评估" for item in progress)


def test_degradation_update_backs_up_and_writes_model_state(tmp_path: Path) -> None:
    model_path = write_model_state_yaml(tmp_path / "config" / "calibration_result.yaml")
    write_monitoring_config_yaml(tmp_path / "config" / "model_monitoring.yaml")
    drift = make_transform([0.001, 0.0, 0.0])
    reference = write_observation_npz(tmp_path / "reference.npz", ["pose_a"], [np.eye(4)])
    current = write_observation_npz(tmp_path / "current.npz", ["pose_a"], [rigid_inverse(drift)])
    service = ModelDegradationMonitoringService(tmp_path)
    result = service.evaluate(reference, current, calibration_result_path=model_path)

    backup_path = service.apply_recommended_update(model_path, result)
    updated = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    section = updated["identification"]

    assert backup_path.exists()
    assert backup_path.parent == tmp_path / "storage" / "model_versions"
    assert section["confidence"] == pytest.approx(result.confidence_after)
    assert section["metrics"]["position_uncertainty_rmse_mm"] == pytest.approx(
        result.position_uncertainty_after_mm
    )
    assert section["monitoring"]["last_degradation_evaluation"]["sample_count"] == 1
    assert "hand_eye" not in section["monitoring"]
    assert "camera" not in section["monitoring"]


def test_degradation_update_appends_confidence_history_in_active_identified_model(
    tmp_path: Path,
) -> None:
    repo = ParameterFileRepository(tmp_path)
    model_path = repo.create_version(
        "identified_model",
        {
            "identification": {
                "confidence_current": 90.0,
                "confidence_history": [
                    {
                        "timestamp": "2026-06-12T12:00:00+08:00",
                        "value": 90.0,
                        "source": "test",
                        "reason": "initial",
                    }
                ],
                "metrics": {"position_uncertainty_rmse_mm": 0.2},
            }
        },
    )
    repo.activate_version("identified_model", model_path)
    camera_path = repo.create_version(
        "camera_monitoring",
        {
            "model_monitoring": {
                "hand_eye": {"T_tool_camera": np.eye(4).tolist()},
                "camera": {
                    "K": np.eye(3).tolist(),
                    "D": [0, 0, 0, 0, 0],
                    "board_grid": [39, 34],
                    "square_size_mm": 0.5,
                },
            }
        },
    )
    repo.activate_version("camera_monitoring", camera_path)
    drift = make_transform([0.001, 0.0, 0.0])
    reference = write_observation_npz(tmp_path / "reference.npz", ["pose_a"], [np.eye(4)])
    current = write_observation_npz(tmp_path / "current.npz", ["pose_a"], [rigid_inverse(drift)])
    service = ModelDegradationMonitoringService(tmp_path)
    result = service.evaluate(reference, current, calibration_result_path=model_path)

    backup_path = service.apply_recommended_update(model_path, result)
    updated = yaml.safe_load(model_path.read_text(encoding="utf-8"))
    section = updated["payload"]["identification"]

    assert backup_path is None
    assert not (tmp_path / "storage" / "model_versions").exists()
    assert section["confidence_current"] == pytest.approx(result.confidence_after)
    assert section["confidence_history"][-1]["value"] == pytest.approx(result.confidence_after)
    assert len(section["confidence_history"]) == 2
    assert section["metrics"]["position_uncertainty_rmse_mm"] == pytest.approx(
        result.position_uncertainty_after_mm
    )
    assert section["monitoring"]["last_degradation_evaluation"]["sample_count"] == 1


def test_legacy_model_embedded_monitoring_config_still_loads(tmp_path: Path) -> None:
    model_path = write_legacy_monitoring_yaml(tmp_path / "config" / "calibration_result.yaml")

    config = ModelDegradationMonitoringService(tmp_path).load_monitoring_config(model_path)

    assert np.allclose(config.T_tool_camera, np.eye(4))
    assert config.camera.board_grid == (39, 34)


def test_monitoring_config_survives_identification_result_overwrite(tmp_path: Path) -> None:
    write_monitoring_config_yaml(
        tmp_path / "config" / "model_monitoring.yaml",
        board_grid=(9, 8),
    )
    model_path = save_identification_result(
        tmp_path / "config" / "calibration_result.yaml",
        {"delta_a_2": 0.0},
        nominal_robot={"tool_xyz": [0.0, 0.0, 0.0]},
        identified_robot={"tool_xyz": [0.0, 0.0, 0.0]},
        fit_rmse_mm=0.0,
        fit_max_error_mm=0.0,
        position_error_rmse_mm=0.0,
        position_error_max_mm=0.0,
        sample_count=12,
    )

    config = ModelDegradationMonitoringService(tmp_path).load_monitoring_config(model_path)
    saved_model = yaml.safe_load(model_path.read_text(encoding="utf-8"))

    assert config.camera.board_grid == (9, 8)
    assert "monitoring" not in saved_model["identification"]


def test_active_model_uncertainty_inflates_live_accuracy_rms(tmp_path: Path) -> None:
    service = CalibrationService(project_root=tmp_path)
    errors = {"delta_a_2": 0.0}
    nominal = service._current_nominal_robot_config()
    identified = nominal_after_applying_error_parameters(nominal, errors)
    path = save_identification_result(
        tmp_path / "config" / "calibration_result.yaml",
        errors,
        nominal_robot=nominal,
        identified_robot=identified,
        fit_rmse_mm=0.0,
        fit_max_error_mm=0.0,
        position_error_rmse_mm=0.0,
        position_error_max_mm=0.0,
        sample_count=12,
        position_uncertainty_rmse_mm=1.25,
        confidence=88.0,
    )

    service.load_active_parameters(path)
    state = service.compute_predicted_position(
        [0.0, -58.0, 82.0, -112.0, -90.0, 0.0],
        joint_unit="degrees",
    )

    assert state.rms_mm == pytest.approx(1.25)
    assert state.max_error_mm == pytest.approx(1.25)
    assert state.model_rms_mm == pytest.approx(0.0)
    assert state.model_max_error_mm == pytest.approx(0.0)
    assert state.position_uncertainty_rmse_mm == pytest.approx(1.25)
    assert state.confidence == pytest.approx(88.0)
