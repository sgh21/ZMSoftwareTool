"""Tests for S1 parameter identification, persistence, and UI integration."""

from __future__ import annotations

import os
import pickle
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import yaml
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QSpinBox, QWidget

from app.pages.calibration_page import CalibrationPage
from app.pages.initialization_page import InitializationPage
from core.calibration_persistence import (
    list_identification_history,
    load_identification_result,
    record_identification_history,
    save_identification_result,
)
from core.calibration_service import CalibrationResult, CalibrationService, IdentificationOptions


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def child(widget, cls, name: str):
    found = widget.findChild(cls, name)
    assert found is not None, f"missing widget: {name}"
    return found


def make_synthetic_dataset(service: CalibrationService, sample_count: int = 12) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    joints = rng.uniform(
        low=[-1.5, -2.0, -2.0, -2.0, -2.0, -2.0],
        high=[1.5, -0.5, 0.0, 2.0, 2.0, 2.0],
        size=(sample_count, 6),
    )
    truth = np.zeros(33, dtype=float)
    truth[1] = 2.0e-4
    truth[8] = -3.0e-4
    truth[24] = 1.0e-4
    measured = service.model.batch_positions(joints, truth, service.geometric_parameters)
    return joints, measured


def fast_s1_options() -> IdentificationOptions:
    return IdentificationOptions(
        max_nfev=30,
        cv_folds=2,
        lambda_grid=(1.0e-10,),
        subspace_k_candidates=(2,),
        subspace_min_cluster_size=3,
    )


def write_pkl(path: Path, joints: np.ndarray, measured: np.ndarray) -> Path:
    with path.open("wb") as file:
        pickle.dump({"joints": joints, "measured_positions": measured}, file)
    return path


def test_nominal_config_file_is_used(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    nominal_config = {
        "nominal_robot": {
            "base_xyz": [0.0, 0.0, 0.0],
            "base_rpy": [0.0, 0.0, 0.0],
            "tool_xyz": [0.0, 0.0, 0.0],
            "tool_rpy": [0.0, 0.0, 0.0],
            "mdh": {
                "alpha": [0.0, 1.5707963267948966, 0.0, 0.0, 1.5707963267948966, -1.5707963267948966],
                "a": [0.0, 0.0, -0.612, -0.5723, 0.0, 0.0],
                "d": [0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922],
                "theta_offset": [0.0] * 6,
            },
        }
    }
    (config_dir / "nominal_robot.yaml").write_text(
        yaml.safe_dump(nominal_config), encoding="utf-8"
    )

    service = CalibrationService(project_root=tmp_path)
    pos = service.compute_nominal_position([0.0, -58.0, 82.0, -112.0, -90.0, 0.0])
    assert pos.shape == (3,)
    assert np.all(np.isfinite(pos))


def test_load_identification_data_accepts_multiple_pkl_files(tmp_path: Path) -> None:
    service = CalibrationService()
    joints, measured = make_synthetic_dataset(service, sample_count=12)
    first = write_pkl(tmp_path / "a.pkl", joints[:5], measured[:5])
    second = write_pkl(tmp_path / "b.pkl", joints[5:], measured[5:])

    data = service.load_identification_data([first, second])
    assert data["joints"].shape == (12, 6)
    assert data["measured_positions"].shape == (12, 3)
    assert data["sample_counts"] == [5, 7]
    assert len(data["dataset_paths"]) == 2


def test_s1_identification_wraps_algorithm_and_separates_error_concepts() -> None:
    service = CalibrationService()
    joints, measured = make_synthetic_dataset(service, sample_count=12)

    result = service.run_identification(
        joints,
        measured,
        options=fast_s1_options(),
        joint_unit="radians",
    )

    assert result.success
    assert result.method == "S1"
    assert result.selected_lambda == pytest.approx(1.0e-10)
    assert len(result.cv_scores) == 1
    assert result.rmse_mm < result.nominal_to_measured_rmse_mm
    expected_position_errors = np.linalg.norm(
        result.predicted_positions - result.nominal_positions, axis=1
    ) * 1000.0
    assert result.per_sample_position_errors_mm == pytest.approx(expected_position_errors)
    assert result.position_error_rmse_mm == pytest.approx(
        float(np.sqrt(np.mean(expected_position_errors**2)))
    )


def test_identify_from_files_persists_active_model_for_live_prediction(tmp_path: Path) -> None:
    service = CalibrationService(project_root=tmp_path)
    joints, measured = make_synthetic_dataset(service, sample_count=12)
    pkl_path = write_pkl(tmp_path / "identification.pkl", joints, measured)

    result = service.identify_from_files(pkl_path, options=fast_s1_options())
    assert result.success

    state = service.compute_predicted_position(
        [0.0, -58.0, 82.0, -112.0, -90.0, 0.0],
        joint_unit="degrees",
    )
    direct_nominal = service.compute_nominal_position(
        [0.0, -58.0, 82.0, -112.0, -90.0, 0.0],
        joint_unit="degrees",
    )
    assert np.allclose(state.nominal_position, direct_nominal)
    assert state.error_norm_mm >= 0.0
    assert state.health_level in {"good", "warning", "critical"}


def test_identification_yaml_and_sqlite_history_round_trip(tmp_path: Path) -> None:
    yaml_path = save_identification_result(
        tmp_path / "result.yaml",
        {"delta_a_1": 1.0e-4},
        fit_rmse_mm=0.1,
        fit_max_error_mm=0.2,
        position_error_rmse_mm=0.3,
        position_error_max_mm=0.4,
        sample_count=12,
        confidence=98.0,
        method="S1",
        selected_lambda=1.0e-10,
        dataset_paths=["a.pkl", "b.pkl"],
        cv_scores=[{"lambda": 1.0e-10, "mean_rmse_mm": 0.1, "max_rmse_mm": 0.2}],
    )
    loaded = load_identification_result(yaml_path)
    assert loaded["method"] == "S1"
    assert loaded["error_parameters"]["delta_a_1"] == pytest.approx(1.0e-4)
    assert loaded["position_error_rmse_mm"] == pytest.approx(0.3)

    db_path = tmp_path / "history.sqlite"
    row_id = record_identification_history(
        db_path,
        result_yaml_path=yaml_path,
        method="S1",
        success=True,
        message="ok",
        sample_count=12,
        fit_rmse_mm=0.1,
        fit_max_error_mm=0.2,
        position_error_rmse_mm=0.3,
        position_error_max_mm=0.4,
        selected_lambda=1.0e-10,
        confidence=98.0,
        dataset_paths=["a.pkl", "b.pkl"],
    )
    rows = list_identification_history(db_path)
    assert row_id == 1
    assert rows[0]["method"] == "S1"
    assert rows[0]["dataset_paths"] == ["a.pkl", "b.pkl"]


class FakeCalibrationService:
    def __init__(self) -> None:
        self._real = CalibrationService()

    def compute_nominal_position(self, joint_angles, *, joint_unit="auto"):
        return self._real.compute_nominal_position(joint_angles, joint_unit=joint_unit)

    def load_identification_data(self, paths):
        joints, measured = make_synthetic_dataset(self._real, sample_count=8)
        return {
            "joints": joints,
            "measured_positions": measured,
            "payloads": np.zeros(len(joints)),
            "dataset_paths": [str(path) for path in paths],
            "sample_counts": [len(joints)],
        }

    def run_identification(self, joint_configs, measured_positions, **kwargs):
        nominal = self._real.compute_nominal_positions(joint_configs, joint_unit="radians")
        predicted = nominal + np.array([0.0001, 0.0, 0.0])
        parameters = self._real.geometric_parameters
        return CalibrationResult(
            success=True,
            message="ok",
            method="S1",
            nominal_positions=nominal,
            predicted_positions=predicted,
            calibrated_positions=predicted,
            measured_positions=np.asarray(measured_positions),
            positioning_errors=predicted - nominal,
            error_vector=np.zeros(len(parameters)),
            error_parameters=parameters,
            parameter_names=[param.name for param in parameters],
            parameter_values={"delta_a_1": 1.0e-4},
            rmse_mm=0.05,
            max_error_mm=0.08,
            position_error_rmse_mm=0.1,
            position_error_max_mm=0.1,
            joint_count=len(joint_configs),
            confidence=99.0,
            selected_lambda=1.0e-10,
            cv_scores=[{"lambda": 1.0e-10, "mean_rmse_mm": 0.05, "max_rmse_mm": 0.08}],
            dataset_paths=[str(path) for path in kwargs.get("dataset_paths", [])],
        )


def test_calibration_page_loads_multiple_files_runs_and_persists(
    qapp: QApplication, tmp_path: Path
) -> None:
    page = CalibrationPage(project_root=tmp_path, calibration_service=FakeCalibrationService())
    first = tmp_path / "a.pkl"
    second = tmp_path / "b.pkl"
    first.write_bytes(b"placeholder")
    second.write_bytes(b"placeholder")

    page._load_calib_data([first, second])
    assert child(page, QPushButton, "run_calibration_button").isEnabled()
    assert "2 个文件" in child(page, QLabel, "data_info_label").text()

    page._run_calibration()
    deadline = 3000
    elapsed = 0
    while page._identification_thread is not None and elapsed < deadline:
        qapp.processEvents()
        QTest.qWait(20)
        elapsed += 20
    assert child(page, QLabel, "rmse_label").text() == "0.1000 mm"
    assert (tmp_path / "config" / "calibration_result.yaml").exists()
    assert list_identification_history(tmp_path / "storage" / "records" / "identification_history.sqlite")


def test_initialization_page_realtime_accuracy_labels_are_reasonable(
    qapp: QApplication, tmp_path: Path
) -> None:
    (tmp_path / "config").mkdir()
    save_identification_result(
        tmp_path / "config" / "calibration_result.yaml",
        {"delta_a_2": 1.0e-3},
        fit_rmse_mm=0.1,
        fit_max_error_mm=0.2,
        position_error_rmse_mm=1.0,
        position_error_max_mm=1.0,
        sample_count=12,
        confidence=95.0,
        method="S1",
        selected_lambda=1.0e-10,
    )

    page = InitializationPage(project_root=tmp_path)
    assert page.active_parameters_loaded
    assert child(page, QLabel, "accuracy_value_1").text().endswith("mm")
    assert child(page, QLabel, "accuracy_value_2").text().endswith("mm")
    assert child(page, QLabel, "health_value_1").text() == "95%"
    assert "\n" in child(page, QLabel, "health_ring_label").text()
    assert page.findChild(QWidget, "health_gauge") is not None
    # Red dot removed - health_status_dot should no longer exist
    assert page.findChild(QLabel, "health_status_dot") is None
    assert page.findChild(QPushButton, "refresh_accuracy_button") is None
    threshold = child(page, QSpinBox, "threshold_spin")
    threshold.setValue(500)
    threshold.editingFinished.emit()
    assert child(page, QLabel, "accuracy_value_3").text().endswith("mm")
    assert child(page, QLabel, "accuracy_value_4").text() in {"\u6b63\u5e38", "\u8d85\u5dee"}
    assert page.findChild(QLabel, "accuracy_value_5") is None
    assert page.findChild(QLabel, "accuracy_alarm_dot") is not None
    # health_state_cell removed with the red dot
    assert page.findChild(QWidget, "health_state_cell") is None
