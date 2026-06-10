from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import yaml
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
)

from app.dialogs.workstation_verification_dialog import WorkstationVerificationDialog
from app.pages.initialization_page import InitializationPage
from core.calibration_persistence import save_identification_result
from core.calibration_service import CalibrationService
from core.nominal_parameter_service import nominal_after_applying_error_parameters
from core.workstation_verification import (
    WorkstationConfig,
    WorkstationVerificationService,
)


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


def write_nominal(root: Path, *, tool_z: float = 0.039) -> Path:
    path = root / "config" / "nominal_robot.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "nominal_robot": {
                    "base_xyz": [0.0, 0.0, 0.0],
                    "base_rpy": [0.0, 0.0, 0.0],
                    "tool_xyz": [0.0, 0.0, tool_z],
                    "tool_rpy": [0.0, 0.0, 0.0],
                    "mdh": {
                        "alpha": [0.0, 1.57, 0.0, 0.0, 1.57, -1.57],
                        "a": [0.0, -0.612, -0.5723, 0.0, 0.0, 0.0],
                        "d": [0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922],
                        "theta_offset": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def metric_value(text: str) -> float:
    return float(text.split()[0])


def read_nominal(root: Path) -> dict:
    data = yaml.safe_load((root / "config" / "nominal_robot.yaml").read_text(encoding="utf-8"))
    return data["nominal_robot"]


def identified_payload(root: Path, errors: dict[str, float]) -> dict:
    return nominal_after_applying_error_parameters(read_nominal(root), errors)


def test_workstation_pose_input_saves_and_loads(tmp_path: Path) -> None:
    service = WorkstationVerificationService(tmp_path)
    pose = (1.0, 2.0, 3.0, 0.1, 0.2, 0.3)

    saved = service.save_workstations([
        WorkstationConfig("OP10", "pose", pose=pose),
    ])
    loaded = service.load_workstations()

    assert saved == tmp_path / "config" / "workstations.yaml"
    assert loaded == [WorkstationConfig("OP10", "pose", pose=pose)]


def test_joint_input_uses_nominal_model_for_tcp_pose(tmp_path: Path) -> None:
    calibration_service = CalibrationService(project_root=tmp_path)
    service = WorkstationVerificationService(
        tmp_path,
        calibration_service=calibration_service,
    )
    joints = (130.0, -90.0, 90.0, -90.0, -90.0, 0.0)

    completed = service.complete_workstation(
        WorkstationConfig("OP20", "joint", joint_degrees=joints)
    )
    expected_pose = calibration_service.compute_nominal_pose(
        joints,
        joint_unit="degrees",
    )

    assert completed.pose is not None
    assert completed.pose == pytest.approx(expected_pose)


def test_workstation_preview_outputs_errors_and_alarm_status(tmp_path: Path) -> None:
    calibration_service = CalibrationService(project_root=tmp_path)
    calibration_service.set_active_parameters({"delta_Btx": 0.001}, confidence=96.0)
    service = WorkstationVerificationService(
        tmp_path,
        calibration_service=calibration_service,
    )
    joints = (130.0, -90.0, 90.0, -90.0, -90.0, 0.0)
    pose = tuple(calibration_service.compute_nominal_pose(joints, joint_unit="degrees"))

    previews = service.preview_workstations(
        [WorkstationConfig("OP30", "pose", pose=pose)],
        threshold_mm=0.5,
        initial_joint_degrees=joints,
    )

    assert len(previews) == 1
    assert previews[0].error_mm == pytest.approx(1.0)
    assert previews[0].threshold_mm == pytest.approx(0.5)
    assert previews[0].over_limit is True
    assert len(previews[0].joint_degrees) == 6


def test_workstation_dialog_uses_selector_and_separate_preview_window(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    calibration_service = CalibrationService(project_root=tmp_path)
    calibration_service.set_active_parameters({"delta_Btx": 0.001}, confidence=96.0)
    service = WorkstationVerificationService(
        tmp_path,
        calibration_service=calibration_service,
    )
    dialog = WorkstationVerificationDialog(
        tmp_path,
        verification_service=service,
        threshold_mm_getter=lambda: 0.5,
        current_joint_degrees_getter=lambda: [130.0, -90.0, 90.0, -90.0, -90.0, 0.0],
    )

    selector = child(dialog, QComboBox, "workstation_selector_combo")
    assert selector.count() == 1
    assert dialog.findChild(QTableWidget, "workstation_table") is None

    QTest.mouseClick(child(dialog, QPushButton, "add_workstation_button"), Qt.LeftButton)
    qapp.processEvents()
    assert selector.count() == 2
    assert selector.currentText() == "WS-002"

    child(dialog, QLineEdit, "workstation_id_edit").setText("OP-B")
    child(dialog, QComboBox, "workstation_input_mode_combo").setCurrentIndex(0)
    child(dialog, QDoubleSpinBox, "workstation_pose_x_spin").setValue(1.234)
    selector.setCurrentIndex(0)
    qapp.processEvents()
    assert selector.itemText(1) == "OP-B"

    QTest.mouseClick(child(dialog, QPushButton, "preview_workstation_accuracy_button"), Qt.LeftButton)
    qapp.processEvents()

    assert dialog.preview_dialog is not None
    assert dialog.preview_dialog.isVisible()
    preview_table = child(dialog.preview_dialog, QTableWidget, "workstation_preview_table")
    alarm_table = child(dialog.preview_dialog, QTableWidget, "workstation_alarm_table")
    assert preview_table.rowCount() == 2
    assert preview_table.item(0, 3).text() == "1.000"
    assert preview_table.item(0, 5).text() == "超差"
    assert preview_table.item(0, 5).background().color().name() == "#ffe4e6"
    assert alarm_table.rowCount() == 2
    assert alarm_table.item(0, 0).text() == "WS-001"


def test_workstation_input_mode_switch_preserves_user_values(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    dialog = WorkstationVerificationDialog(
        tmp_path,
        threshold_mm_getter=lambda: 0.5,
        current_joint_degrees_getter=lambda: [130.0, -90.0, 90.0, -90.0, -90.0, 0.0],
    )
    mode = child(dialog, QComboBox, "workstation_input_mode_combo")
    pose_x = child(dialog, QDoubleSpinBox, "workstation_pose_x_spin")
    joint_q1 = child(dialog, QDoubleSpinBox, "workstation_joint_q1_spin")

    mode.setCurrentIndex(0)
    pose_x.setValue(1.234)
    mode.setCurrentIndex(1)
    joint_q1.setValue(25.0)
    mode.setCurrentIndex(0)
    assert pose_x.value() == pytest.approx(1.234)
    mode.setCurrentIndex(1)
    assert joint_q1.value() == pytest.approx(25.0)


def test_verification_menu_exposes_workstation_entry(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    page = InitializationPage(project_root=tmp_path)
    menu_button = child(page, QPushButton, "verification_menu_button")

    assert menu_button.menu() is not None
    assert [action.text() for action in menu_button.menu().actions()] == [
        "加工工位精度校验",
        "标定分析",
    ]

    page.show_workstation_verification_dialog()
    assert page.workstation_verification_dialog is not None
    assert "已打开加工工位精度校验窗口" in page.footer_status_label.text()

    page.show_calibration_analysis_dialog()
    assert page.calibration_dialog is not None
    assert page.calibration_page is not None
    assert "已打开标定分析窗口" in page.footer_status_label.text()


def test_health_value_updates_when_threshold_changes(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    write_nominal(tmp_path)
    errors = {"delta_Btx": 0.001}
    save_identification_result(
        tmp_path / "config" / "calibration_result.yaml",
        errors,
        nominal_robot=read_nominal(tmp_path),
        identified_robot=identified_payload(tmp_path, errors),
        fit_rmse_mm=0.1,
        fit_max_error_mm=0.2,
        position_error_rmse_mm=1.0,
        position_error_max_mm=1.0,
        sample_count=12,
        confidence=95.0,
    )
    page = InitializationPage(project_root=tmp_path)
    threshold = child(page, QSpinBox, "threshold_spin")

    threshold.setValue(1000)
    qapp.processEvents()
    wide_score = page.health_gauge._score
    threshold.setValue(100)
    qapp.processEvents()
    tight_score = page.health_gauge._score

    assert tight_score < wide_score
    assert child(page, QLabel, "health_value_3").text() == "实时"


def test_realtime_accuracy_reload_external_nominal_model_file_change(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    write_nominal(tmp_path, tool_z=0.039)
    errors = {"delta_theta_5": 0.01}
    save_identification_result(
        tmp_path / "config" / "calibration_result.yaml",
        errors,
        nominal_robot=read_nominal(tmp_path),
        identified_robot=identified_payload(tmp_path, errors),
        fit_rmse_mm=0.1,
        fit_max_error_mm=0.2,
        position_error_rmse_mm=0.4,
        position_error_max_mm=0.4,
        sample_count=12,
        confidence=95.0,
    )
    page = InitializationPage(project_root=tmp_path)
    qapp.processEvents()
    before = metric_value(child(page, QLabel, "accuracy_value_1").text())

    write_nominal(tmp_path, tool_z=0.2)
    page._refresh_realtime_accuracy()
    qapp.processEvents()
    after = metric_value(child(page, QLabel, "accuracy_value_1").text())

    assert after > before * 1.5
    assert "名义模型文件已更新" in page.footer_status_label.text()


def test_health_value_updates_when_current_workstation_changes(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    parameter = "delta_alpha_2"
    first_joints = (130.0, -90.0, 90.0, -90.0, -90.0, 0.0)
    second_joints = (20.0, -120.0, 70.0, -80.0, -70.0, 25.0)
    calibration_service = CalibrationService(project_root=tmp_path)
    calibration_service.set_active_parameters({parameter: 0.01}, confidence=95.0)
    first_state = calibration_service.compute_predicted_position(
        first_joints,
        joint_unit="degrees",
        tolerance_mm=0.3,
    )
    second_state = calibration_service.compute_predicted_position(
        second_joints,
        joint_unit="degrees",
        tolerance_mm=0.3,
    )
    assert not np.isclose(first_state.health_score, second_state.health_score)

    write_nominal(tmp_path)
    errors = {parameter: 0.01}
    save_identification_result(
        tmp_path / "config" / "calibration_result.yaml",
        errors,
        nominal_robot=read_nominal(tmp_path),
        identified_robot=identified_payload(tmp_path, errors),
        fit_rmse_mm=0.1,
        fit_max_error_mm=0.2,
        position_error_rmse_mm=1.0,
        position_error_max_mm=1.0,
        sample_count=12,
        confidence=95.0,
    )
    page = InitializationPage(project_root=tmp_path)
    child(page, QSpinBox, "threshold_spin").setValue(300)
    service = WorkstationVerificationService(
        tmp_path,
        calibration_service=page._calibration_service,
    )
    first_preview, second_preview = service.preview_workstations(
        [
            WorkstationConfig("OP-A", "joint", joint_degrees=first_joints),
            WorkstationConfig("OP-B", "joint", joint_degrees=second_joints),
        ],
        threshold_mm=0.3,
        initial_joint_degrees=first_joints,
    )

    page._apply_workstation_preview_state(first_preview)
    qapp.processEvents()
    first_score = page.health_gauge._score
    page._apply_workstation_preview_state(second_preview)
    qapp.processEvents()

    assert page.health_gauge._score != pytest.approx(first_score)
    assert child(page, QLabel, "health_value_3").text() == "工位 OP-B"
