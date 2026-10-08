"""参数表单的数值含义、保存边界与已有标定结果保真。"""

from copy import deepcopy

import numpy as np
import pytest
from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QPlainTextEdit

from app.dialogs.robot_position_parameters_dialog import RobotPositionParametersDialog
from core.algorithms.pose_fields import rotation_from_rpy_degrees
from core.services.position_monitoring_service import PositionMonitoringService


@pytest.fixture
def service(tmp_path):
    service = PositionMonitoringService(root=tmp_path)
    hand_eye = np.eye(4)
    hand_eye[:3, :3] = rotation_from_rpy_degrees([23.5, -41.25, 108.75])
    hand_eye[:3, 3] = [12.125, -34.25, 56.875]
    service.save_parameters({
        "camera_matrix": [[3675.270712345678, 0.25, 1231.6813],
                          [0, 3674.6618, 1073.9725], [0, 0, 1]],
        "hand_eye": hand_eye.tolist(),
        "dist_coeffs": [-0.1157123456789123, 0.3235, 0.001, -0.0006, 0.021],
        "board_grid": [12, 9], "square_size_mm": 15,
        "reference_rotation": rotation_from_rpy_degrees([11.5, 17.25, -21.75]).tolist(),
        "source_note": "外部标定记录", "calibration_details": {"operator": "测试记录"},
    })
    return service


@pytest.fixture
def open_dialog(application, service):
    dialogs = []

    def create():
        dialog = RobotPositionParametersDialog(service)
        dialogs.append(dialog)
        return dialog

    yield create
    for dialog in dialogs:
        dialog.close()


def saved_documents(service):
    return {path: path.read_bytes() for path in service.storage.rglob("*.json")}


def test_existing_parameters_fill_individual_fields_with_units(open_dialog):
    dialog = open_dialog()
    assert not dialog.findChildren(QPlainTextEdit)
    expected = {
        "fx": 3675.270712345678, "fy": 3674.6618, "cx": 1231.6813, "cy": 1073.9725,
        "tx": 12.125, "ty": -34.25, "tz": 56.875,
        "roll": 23.5, "pitch": -41.25, "yaw": 108.75,
        "board_columns": 12, "board_rows": 9, "square_size": 15,
    }
    for name, value in expected.items():
        assert float(dialog.fields[name].text()) == pytest.approx(value)
    assert "px" in dialog.fields["fx"].accessibleName()
    assert "mm" in dialog.fields["tx"].accessibleName()
    assert "°" in dialog.fields["roll"].accessibleName()
    assert dialog.distortion_count.currentData() == 5


def test_save_updates_fields_and_preserves_old_version_and_metadata(service, open_dialog):
    original = deepcopy(service.parameters)
    original_file = service.parameter_path
    original_bytes = original_file.read_bytes()
    current_batch = {"batch_id": "待重新导入"}
    service.current_batch = current_batch
    dialog = open_dialog()
    for name, value in {
        "fx": "4100", "fy": "4200", "cx": "1300.5", "cy": "980.5",
        "tx": "80", "ty": "-12", "tz": "41.5",
        "roll": "30", "pitch": "-20", "yaw": "70",
        "board_columns": "10", "board_rows": "7", "square_size": "18.5",
    }.items():
        dialog.fields[name].setText(value)
    dialog._save()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert service.parameters["version"] != original["version"]
    assert service.previous_parameter_path.read_bytes() == original_bytes
    assert len(list((service.storage / "parameters").glob("*.json"))) == 2
    assert service.current_batch is None
    np.testing.assert_array_equal(
        service.parameters["camera_matrix"], [[4100, 0.25, 1300.5], [0, 4200, 980.5], [0, 0, 1]],
    )
    pose = np.asarray(service.parameters["hand_eye"])
    np.testing.assert_array_equal(pose[:3, 3], [80, -12, 41.5])
    np.testing.assert_allclose(pose[:3, :3], rotation_from_rpy_degrees([30, -20, 70]))
    assert service.parameters["board_grid"] == [10, 7]
    assert service.parameters["square_size_mm"] == 18.5
    for name in ("source_note", "calibration_details", "dist_coeffs", "reference_rotation"):
        assert service.parameters[name] == original[name]


@pytest.mark.parametrize("pitch", [17.1234567890123, 90, -90])
def test_unchanged_matrix_values_are_preserved_exactly(service, open_dialog, pitch):
    pose = np.asarray(service.parameters["hand_eye"])
    pose[:3, :3] = rotation_from_rpy_degrees([37.1234567890123, pitch, -123.987654321098])
    pose[:3, 3] = [12.123456789012345, -34.987654321098765, 56.11223344556677]
    service.save_parameters({
        "hand_eye": pose.tolist(), "reference_rotation": pose[:3, :3].tolist(),
        "square_size_mm": 15.123456789012345,
        "max_reprojection_error_px": 0.1234567890123456,
    })
    original = deepcopy(service.parameters)
    dialog = open_dialog()
    dialog._save()
    assert dialog.result() == QDialog.DialogCode.Accepted
    for name in ("camera_matrix", "hand_eye", "reference_rotation", "dist_coeffs",
                 "square_size_mm", "max_reprojection_error_px"):
        assert service.parameters[name] == original[name]


def test_complete_empty_optional_groups_clear_camera_and_hand_eye(service, open_dialog):
    dialog = open_dialog()
    for name in ("fx", "fy", "cx", "cy", "tx", "ty", "tz", "roll", "pitch", "yaw"):
        dialog.fields[name].clear()
    dialog._save()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert service.parameters["camera_matrix"] is None
    assert service.parameters["hand_eye"] is None


@pytest.mark.parametrize("name", ["cx", "roll", "reference_pitch"])
def test_partial_optional_group_shows_field_error_without_saving(service, open_dialog, name):
    original = deepcopy(service.parameters)
    original_files = saved_documents(service)
    current_batch = {"batch_id": "本次观测"}
    service.current_batch = current_batch
    dialog = open_dialog()
    dialog.fields[name].clear()
    dialog._save()
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert dialog.fields[name].accessibleName() in dialog.error_label.text()
    assert service.parameters == original
    assert service.current_batch is current_batch
    assert saved_documents(service) == original_files


@pytest.mark.parametrize("name, text, message", [
    ("fx", "非数值", "请填写数值"),
    ("fy", "-1", "焦距为正"),
    ("square_size", "0", "格长必须为正数"),
    ("board_columns", "7.5", "整数"),
    ("k1", "nan", "有限数值"),
])
def test_invalid_number_does_not_create_parameter_version(service, open_dialog, name, text, message):
    original = deepcopy(service.parameters)
    original_files = saved_documents(service)
    dialog = open_dialog()
    dialog.fields[name].setText(text)
    dialog._save()
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert message in dialog.error_label.text()
    assert service.parameters == original
    assert saved_documents(service) == original_files


def test_fourteen_distortion_coefficients_round_trip_and_can_be_edited(service, open_dialog):
    coefficients = [float(f"0.{index + 1:02}12345678901234") for index in range(14)]
    service.save_parameters({"dist_coeffs": coefficients})
    dialog = open_dialog()
    assert dialog.distortion_count.currentData() == 14
    assert dialog._document()["dist_coeffs"] == coefficients
    dialog.fields["k6"].setText("0.25")
    dialog.fields["tau_y"].setText("-0.002")
    dialog._save()
    assert dialog.result() == QDialog.DialogCode.Accepted
    expected = coefficients.copy()
    expected[7], expected[13] = 0.25, -0.002
    assert service.parameters["dist_coeffs"] == expected


@pytest.mark.parametrize("count", [0, 4, 5, 8, 12, 14])
def test_distortion_count_controls_saved_coefficients(service, open_dialog, count):
    dialog = open_dialog()
    dialog.distortion_count.setCurrentIndex(dialog.distortion_count.findData(count))
    document = dialog._document()
    assert len(document["dist_coeffs"]) == count
    for index, widgets in enumerate(dialog.distortion_widgets):
        assert all(widget.isHidden() == (index >= count) for widget in widgets)
    dialog._save()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert len(service.parameters["dist_coeffs"]) == count


def test_optional_reference_angles_and_reprojection_limit_can_be_set_and_cleared(service, open_dialog):
    service.save_parameters({"reference_rotation": None, "max_reprojection_error_px": None})
    dialog = open_dialog()
    assert all(not dialog.fields[name].text()
               for name in ("reference_roll", "reference_pitch", "reference_yaw", "reprojection_limit"))
    for name, value in {
        "reference_roll": "12", "reference_pitch": "25", "reference_yaw": "-40",
        "reprojection_limit": "0.75",
    }.items():
        dialog.fields[name].setText(value)
    dialog._save()
    assert dialog.result() == QDialog.DialogCode.Accepted
    np.testing.assert_allclose(service.parameters["reference_rotation"], rotation_from_rpy_degrees([12, 25, -40]))
    assert service.parameters["max_reprojection_error_px"] == 0.75
    dialog = open_dialog()
    for name in ("reference_roll", "reference_pitch", "reference_yaw", "reprojection_limit"):
        dialog.fields[name].clear()
    dialog._save()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert service.parameters["reference_rotation"] is None
    assert service.parameters["max_reprojection_error_px"] is None


def test_cancel_keeps_parameters_files_and_current_observations(service, open_dialog):
    original = deepcopy(service.parameters)
    original_files = saved_documents(service)
    current_batch = {"batch_id": "本次观测"}
    service.current_batch = current_batch
    dialog = open_dialog()
    dialog.fields["fx"].setText("999")
    dialog.fields["tx"].setText("100")
    buttons = dialog.findChild(QDialogButtonBox)
    buttons.button(QDialogButtonBox.StandardButton.Cancel).click()
    assert dialog.result() == QDialog.DialogCode.Rejected
    assert service.parameters == original
    assert service.current_batch is current_batch
    assert saved_documents(service) == original_files
