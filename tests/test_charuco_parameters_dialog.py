"""ChArUco 与固定靶标位姿表单；不读取或覆盖用户参数。"""

from copy import deepcopy

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QComboBox, QDialog, QLabel, QScrollArea

from app.dialogs.robot_position_parameters_dialog import RobotPositionParametersDialog, TARGET_FIELDS
from core.algorithms.pose_fields import rotation_from_rpy_degrees
from core.services.position_monitoring_service import PositionMonitoringService


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def service(tmp_path):
    return PositionMonitoringService(root=tmp_path)


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


def charuco_parameters():
    target = np.eye(4)
    target[:3, :3] = rotation_from_rpy_degrees([23.1234567890123, -34.9876543210123, 157.654321012345])
    target[:3, 3] = [123.123456789012345, -432.98765432109876, 851.01234567891234]
    return {
        "board_type": "charuco", "square_size_mm": 20.123456789012345,
        "charuco": {
            "dictionary": "DICT_5X5_1000", "squares_xy": [9, 7],
            "marker_length_mm": 14.123456789012345, "marker_ids": list(range(100, 131)),
            "legacy_pattern": True,
        },
        "target_pose_base": target.tolist(), "end_frame": "tool0",
    }


def test_category_titles_are_separate_and_board_content_can_scroll(open_dialog):
    dialog = open_dialog()
    headings = {label.text() for label in dialog.findChildren(QLabel)
                if label.property("robotSectionTitle")}
    assert {"相机内参", "镜头畸变", "平移", "旋转", "标定板几何",
            "固定靶标在基座中的位姿（可选）"} <= headings
    assert all(not label.property("robotNote") for label in dialog.findChildren(QLabel)
               if label.property("robotSectionTitle"))
    assert dialog.tabs.count() == 3
    assert isinstance(dialog.tabs.widget(2), QScrollArea)
    assert dialog.tabs.widget(2).widgetResizable()
    assert all(combo.property("robotInput") for combo in dialog.findChildren(QComboBox))


def test_legacy_parameters_do_not_gain_default_keys_or_a_new_version(service, open_dialog):
    original = deepcopy(service.parameters)
    files = {path: path.read_bytes() for path in service.storage.rglob("*.json")}
    dialog = open_dialog()
    document = dialog._document()
    assert dialog.board_type.currentData() == "checkerboard"
    assert not dialog.checkerboard_panel.isHidden()
    assert dialog.charuco_panel.isHidden()
    assert "board_type" not in document
    assert "charuco" not in document
    assert "target_pose_base" not in document
    dialog._save()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert service.parameters == original
    assert {path: path.read_bytes() for path in service.storage.rglob("*.json")} == files


def test_existing_charuco_metadata_and_target_matrix_keep_full_precision(service, open_dialog):
    parameters = service.save_parameters(charuco_parameters())
    dialog = open_dialog()
    assert dialog.board_type.currentData() == "charuco"
    assert dialog.checkerboard_panel.isHidden()
    assert not dialog.charuco_panel.isHidden()
    assert dialog.fields["charuco_columns"].text() == "9"
    assert dialog.fields["charuco_rows"].text() == "7"
    assert "方格" in dialog.fields["charuco_columns"].accessibleName()
    assert any("tool0" in label.text() for label in dialog.findChildren(QLabel))
    document = dialog._document()
    assert document["charuco"] == parameters["charuco"]
    assert document["target_pose_base"] == parameters["target_pose_base"]
    assert document["square_size_mm"] == parameters["square_size_mm"]
    dialog._save()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert service.parameters["version"] == parameters["version"]


def test_new_charuco_fields_save_as_board_geometry(service, open_dialog):
    dialog = open_dialog()
    dialog.board_type.setCurrentIndex(dialog.board_type.findData("charuco"))
    for name, value in {"charuco_columns": "9", "charuco_rows": "7",
                        "marker_length": "14", "square_size": "20"}.items():
        dialog.fields[name].setText(value)
    dialog.charuco_dictionary.setCurrentText("DICT_6X6_250")
    dialog._save()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert service.parameters["board_type"] == "charuco"
    assert service.parameters["charuco"] == {
        "squares_xy": [9, 7], "marker_length_mm": 14, "dictionary": "DICT_6X6_250",
    }
    assert service.parameters["square_size_mm"] == 20
    assert "target_pose_base" not in service.parameters


def test_target_pose_can_be_set_and_cleared_without_changing_board_type(service, open_dialog):
    dialog = open_dialog()
    values = [100, -200, 850, 20, -30, 40]
    for name, value in zip(TARGET_FIELDS, values):
        dialog.fields[name].setText(str(value))
    dialog._save()
    assert dialog.result() == QDialog.DialogCode.Accepted
    pose = np.asarray(service.parameters["target_pose_base"])
    np.testing.assert_array_equal(pose[:3, 3], values[:3])
    np.testing.assert_allclose(pose[:3, :3], rotation_from_rpy_degrees(values[3:]))
    assert "board_type" not in service.parameters
    dialog = open_dialog()
    for name in TARGET_FIELDS:
        dialog.fields[name].clear()
    dialog._save()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert service.parameters["target_pose_base"] is None


def test_target_translation_edit_keeps_unchanged_rotation_exact(service, open_dialog):
    service.save_parameters(charuco_parameters())
    original = deepcopy(service.parameters["target_pose_base"])
    dialog = open_dialog()
    dialog.fields["target_tx"].setText("200")
    pose = np.asarray(dialog._document()["target_pose_base"])
    np.testing.assert_array_equal(pose[:3, :3], np.asarray(original)[:3, :3])
    np.testing.assert_array_equal(pose[1:3, 3], np.asarray(original)[1:3, 3])
    assert pose[0, 3] == 200


def test_partial_target_pose_does_not_save(service, open_dialog):
    original = deepcopy(service.parameters)
    dialog = open_dialog()
    dialog.fields["target_tx"].setText("100")
    dialog._save()
    assert dialog.result() != QDialog.DialogCode.Accepted
    assert dialog.fields["target_ty"].accessibleName() in dialog.error_label.text()
    assert service.parameters == original
