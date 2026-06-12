from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import yaml
from PySide6.QtCore import Qt, QUrl
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QLabel,
    QDoubleSpinBox,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
    QWidget,
)

from app.dialogs.nominal_parameter_dialog import NominalParameterUpdateDialog
from app.pages.initialization_page import InitializationPage
from core.calibration_persistence import save_identification_result
from core.nominal_parameter_service import (
    NominalParameterService,
    nominal_after_applying_error_parameters,
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


def write_nominal_for_fk(root: Path, *, tool_z: float = 0.039) -> Path:
    path = root / "config" / "nominal_robot.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "nominal_robot": {
                    "base_xyz": [1.0, 2.0, 3.0],
                    "base_rpy": [0.1, 0.2, 0.3],
                    "tool_xyz": [0.0, 0.0, tool_z],
                    "tool_rpy": [0.0, 0.0, 0.0],
                    "mdh": {
                        "alpha": [0.0, 1.57, 0.0, 0.0, 1.57, -1.57],
                        "a": [0.0, -0.612, -0.5723, 0.0, 0.0, 0.0],
                        "d": [0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922],
                        "theta_offset": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    },
                    "joint_limits": [[-3.14, 3.14]] * 6,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def accuracy_metric_mm(page: InitializationPage) -> float:
    return float(child(page, QLabel, "accuracy_value_1").text().split()[0])


def nominal_position_for_current_view(page: InitializationPage) -> np.ndarray:
    return page._calibration_service.compute_nominal_position(
        page.robot_view.joint_degrees,
        joint_unit="degrees",
    )


def read_nominal_for_fk(root: Path) -> dict:
    data = yaml.safe_load((root / "config" / "nominal_robot.yaml").read_text(encoding="utf-8"))
    return data["nominal_robot"]


def identified_payload(root: Path, errors: dict[str, float]) -> dict:
    return nominal_after_applying_error_parameters(read_nominal_for_fk(root), errors)


def test_initialization_page_default_state(qapp: QApplication, tmp_path: Path) -> None:
    page = InitializationPage(project_root=tmp_path)

    assert child(page, QPushButton, "load_model_button").text().endswith("加载三维模型")
    assert child(page, QPushButton, "load_params_button").text().endswith("加载参数文件")
    assert child(page, QLabel, "config_status_label").text() == "⚠ 未加载"
    assert "配置不完整" in child(page, QLabel, "config_warning_label").text()
    assert "未找到机器人三维模型" in child(page, QLabel, "prompt_title").text()


def test_load_model_button_reads_file(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_file = tmp_path / "robot.urdf"
    model_file.write_text("<robot name='test_robot'><link name='base' /></robot>", encoding="utf-8")
    page = InitializationPage(project_root=tmp_path)

    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(model_file), "Robot model (*.urdf)"),
    )

    QTest.mouseClick(child(page, QPushButton, "load_model_button"), Qt.MouseButton.LeftButton)

    assert page.model_loaded is True
    assert child(page, QLabel, "config_status_label").text() == "⚠ 参数未加载"
    assert page.model_path_edit.text() == str(model_file)
    assert not page.findChild(QToolButton, "model_path_edit_browse_button").icon().isNull()
    assert "模型已加载" in page.footer_status_label.text()


def test_load_parameter_button_reads_yaml(
    qapp: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    param_file = tmp_path / "calib_params.yaml"
    write_nominal_for_fk(tmp_path)
    errors = {"delta_Btx": 0.001}
    save_identification_result(
        param_file,
        errors,
        nominal_robot=read_nominal_for_fk(tmp_path),
        identified_robot=identified_payload(tmp_path, errors),
        fit_rmse_mm=0.1,
        fit_max_error_mm=0.2,
        position_error_rmse_mm=1.0,
        position_error_max_mm=1.0,
        sample_count=12,
    )
    page = InitializationPage(project_root=tmp_path)

    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(param_file), "Parameter file (*.yaml)"),
    )

    QTest.mouseClick(child(page, QPushButton, "load_params_button"), Qt.MouseButton.LeftButton)

    assert page.params_loaded is True
    assert child(page, QLabel, "config_status_label").text() == "⚠ 模型未加载"
    assert page.param_path_edit.text() == str(param_file)
    assert "参数文件已加载" in page.footer_status_label.text()


def test_invalid_parameter_file_does_not_mark_loaded(qapp: QApplication, tmp_path: Path) -> None:
    param_file = tmp_path / "bad.json"
    param_file.write_text("{bad json", encoding="utf-8")
    page = InitializationPage(project_root=tmp_path)

    page.load_param_file(param_file)

    assert page.params_loaded is False
    assert child(page, QLabel, "config_status_label").text() == "⚠ 未加载"
    assert "参数加载失败" in page.footer_status_label.text()


def test_complete_configuration_updates_status(qapp: QApplication, tmp_path: Path) -> None:
    model_file = tmp_path / "robot.xacro"
    model_file.write_text("<robot name='test_robot' />", encoding="utf-8")
    write_nominal_for_fk(tmp_path)
    param_file = tmp_path / "calib_params.yaml"
    errors = {"delta_Btx": 0.001}
    save_identification_result(
        param_file,
        errors,
        nominal_robot=read_nominal_for_fk(tmp_path),
        identified_robot=identified_payload(tmp_path, errors),
        fit_rmse_mm=0.1,
        fit_max_error_mm=0.2,
        position_error_rmse_mm=1.0,
        position_error_max_mm=1.0,
        sample_count=12,
    )
    page = InitializationPage(project_root=tmp_path)

    page.load_model_file(model_file)
    page.load_param_file(param_file)

    assert page.model_loaded is True
    assert page.params_loaded is True
    assert child(page, QLabel, "config_status_label").text() == "● 已加载"
    assert "初始化完成" in child(page, QLabel, "config_warning_label").text()
    assert "模型与参数已加载" in page.prompt_message_label.text()


def test_default_project_loads_ur10_and_debug_angles(qapp: QApplication) -> None:
    page = InitializationPage(project_root=Path.cwd())

    assert page.model_loaded is True
    assert page.model_path_edit.text().endswith(str(Path("models") / "urdf" / "ur10.urdf"))
    assert page.robot_view.joint_names == [
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    ]
    assert page.robot_view.visual_shape_colors
    assert any(color[:3] != (1.0, 1.0, 1.0) for color in page.robot_view.visual_shape_colors)
    # 验证蓝色端盖颜色存在（hex #9ED6F5 → 0.62/0.84/0.96 区间）
    cap_colors = [
        c for c in page.robot_view.visual_shape_colors
        if c[2] > 0.90 and c[1] > 0.80  # blue cap: B high, G medium-high
    ]
    assert len(cap_colors) >= 5, f"Expected >=5 cap meshes, got {len(cap_colors)}"
    # 验证关节外壳颜色存在（hex #6E7570）
    joint_colors = [
        c for c in page.robot_view.visual_shape_colors
        if 0.40 < c[0] < 0.48 and abs(c[0] - c[1]) < 0.05  # joint: dark gray, near-uniform
    ]
    assert len(joint_colors) >= 8, f"Expected >=8 joint meshes, got {len(joint_colors)}"

    assert page.findChild(QPushButton, "open_joint_debug_button") is None
    QTest.mouseClick(child(page, QPushButton, "joint_debug_menu_button"), Qt.MouseButton.LeftButton)
    assert page.joint_debug_dialog is not None
    child(page, QDoubleSpinBox, "joint_angle_spin_1").setValue(15.0)
    QTest.mouseClick(child(page, QPushButton, "apply_joint_angles_button"), Qt.MouseButton.LeftButton)
    assert page.robot_view.joint_degrees[0] == 15.0
    assert "已应用 UR10 调试关节角" in page.footer_status_label.text()


def test_settings_dialog_menu_and_main_layout(
    qapp: QApplication, tmp_path: Path
) -> None:
    opened_urls: list[QUrl] = []
    page = InitializationPage(project_root=tmp_path, open_url=lambda url: opened_urls.append(url) or True)

    body = child(page, QWidget, "body")
    assert body.layout().count() == 1
    assert body.layout().itemAt(0).widget().objectName() == "simulation_card"
    assert page.findChild(QWidget, "guide_card") is None
    assert page.findChild(QPushButton, "recent_project_open_button_1") is None
    assert page.findChild(QPushButton, "template_button_1") is None

    edit_button = child(page, QPushButton, "edit_menu_button")
    assert edit_button.menu() is not None
    assert [action.text() for action in edit_button.menu().actions()] == ["设置"]

    page.show_settings_dialog()
    qapp.processEvents()
    assert page.settings_dialog is not None
    assert page.settings_dialog.isVisible()
    assert "已打开常用设置窗口" in page.footer_status_label.text()

    QTest.mouseClick(child(page, QPushButton, "open_default_dir_button"), Qt.MouseButton.LeftButton)
    assert opened_urls
    assert Path(opened_urls[0].toLocalFile()) == tmp_path.resolve()
    assert "已打开默认目录" in page.footer_status_label.text()


def test_main_layout_resizes_without_header_overflow(
    qapp: QApplication, tmp_path: Path
) -> None:
    page = InitializationPage(project_root=tmp_path)
    page.show()

    for width, height in ((1180, 700), (1280, 720), (1440, 810)):
        page.resize(width, height)
        qapp.processEvents()

        header = child(page, QWidget, "header")
        header_layout = header.layout()
        assert header_layout is not None
        for index in range(header_layout.count()):
            widget = header_layout.itemAt(index).widget()
            if widget is None or not widget.isVisible():
                continue
            geometry = widget.geometry()
            assert geometry.left() >= 0
            assert geometry.right() <= header.width()

        body = child(page, QWidget, "body")
        body_layout = body.layout()
        assert body_layout is not None
        assert body_layout.count() == 1
        simulation_card = body_layout.itemAt(0).widget()
        assert simulation_card is not None
        assert simulation_card.objectName() == "simulation_card"
        assert simulation_card.width() > 900
        assert simulation_card.height() > 560
        assert page.robot_view.width() > 800
        assert page.robot_view.height() > 500

    page.close()


def test_header_window_control_buttons(qapp: QApplication, tmp_path: Path) -> None:
    page = InitializationPage(project_root=tmp_path)
    page.show()
    qapp.processEvents()

    QTest.mouseClick(child(page, QToolButton, "window_minimize_button"), Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert page.isMinimized()

    page.showNormal()
    qapp.processEvents()
    QTest.mouseClick(child(page, QToolButton, "window_maximize_button"), Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert page.isMaximized()

    QTest.mouseClick(child(page, QToolButton, "window_maximize_button"), Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert not page.isMaximized()

    QTest.mouseClick(child(page, QToolButton, "window_close_button"), Qt.MouseButton.LeftButton)
    qapp.processEvents()
    assert not page.isVisible()


def test_file_menu_opens_nominal_parameter_update_dialog(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []
    result_path = tmp_path / "config" / "nominal_robot.yaml"

    class FakeNominalDialog:
        result_mode = "direct"

        def __init__(self, project_root: Path, parent=None) -> None:
            self.result_path = result_path
            calls.append(Path(project_root))

        def exec(self):
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        "app.pages.initialization_page.NominalParameterUpdateDialog",
        FakeNominalDialog,
    )
    page = InitializationPage(project_root=tmp_path)

    file_button = child(page, QPushButton, "file_menu_button")
    assert file_button.menu() is not None
    assert [action.text() for action in file_button.menu().actions()] == ["更新名义参数"]

    page.show_nominal_parameter_update_dialog()

    assert calls == [tmp_path.resolve()]
    assert "名义参数已更新" in page.footer_status_label.text()


def test_nominal_update_menu_uses_complete_identified_model_after_nominal_update(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    nominal_path = config_dir / "nominal_robot.yaml"
    nominal_path.write_text(
        """
nominal_robot:
  base_xyz: [1.0, 2.0, 3.0]
  base_rpy: [0.1, 0.2, 0.3]
  tool_xyz: [0.0, 0.0, 0.039]
  tool_rpy: [0.0, 0.0, 0.0]
  mdh:
    alpha: [0.0, 1.57, 0.0, 0.0, 1.57, -1.57]
    a: [0.0, -0.612, -0.5723, 0.0, 0.0, 0.0]
    d: [0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922]
    theta_offset: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
""",
        encoding="utf-8",
    )
    result_path = config_dir / "calibration_result.yaml"
    errors = {"delta_a_2": 0.02, "delta_Btx": 0.01}
    save_identification_result(
        result_path,
        errors,
        nominal_robot=read_nominal_for_fk(tmp_path),
        identified_robot=identified_payload(tmp_path, errors),
        fit_rmse_mm=0.1,
        fit_max_error_mm=0.2,
        position_error_rmse_mm=10.0,
        position_error_max_mm=10.0,
        sample_count=10,
    )
    page = InitializationPage(project_root=tmp_path)
    before = page._calibration_service.compute_predicted_position(
        [0.0, -58.0, 82.0, -112.0, -90.0, 0.0]
    )
    assert before.error_norm_mm > 0.0

    class FakeNominalDialog:
        result_mode = "identification"

        def __init__(self, project_root: Path, parent=None) -> None:
            self.result_path = result_path
            NominalParameterService(project_root).update_from_identification_file(result_path)

        def exec(self):
            return QDialog.DialogCode.Accepted

    monkeypatch.setattr(
        "app.pages.initialization_page.NominalParameterUpdateDialog",
        FakeNominalDialog,
    )

    page.show_nominal_parameter_update_dialog()
    after = page._calibration_service.compute_predicted_position(
        [0.0, -58.0, 82.0, -112.0, -90.0, 0.0]
    )

    assert after.error_norm_mm == pytest.approx(0.0)
    assert "名义参数已更新" in page.footer_status_label.text()


def test_nominal_update_dialog_modes_refresh_live_fk_and_accuracy_status(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_nominal_for_fk(tmp_path, tool_z=0.039)
    active_param_path = tmp_path / "config" / "calibration_result.yaml"
    active_errors = {"delta_theta_5": 0.01}
    save_identification_result(
        active_param_path,
        active_errors,
        nominal_robot=read_nominal_for_fk(tmp_path),
        identified_robot=identified_payload(tmp_path, active_errors),
        fit_rmse_mm=0.1,
        fit_max_error_mm=0.2,
        position_error_rmse_mm=1.0,
        position_error_max_mm=1.5,
        sample_count=10,
    )
    imported_param_path = tmp_path / "config" / "imported_result.yaml"
    imported_errors = {"delta_a_2": 0.02, "delta_Ttz": 0.5}
    save_identification_result(
        imported_param_path,
        imported_errors,
        nominal_robot=read_nominal_for_fk(tmp_path),
        identified_robot=identified_payload(tmp_path, imported_errors),
        fit_rmse_mm=0.1,
        fit_max_error_mm=0.2,
        position_error_rmse_mm=500.0,
        position_error_max_mm=500.0,
        sample_count=10,
    )
    page = InitializationPage(project_root=tmp_path)
    assert page.active_parameters_loaded is True

    initial_metric = accuracy_metric_mm(page)
    initial_nominal = nominal_position_for_current_view(page)

    def apply_dialog(edit_dialog) -> None:
        def exec_with_ui_actions(dialog: NominalParameterUpdateDialog):
            edit_dialog(dialog)
            QTest.mouseClick(
                child(dialog, QPushButton, "nominal_update_save_button"),
                Qt.MouseButton.LeftButton,
            )
            assert dialog.result_path == tmp_path / "config" / "nominal_robot.yaml"
            return QDialog.DialogCode.Accepted

        monkeypatch.setattr(NominalParameterUpdateDialog, "exec", exec_with_ui_actions)
        page.show_nominal_parameter_update_dialog()
        qapp.processEvents()

    def apply_rollback() -> None:
        def exec_with_rollback(dialog: NominalParameterUpdateDialog):
            rollback_button = child(dialog, QPushButton, "nominal_rollback_button")
            assert rollback_button.isEnabled()
            QTest.mouseClick(rollback_button, Qt.MouseButton.LeftButton)
            assert dialog.result_mode == "rollback"
            return QDialog.DialogCode.Accepted

        monkeypatch.setattr(NominalParameterUpdateDialog, "exec", exec_with_rollback)
        page.show_nominal_parameter_update_dialog()
        qapp.processEvents()

    def set_direct_tool(dialog: NominalParameterUpdateDialog) -> None:
        child(dialog, QComboBox, "nominal_update_mode_combo").setCurrentIndex(0)
        edit = child(dialog, QPlainTextEdit, "nominal_direct_yaml_edit")
        document = yaml.safe_load(edit.toPlainText())
        document["nominal_robot"]["tool_xyz"][2] = 0.2
        edit.setPlainText(yaml.safe_dump(document, sort_keys=False))

    apply_dialog(set_direct_tool)
    direct_metric = accuracy_metric_mm(page)
    direct_nominal = nominal_position_for_current_view(page)
    assert direct_metric != pytest.approx(initial_metric)
    assert not np.allclose(direct_nominal, initial_nominal)
    assert "名义参数已更新" in page.footer_status_label.text()

    def set_values_tool(dialog: NominalParameterUpdateDialog) -> None:
        child(dialog, QComboBox, "nominal_update_mode_combo").setCurrentIndex(1)
        child(dialog, QPlainTextEdit, "nominal_values_yaml_edit").setPlainText(
            """
nominal_values:
  tool_xyz: [0.0, 0.0, 0.05]
"""
        )

    apply_dialog(set_values_tool)
    values_metric = accuracy_metric_mm(page)
    values_nominal = nominal_position_for_current_view(page)
    assert values_metric != pytest.approx(direct_metric)
    assert not np.allclose(values_nominal, direct_nominal)

    def import_identification(dialog: NominalParameterUpdateDialog) -> None:
        child(dialog, QComboBox, "nominal_update_mode_combo").setCurrentIndex(2)
        child(dialog, QLineEdit, "nominal_identification_path_edit").setText(
            str(imported_param_path)
        )

    apply_dialog(import_identification)
    imported_metric = accuracy_metric_mm(page)
    imported_nominal = nominal_position_for_current_view(page)
    assert imported_metric != pytest.approx(values_metric)
    assert not np.allclose(imported_nominal, values_nominal)
    assert read_nominal_for_fk(tmp_path)["tool_xyz"][2] == pytest.approx(0.05)

    apply_rollback()
    rollback_metric = accuracy_metric_mm(page)
    rollback_nominal = nominal_position_for_current_view(page)
    assert rollback_metric == pytest.approx(values_metric)
    assert np.allclose(rollback_nominal, values_nominal)
    assert "名义参数已回退" in page.footer_status_label.text()
