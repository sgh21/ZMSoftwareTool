from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt, QUrl
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QLabel, QPushButton, QDoubleSpinBox, QToolButton

from app.pages.initialization_page import InitializationPage


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
    param_file.write_text("calibration:\n  version: test\n", encoding="utf-8")
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
    param_file = tmp_path / "calib_params.json"
    param_file.write_text('{"calibration": {"version": "test"}}', encoding="utf-8")
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


def test_default_directory_recent_project_and_template_buttons_respond(
    qapp: QApplication, tmp_path: Path
) -> None:
    opened_urls: list[QUrl] = []
    page = InitializationPage(project_root=tmp_path, open_url=lambda url: opened_urls.append(url) or True)

    QTest.mouseClick(child(page, QPushButton, "open_default_dir_button"), Qt.MouseButton.LeftButton)
    assert opened_urls
    assert Path(opened_urls[0].toLocalFile()) == tmp_path.resolve()
    assert "已打开默认目录" in page.footer_status_label.text()

    QTest.mouseClick(child(page, QPushButton, "recent_project_open_button_1"), Qt.MouseButton.LeftButton)
    assert "已选择最近项目" in page.footer_status_label.text()

    QTest.mouseClick(child(page, QPushButton, "template_button_1"), Qt.MouseButton.LeftButton)
    assert "已选择模板" in page.footer_status_label.text()
