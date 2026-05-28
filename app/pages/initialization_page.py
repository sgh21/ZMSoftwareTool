from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import yaml
from PySide6.QtCore import QRectF, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.widgets.robot_simulation_widget import DEFAULT_JOINT_DEGREES, RobotSimulationWidget
from core.calibration_service import CalibrationService


MODEL_SUFFIXES = {".urdf", ".xacro", ".stl", ".dae", ".obj"}
PARAM_SUFFIXES = {".yaml", ".yml", ".json"}
DEFAULT_STATUS_COLORS = {
    "alarm": "#db3444",
    "normal": "#45db34",
    "warning": "#db7734",
}


class CardFrame(QFrame):
    def __init__(self, object_name: str = "") -> None:
        super().__init__()
        if object_name:
            self.setObjectName(object_name)
        self.setFrameShape(QFrame.Shape.NoFrame)


class HealthGaugeWidget(QWidget):
    """Compact ring gauge for health score and status level."""

    LEVEL_COLORS = {
        "good": "#22b573",
        "warning": "#f59e0b",
        "critical": "#db3444",
        "unknown": "#9aa6b2",
    }

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("health_gauge")
        self.setMinimumSize(116, 116)
        self.setMaximumSize(132, 132)
        self._score = 0.0
        self._level = "unknown"
        self._caption = "未初始化"

    def set_status(self, score: float | None, level: str | None) -> None:
        if score is None:
            self._score = 0.0
            self._level = "unknown"
            self._caption = "未初始化"
        else:
            self._score = max(0.0, min(100.0, float(score)))
            self._level = (level or "unknown").lower()
            self._caption = self._level
        self.update()

    @property
    def level_color(self) -> str:
        return self.LEVEL_COLORS.get(self._level, self.LEVEL_COLORS["unknown"])

    def paintEvent(self, event) -> None:  # type: ignore[override]
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        side = min(self.width(), self.height()) - 16
        rect = QRectF(
            (self.width() - side) / 2,
            (self.height() - side) / 2,
            side,
            side,
        )
        pen_width = 11
        painter.setPen(QPen(QColor("#dfe4ea"), pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(rect, 90 * 16, -360 * 16)

        painter.setPen(QPen(QColor(self.level_color), pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawArc(rect, 90 * 16, int(-360 * 16 * (self._score / 100.0)))

        painter.setPen(QColor("#172033"))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, f"{self._score:.0f}\n{self._caption}")


class InitializationPage(QWidget):
    def __init__(
        self,
        project_root: str | Path | None = None,
        open_url: Callable[[QUrl], bool] | None = None,
        navigate_to_calibration: Callable[[], None] | None = None,
    ) -> None:
        super().__init__()
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self._open_url = open_url or QDesktopServices.openUrl
        self._navigate_to_calibration = navigate_to_calibration
        self.status_colors = self._load_status_colors()
        self._calibration_service = CalibrationService(project_root=self.project_root)
        self.active_parameters_loaded = False
        self.model_loaded = False
        self.params_loaded = False
        self.current_joint_names: list[str] = []
        self.joint_name_labels: list[QLabel] = []
        self.joint_angle_spins: list[QDoubleSpinBox] = []
        self.accuracy_value_labels: list[QLabel] = []
        self.health_value_labels: list[QLabel] = []
        self.joint_debug_dialog: QDialog | None = None
        self.setObjectName("initialization_page")
        self._build_ui()
        self._apply_style()
        self._load_default_identification_parameters()
        self._load_default_robot_model()
        self._refresh_status()

    def _load_status_colors(self) -> dict[str, str]:
        theme_path = self.project_root / "config" / "theme.yaml"
        colors = dict(DEFAULT_STATUS_COLORS)
        if not theme_path.exists():
            return colors
        try:
            data = yaml.safe_load(theme_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return colors
        configured = data.get("status_colors", {}) if isinstance(data, dict) else {}
        if not isinstance(configured, dict):
            return colors
        for key in ("alarm", "normal", "warning"):
            value = configured.get(key)
            if isinstance(value, str) and value.startswith("#") and len(value) == 7:
                colors[key] = value
        return colors

    def _load_default_robot_model(self) -> None:
        default_urdf = self.project_root / "models" / "urdf" / "ur10.urdf"
        if default_urdf.exists():
            self.load_model_file(default_urdf)

    def _load_default_identification_parameters(self) -> None:
        for candidate in (
            self.project_root / "config" / "calibration_result.yaml",
            self.project_root / "storage" / "model_versions" / "active_calib_params.yaml",
        ):
            if candidate.exists():
                self.load_param_file(candidate)
                return

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        root_layout.addWidget(self._build_header())

        body = QWidget()
        body.setObjectName("body")
        body_layout = QGridLayout(body)
        body_layout.setContentsMargins(16, 14, 16, 8)
        body_layout.setHorizontalSpacing(14)
        body_layout.setVerticalSpacing(14)
        body_layout.setColumnStretch(0, 7)
        body_layout.setColumnStretch(1, 6)
        body_layout.setRowStretch(0, 1)
        body_layout.setRowStretch(1, 1)

        body_layout.addWidget(self._build_simulation_card(), 0, 0, 2, 1)
        body_layout.addWidget(self._build_settings_card(), 0, 1)
        body_layout.addWidget(self._build_guide_card(), 1, 1)

        root_layout.addWidget(body, stretch=1)
        root_layout.addWidget(self._build_footer())

    def _build_header(self) -> QWidget:
        header = QWidget()
        header.setObjectName("header")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(22, 0, 18, 0)
        header_layout.setSpacing(18)

        logo = QLabel("△")
        logo.setObjectName("logo_badge")
        logo.setAlignment(Qt.AlignmentFlag.AlignCenter)
        logo_path = self.project_root / "app" / "resources" / "app_logo.png"
        if logo_path.exists():
            pixmap = QPixmap(str(logo_path))
            logo.setText("")
            logo.setPixmap(
                pixmap.scaled(
                    34,
                    34,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        header_layout.addWidget(logo)

        title = QLabel("本地机器人数字孪生工具")
        title.setObjectName("app_title")
        header_layout.addWidget(title)

        for text in ("文件", "编辑", "查看", "视图", "校验"):
            button = QPushButton(text)
            button.setObjectName("nav_button")
            button.setFlat(True)
            header_layout.addWidget(button)
        self.joint_debug_menu_button = QPushButton("调试")
        self.joint_debug_menu_button.setObjectName("joint_debug_menu_button")
        self.joint_debug_menu_button.setFlat(True)
        self.joint_debug_menu_button.clicked.connect(self.show_joint_debug_window)
        header_layout.addWidget(self.joint_debug_menu_button)

        header_layout.addStretch(1)
        self.current_project_label = QLabel("当前项目：UR_示例产线_20240516")
        self.current_project_label.setObjectName("current_project_label")
        header_layout.addWidget(self.current_project_label)

        self.connection_status_label = QLabel("● 已连接")
        self.connection_status_label.setObjectName("connection_status_label")
        header_layout.addWidget(self.connection_status_label)

        self.config_status_label = QLabel("⚠ 未加载")
        self.config_status_label.setObjectName("config_status_label")
        header_layout.addWidget(self.config_status_label)

        for text in ("通知", "帮助", "设置", "−", "□", "×"):
            button = QToolButton()
            button.setText(text)
            button.setObjectName("header_tool_button")
            button.setAutoRaise(True)
            header_layout.addWidget(button)
        return header

    def _build_simulation_card(self) -> QWidget:
        card = CardFrame("simulation_card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)

        title_row = QHBoxLayout()
        title = QLabel("机器人实时仿真")
        title.setObjectName("card_title")
        self.simulation_status_label = QLabel("⚠ 配置未加载，等待初始化")
        self.simulation_status_label.setObjectName("simulation_status_label")
        title_row.addWidget(title)
        title_row.addWidget(self.simulation_status_label)
        title_row.addStretch(1)
        layout.addLayout(title_row)

        scene_card = CardFrame("scene_card")
        scene_layout = QGridLayout(scene_card)
        scene_layout.setContentsMargins(8, 8, 8, 8)
        scene_layout.setSpacing(0)

        self.robot_view = RobotSimulationWidget()
        self.robot_view.joint_angles_changed.connect(
            lambda _angles: self._refresh_realtime_accuracy()
        )
        scene_layout.addWidget(self.robot_view, 0, 0)

        tool_column = QWidget(scene_card)
        tool_column.setObjectName("view_tool_column")
        tool_layout = QVBoxLayout(tool_column)
        tool_layout.setContentsMargins(4, 4, 4, 4)
        tool_layout.setSpacing(8)
        for name, text in (
            ("select_tool_button", "↖"),
            ("rotate_tool_button", "⟳"),
            ("pan_tool_button", "↕"),
            ("zoom_tool_button", "⊕"),
            ("fit_tool_button", "⛶"),
            ("cube_tool_button", "□"),
        ):
            button = QToolButton()
            button.setObjectName(name)
            button.setText(text)
            tool_layout.addWidget(button)
        scene_layout.addWidget(tool_column, 0, 0, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        view_toolbar = QWidget(scene_card)
        view_toolbar.setObjectName("view_toolbar")
        view_toolbar_layout = QHBoxLayout(view_toolbar)
        view_toolbar_layout.setContentsMargins(6, 4, 6, 4)
        view_toolbar_layout.setSpacing(4)
        for name, text in (("home_view_button", "⌂"), ("cube_view_button", "◇"), ("fit_view_button", "⛶")):
            button = QToolButton()
            button.setObjectName(name)
            button.setText(text)
            if name in {"home_view_button", "fit_view_button"}:
                button.clicked.connect(self.robot_view.reset_camera_to_fit)
            view_toolbar_layout.addWidget(button)
        scene_layout.addWidget(
            view_toolbar,
            0,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
        )

        self.init_prompt_card = self._build_init_prompt()
        scene_layout.addWidget(self.init_prompt_card, 0, 0, Qt.AlignmentFlag.AlignCenter)

        status_cards = self._build_status_cards()
        scene_layout.addWidget(
            status_cards,
            0,
            0,
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
        )
        layout.addWidget(scene_card, stretch=1)
        return card

    def _build_init_prompt(self) -> QWidget:
        card = CardFrame("init_prompt_card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(34, 28, 34, 28)
        layout.setSpacing(14)

        title = QLabel("⚠ 未找到机器人三维模型与误差参数文件")
        title.setObjectName("prompt_title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        self.prompt_message_label = QLabel(
            "在当前配置目录中未找到可用的三维模型文件（.urdf/.xacro）\n"
            "或误差参数文件（calib_params.yaml）。\n"
            "请先加载所需文件以继续初始化。"
        )
        self.prompt_message_label.setObjectName("prompt_message_label")
        self.prompt_message_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.prompt_message_label)

        button_row = QHBoxLayout()
        button_row.setSpacing(18)
        self.load_model_button = QPushButton("◇ 加载三维模型")
        self.load_model_button.setObjectName("load_model_button")
        self.load_model_button.clicked.connect(self.choose_model_file)
        self.load_params_button = QPushButton("▣ 加载参数文件")
        self.load_params_button.setObjectName("load_params_button")
        self.load_params_button.clicked.connect(self.choose_param_file)
        button_row.addWidget(self.load_model_button)
        button_row.addWidget(self.load_params_button)
        layout.addLayout(button_row)

        self.open_default_dir_button = QPushButton("□ 打开默认目录")
        self.open_default_dir_button.setObjectName("open_default_dir_button")
        self.open_default_dir_button.clicked.connect(self.open_default_directory)
        layout.addWidget(self.open_default_dir_button, alignment=Qt.AlignmentFlag.AlignCenter)
        return card

    def _build_status_cards(self) -> QWidget:
        container = QWidget()
        container.setObjectName("status_cards")
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        metrics = CardFrame("accuracy_card")
        metrics_layout = QGridLayout(metrics)
        metrics_layout.setContentsMargins(18, 14, 18, 14)
        metrics_layout.setHorizontalSpacing(18)
        metrics_layout.setVerticalSpacing(13)
        metrics_layout.addWidget(self._section_title("⚙ 当前精度指标"), 0, 0, 1, 2)
        self.accuracy_value_labels = []
        for row, label in enumerate(("\u4f4d\u7f6e\u8bef\u5dee RMS", "\u6700\u5927\u8bef\u5dee", "\u8d85\u5dee\u9608\u503c", "\u5f53\u524d\u7ed3\u8bba"), start=1):
            metrics_layout.addWidget(QLabel(label), row, 0)
            value = QLabel("\u672a\u521d\u59cb\u5316" if label == "\u5f53\u524d\u7ed3\u8bba" else "--")
            value.setObjectName(f"accuracy_value_{row}")
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.accuracy_value_labels.append(value)
            if label == "\u5f53\u524d\u7ed3\u8bba":
                conclusion_cell = QWidget()
                conclusion_cell.setObjectName("accuracy_conclusion_cell")
                conclusion_layout = QHBoxLayout(conclusion_cell)
                conclusion_layout.setContentsMargins(0, 0, 0, 0)
                conclusion_layout.setSpacing(6)
                conclusion_layout.addStretch(1)
                self.accuracy_alarm_dot = QLabel()
                self.accuracy_alarm_dot.setObjectName("accuracy_alarm_dot")
                conclusion_layout.addWidget(self.accuracy_alarm_dot)
                conclusion_layout.addWidget(value)
                metrics_layout.addWidget(conclusion_cell, row, 1)
            else:
                metrics_layout.addWidget(value, row, 1)
        layout.addWidget(metrics)

        health = CardFrame("health_card")
        health_layout = QGridLayout(health)
        health_layout.setContentsMargins(18, 14, 18, 14)
        health_layout.setHorizontalSpacing(18)
        health_layout.setVerticalSpacing(13)
        health_title = QHBoxLayout()
        health_title.addWidget(self._section_title("♡ 健康状态"))
        health_layout.addLayout(health_title, 0, 0, 1, 3)

        self.health_gauge = HealthGaugeWidget()
        health_layout.addWidget(self.health_gauge, 1, 0, 3, 1)
        self.health_ring_label = QLabel("--\n未初始化", self)
        self.health_ring_label.setObjectName("health_ring_label")
        self.health_ring_label.setVisible(False)
        self.health_ring_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.health_value_labels = []
        for row, (label, value) in enumerate(
            (("模型置信度", "--"), ("当前状态", "未初始化"), ("更新时间", "--")),
            start=1,
        ):
            health_layout.addWidget(QLabel(label), row, 1)
            value_label = QLabel(value)
            value_label.setObjectName(f"health_value_{row}")
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.health_value_labels.append(value_label)
            if row == 2:
                state_cell = QWidget()
                state_cell.setObjectName("health_state_cell")
                state_layout = QHBoxLayout(state_cell)
                state_layout.setContentsMargins(0, 0, 0, 0)
                state_layout.setSpacing(6)
                self.health_status_dot = QLabel()
                self.health_status_dot.setObjectName("health_status_dot")
                state_layout.addWidget(self.health_status_dot)
                state_layout.addStretch(1)
                state_layout.addWidget(value_label)
                health_layout.addWidget(state_cell, row, 2)
            else:
                health_layout.addWidget(value_label, row, 2)
        layout.addWidget(health)
        return container

    def _build_joint_debug_panel(self) -> QWidget:
        panel = CardFrame("joint_debug_panel")
        panel.setMinimumWidth(260)
        layout = QGridLayout(panel)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(8)
        title = QLabel("UR10 关节角调试")
        title.setObjectName("joint_debug_title")
        layout.addWidget(title, 0, 0, 1, 3)

        self.joint_name_labels = []
        self.joint_angle_spins = []
        for row, default_value in enumerate(DEFAULT_JOINT_DEGREES, start=1):
            joint_name = self.current_joint_names[row - 1] if row <= len(self.current_joint_names) else f"J{row}"
            label = QLabel(joint_name)
            label.setObjectName(f"joint_name_label_{row}")
            self.joint_name_labels.append(label)
            spin = QDoubleSpinBox()
            spin.setObjectName(f"joint_angle_spin_{row}")
            spin.setRange(-360.0, 360.0)
            spin.setDecimals(2)
            spin.setSingleStep(1.0)
            spin.setValue(float(default_value))
            spin.setSuffix("°")
            self.joint_angle_spins.append(spin)
            layout.addWidget(label, row, 0)
            layout.addWidget(spin, row, 1, 1, 2)

        apply_button = QPushButton("应用关节角")
        apply_button.setObjectName("apply_joint_angles_button")
        apply_button.clicked.connect(self.apply_joint_debug_angles)
        layout.addWidget(apply_button, 7, 0, 1, 2)

        reset_button = QPushButton("Home")
        reset_button.setObjectName("reset_joint_angles_button")
        reset_button.clicked.connect(self.reset_joint_debug_angles)
        layout.addWidget(reset_button, 7, 2)
        return panel

    def show_joint_debug_window(self) -> None:
        if self.joint_debug_dialog is None:
            dialog = QDialog(self)
            dialog.setObjectName("joint_debug_dialog")
            dialog.setWindowTitle("UR10 关节角调试")
            dialog.setModal(False)
            layout = QVBoxLayout(dialog)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(self._build_joint_debug_panel())
            self.joint_debug_dialog = dialog
        self.joint_debug_dialog.show()
        self.joint_debug_dialog.raise_()
        self._set_status_message("已打开 UR10 关节角调试窗口")

    def apply_joint_debug_angles(self) -> None:
        values = [spin.value() for spin in self.joint_angle_spins]
        self.robot_view.set_joint_angles(values)
        self._refresh_realtime_accuracy()
        self._set_status_message("已应用 UR10 调试关节角")

    def reset_joint_debug_angles(self) -> None:
        for spin, value in zip(self.joint_angle_spins, DEFAULT_JOINT_DEGREES, strict=False):
            spin.setValue(float(value))
        self.robot_view.reset_home_pose()
        self._refresh_realtime_accuracy()
        self._set_status_message("已恢复 UR10 Home 调试姿态")

    def _sync_joint_debug_labels(self, joint_names: list[str]) -> None:
        self.current_joint_names = list(joint_names)
        for index, label in enumerate(self.joint_name_labels):
            name = joint_names[index] if index < len(joint_names) else f"J{index + 1}"
            label.setText(name)

    def _build_settings_card(self) -> QWidget:
        card = CardFrame("settings_card")
        layout = QGridLayout(card)
        layout.setContentsMargins(22, 16, 22, 16)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(12)
        layout.setColumnStretch(1, 1)

        layout.addWidget(self._section_title("⚙ 常用设置"), 0, 0, 1, 3)

        threshold_label = QLabel("超差阈值")
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setObjectName("threshold_spin")
        self.threshold_spin.setRange(0, 10000)
        self.threshold_spin.setValue(300)
        self.threshold_spin.setSuffix(" μm")
        self.threshold_spin.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.UpDownArrows)
        self.threshold_spin.valueChanged.connect(lambda _value: self._refresh_realtime_accuracy())
        self.threshold_spin.editingFinished.connect(self._refresh_realtime_accuracy)
        layout.addWidget(threshold_label, 1, 0)
        layout.addWidget(self.threshold_spin, 1, 1)
        layout.addWidget(QLabel(""), 1, 2)

        self.model_path_edit = self._path_row(
            layout,
            2,
            "模型路径",
            "D:\\Projects\\UR_示例产线\\model\\",
            self.choose_model_file,
            "model_path_edit",
        )
        self.param_path_edit = self._path_row(
            layout,
            3,
            "参数文件",
            "D:\\Projects\\UR_示例产线\\config\\calib_params.yaml",
            self.choose_param_file,
            "param_path_edit",
        )

        layout.addWidget(QLabel("校验协议"), 4, 0)
        self.protocol_combo = QComboBox()
        self.protocol_combo.setObjectName("protocol_combo")
        self.protocol_combo.addItems(("默认校验协议集（24点）", "快速校验协议（8点）", "完整校验协议（48点）"))
        layout.addWidget(self.protocol_combo, 4, 1)
        manage_button = QPushButton("管理")
        manage_button.setObjectName("manage_protocol_button")
        manage_button.clicked.connect(lambda: self._set_status_message("校验协议管理暂未实现"))
        layout.addWidget(manage_button, 4, 2)

        layout.addWidget(QLabel("扫描点数"), 5, 0)
        self.scan_point_spin = QSpinBox()
        self.scan_point_spin.setObjectName("scan_point_spin")
        self.scan_point_spin.setRange(1, 999)
        self.scan_point_spin.setValue(20)
        layout.addWidget(self.scan_point_spin, 5, 1)

        self.scan_on_start_checkbox = QCheckBox("启动时扫描配置目录")
        self.scan_on_start_checkbox.setObjectName("scan_on_start_checkbox")
        layout.addWidget(self.scan_on_start_checkbox, 6, 1)
        layout.addWidget(QLabel("ⓘ"), 6, 2)

        self.calibration_button = QPushButton("▶ 进入标定分析")
        self.calibration_button.setObjectName("calibration_button")
        self.calibration_button.clicked.connect(self._on_navigate_to_calibration)
        self.calibration_button.setEnabled(False)
        layout.addWidget(self.calibration_button, 7, 1)

        self.config_warning_label = QLabel("⚠ 配置不完整：请加载三维模型与参数文件以完成初始化")
        self.config_warning_label.setObjectName("config_warning_label")
        layout.addWidget(self.config_warning_label, 8, 0, 1, 3)
        return card

    def _path_row(
        self,
        layout: QGridLayout,
        row: int,
        label: str,
        text: str,
        callback: Callable[[], None],
        object_name: str,
    ) -> QLineEdit:
        layout.addWidget(QLabel(label), row, 0)
        edit = QLineEdit(text)
        edit.setObjectName(object_name)
        edit.setReadOnly(True)
        layout.addWidget(edit, row, 1)
        button = QToolButton()
        button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        button.setToolTip("选择文件")
        button.setObjectName(f"{object_name}_browse_button")
        button.clicked.connect(callback)
        layout.addWidget(button, row, 2)
        return edit

    def _build_guide_card(self) -> QWidget:
        card = CardFrame("guide_card")
        layout = QGridLayout(card)
        layout.setContentsMargins(22, 16, 22, 16)
        layout.setHorizontalSpacing(18)
        layout.setVerticalSpacing(12)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(11)
        left_layout.addWidget(self._section_title("初始化指南"))
        for index, (title, desc) in enumerate(
            (
                ("加载三维模型", "选择机器人URDF或Xacro文件"),
                ("加载误差参数", "导入标定误差参数文件"),
                ("导入标定数据", "导入测量/标定数据文件"),
                ("开始分析", "初始化完成，开始精度分析"),
            ),
            start=1,
        ):
            row = QLabel(f"{index}  {title}\n    {desc}")
            row.setObjectName(f"guide_step_{index}")
            left_layout.addWidget(row)
        left_layout.addStretch(1)
        layout.addWidget(left, 0, 0, 2, 1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        recent_header = QHBoxLayout()
        recent_header.addWidget(self._section_title("最近项目"))
        recent_header.addStretch(1)
        all_records = QPushButton("查看全部记录  ›")
        all_records.setObjectName("all_records_button")
        all_records.clicked.connect(lambda: self._set_status_message("查看全部记录暂未实现"))
        recent_header.addWidget(all_records)
        right_layout.addLayout(recent_header)

        for index, (name, path, updated) in enumerate(
            (
                ("UR_示例产线_20240516", "D:\\Projects\\UR_示例产线\\", "2024-05-16 14:28"),
                ("UR_装配工作站_0321", "D:\\Projects\\UR_装配工作站\\", "2024-03-21 09:46"),
                ("UR_焊接单元_0220", "D:\\Projects\\UR_焊接单元\\", "2024-02-20 16:11"),
            ),
            start=1,
        ):
            right_layout.addWidget(self._recent_project_row(index, name, path, updated))

        template_header = QHBoxLayout()
        template_header.addWidget(self._section_title("示例模板"))
        template_header.addStretch(1)
        import_template = QPushButton("导入模板")
        import_template.setObjectName("import_template_button")
        import_template.clicked.connect(lambda: self._set_status_message("模板导入暂未实现"))
        template_header.addWidget(import_template)
        right_layout.addLayout(template_header)

        template_row = QHBoxLayout()
        for index, (name, desc) in enumerate(
            (("UR 通用模板", "24点校验协议"), ("装配应用模板", "定点+轨迹校验"), ("焊接应用模板", "焊缝轨迹校验")),
            start=1,
        ):
            button = QPushButton(f"▣\n{name}\n{desc}")
            button.setObjectName(f"template_button_{index}")
            button.clicked.connect(lambda checked=False, value=name: self._set_status_message(f"已选择模板：{value}"))
            template_row.addWidget(button)
        right_layout.addLayout(template_row)
        layout.addWidget(right, 0, 1, 2, 1)
        return card

    def _recent_project_row(self, index: int, name: str, path: str, updated: str) -> QWidget:
        row = QWidget()
        row.setObjectName(f"recent_project_row_{index}")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        icon = QLabel("▣")
        icon.setObjectName("recent_project_icon")
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon)

        text = QLabel(f"{name}\n{path}\n更新时间：{updated}")
        text.setObjectName(f"recent_project_label_{index}")
        layout.addWidget(text, stretch=1)

        button = QPushButton("打开")
        button.setObjectName(f"recent_project_open_button_{index}")
        button.clicked.connect(lambda checked=False, value=name: self._set_status_message(f"已选择最近项目：{value}"))
        layout.addWidget(button)
        return row

    def _build_footer(self) -> QWidget:
        footer = QWidget()
        footer.setObjectName("footer")
        layout = QHBoxLayout(footer)
        layout.setContentsMargins(18, 0, 18, 0)
        layout.setSpacing(22)
        for text in ("机器人：UR10", "控制器：CB3", "仿真频率：60 Hz", "数字孪生时间：--"):
            layout.addWidget(QLabel(text))
        layout.addStretch(1)
        layout.addWidget(QLabel("坐标系：基坐标系"))
        self.footer_status_label = QLabel("日志")
        self.footer_status_label.setObjectName("footer_status_label")
        layout.addWidget(self.footer_status_label)
        self.alarm_status_label = QLabel("● 无告警")
        self.alarm_status_label.setObjectName("alarm_status_label")
        layout.addWidget(self.alarm_status_label)
        return footer

    def _section_title(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("section_title")
        return label

    def choose_model_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "加载三维模型",
            str(self.project_root / "models"),
            "Robot model (*.urdf *.xacro *.stl *.dae *.obj)",
        )
        if path:
            self.load_model_file(Path(path))

    def choose_param_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "加载参数文件",
            str(self.project_root / "config"),
            "Parameter file (*.yaml *.yml *.json)",
        )
        if path:
            self.load_param_file(Path(path))

    def load_model_file(self, path: str | Path) -> None:
        model_path = Path(path)
        try:
            self._validate_existing_file(model_path, MODEL_SUFFIXES)
            with model_path.open("rb") as file:
                file.read(1)
            joint_names: list[str] = []
            if model_path.suffix.lower() == ".urdf":
                joint_names = self.robot_view.load_robot(model_path)
        except Exception as exc:  # noqa: BLE001
            self.model_loaded = False
            self._set_status_message(f"模型加载失败：{exc}")
            self._refresh_status()
            return
        self.model_loaded = True
        self.model_path_edit.setText(str(model_path))
        if joint_names:
            self._sync_joint_debug_labels(joint_names)
        self._set_status_message(f"模型已加载：{model_path.name}")
        self._refresh_status()

    def load_param_file(self, path: str | Path) -> None:
        param_path = Path(path)
        try:
            self._validate_existing_file(param_path, PARAM_SUFFIXES)
            data = self._read_parameter_file(param_path)
            if not isinstance(data, dict):
                raise TypeError("参数文件顶层必须是映射结构")
        except Exception as exc:  # noqa: BLE001
            self.params_loaded = False
            self.active_parameters_loaded = False
            self._set_status_message(f"参数加载失败：{exc}")
            self._refresh_status()
            return
        self.active_parameters_loaded = False
        try:
            self._calibration_service.load_active_parameters(param_path)
            self.active_parameters_loaded = True
        except Exception:
            self.active_parameters_loaded = False
        self.params_loaded = True
        self.param_path_edit.setText(str(param_path))
        self._set_status_message(f"参数文件已加载：{param_path.name}")
        self._refresh_status()

    def open_default_directory(self) -> None:
        self._open_url(QUrl.fromLocalFile(str(self.project_root)))
        self._set_status_message(f"已打开默认目录：{self.project_root}")

    def _on_navigate_to_calibration(self) -> None:
        if self._navigate_to_calibration is not None:
            self._navigate_to_calibration()

    def _validate_existing_file(self, path: Path, allowed_suffixes: set[str]) -> None:
        if not path.exists():
            raise FileNotFoundError(path)
        if not path.is_file():
            raise ValueError(f"不是文件：{path}")
        if path.suffix.lower() not in allowed_suffixes:
            allowed = ", ".join(sorted(allowed_suffixes))
            raise ValueError(f"不支持的文件类型：{path.suffix}，允许：{allowed}")

    def _read_parameter_file(self, path: Path) -> object:
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".json":
            return json.loads(text)
        return yaml.safe_load(text)

    def _refresh_realtime_accuracy(self) -> None:
        if not hasattr(self, "accuracy_value_labels") or not self.accuracy_value_labels:
            return
        threshold_mm = float(self.threshold_spin.value()) / 1000.0
        if not self.active_parameters_loaded:
            self.accuracy_value_labels[0].setText("--")
            self.accuracy_value_labels[1].setText("--")
            self.accuracy_value_labels[2].setText(f"{threshold_mm:.3f} mm")
            self.accuracy_value_labels[3].setText("\u672a\u52a0\u8f7d\u8fa8\u8bc6\u53c2\u6570")
            self._set_accuracy_alarm_dot("unknown")
            self.health_ring_label.setText("--\n\u672a\u521d\u59cb\u5316")
            self.health_gauge.set_status(None, None)
            self._set_health_dot("unknown")
            if self.health_value_labels:
                self.health_value_labels[0].setText("--")
                self.health_value_labels[1].setText("\u672a\u52a0\u8f7d\u8fa8\u8bc6\u53c2\u6570")
                self.health_value_labels[2].setText("--")
            return

        try:
            state = self._calibration_service.compute_predicted_position(
                self.robot_view.joint_degrees,
                joint_unit="degrees",
            )
        except Exception as exc:  # noqa: BLE001
            self.accuracy_value_labels[3].setText(f"\u8ba1\u7b97\u5931\u8d25\uff1a{exc}")
            self._set_accuracy_alarm_dot("critical")
            return

        is_over_limit = state.rms_mm > threshold_mm
        self.accuracy_value_labels[0].setText(f"{state.rms_mm:.3f} mm")
        self.accuracy_value_labels[1].setText(f"{state.max_error_mm:.3f} mm")
        self.accuracy_value_labels[2].setText(f"{threshold_mm:.3f} mm")
        self.accuracy_value_labels[3].setText("\u8d85\u5dee" if is_over_limit else "\u6b63\u5e38")
        self._set_accuracy_alarm_dot("critical" if is_over_limit else "good")
        self.health_ring_label.setText(f"{state.health_score:.0f}\n{state.health_level}")
        self.health_gauge.set_status(state.health_score, state.health_level)
        self._set_health_dot(state.health_level)
        if self.health_value_labels:
            self.health_value_labels[0].setText(f"{state.confidence:.0f}%")
            self.health_value_labels[1].setText(state.health_level)
            self.health_value_labels[2].setText("\u5b9e\u65f6")

    def _set_accuracy_alarm_dot(self, level: str) -> None:
        if not hasattr(self, "accuracy_alarm_dot"):
            return
        color = HealthGaugeWidget.LEVEL_COLORS.get(
            (level or "unknown").lower(),
            HealthGaugeWidget.LEVEL_COLORS["unknown"],
        )
        self.accuracy_alarm_dot.setStyleSheet(
            f"min-width: 12px; max-width: 12px; min-height: 12px; max-height: 12px;"
            f"border-radius: 2px; background: {color};"
        )

    def _set_health_dot(self, level: str) -> None:
        color = HealthGaugeWidget.LEVEL_COLORS.get(
            (level or "unknown").lower(),
            HealthGaugeWidget.LEVEL_COLORS["unknown"],
        )
        self.health_status_dot.setStyleSheet(
            f"min-width: 12px; max-width: 12px; min-height: 12px; max-height: 12px;"
            f"border-radius: 6px; background: {color};"
        )

    def _refresh_status(self) -> None:
        if self.model_loaded and self.params_loaded:
            self.config_status_label.setText("● 已加载")
            self.config_warning_label.setText("● 初始化完成：三维模型与参数文件已加载")
            self.simulation_status_label.setText("● 初始化完成")
            self.prompt_message_label.setText("模型与参数已加载，可进入精度分析流程。")
            self.alarm_status_label.setText("● 无告警")
        elif self.model_loaded:
            self.config_status_label.setText("⚠ 参数未加载")
            self.config_warning_label.setText("⚠ 配置不完整：请继续加载误差参数文件")
            self.simulation_status_label.setText("⚠ 模型已加载，等待参数")
            self.prompt_message_label.setText("已找到机器人模型，请继续加载误差参数文件。")
            self.alarm_status_label.setText("● 无告警")
        elif self.params_loaded:
            self.config_status_label.setText("⚠ 模型未加载")
            self.config_warning_label.setText("⚠ 配置不完整：请继续加载机器人三维模型")
            self.simulation_status_label.setText("⚠ 参数已加载，等待模型")
            self.prompt_message_label.setText("已找到误差参数，请继续加载机器人三维模型。")
            self.alarm_status_label.setText("● 无告警")
        else:
            self.config_status_label.setText("⚠ 未加载")
            self.config_warning_label.setText("⚠ 配置不完整：请加载三维模型与参数文件以完成初始化")
            self.simulation_status_label.setText("⚠ 配置未加载，等待初始化")
            self.prompt_message_label.setText(
                "在当前配置目录中未找到可用的三维模型文件（.urdf/.xacro）\n"
                "或误差参数文件（calib_params.yaml）。\n"
                "请先加载所需文件以继续初始化。"
            )
            self.alarm_status_label.setText("● 无告警")
        self.init_prompt_card.setVisible(not self.model_loaded)
        if hasattr(self, "calibration_button"):
            self.calibration_button.setEnabled(self.model_loaded)
        self._apply_status_colors()
        self._refresh_realtime_accuracy()

    def _apply_status_colors(self) -> None:
        normal = self.status_colors["normal"]
        warning = self.status_colors["warning"]
        self.connection_status_label.setStyleSheet(f"color: {normal}; font-weight: 600;")
        self.alarm_status_label.setStyleSheet(f"color: {normal}; font-weight: 600;")
        status_color = normal if self.model_loaded and self.params_loaded else warning
        self.config_status_label.setStyleSheet(f"color: {status_color}; font-weight: 600;")
        self.simulation_status_label.setStyleSheet(f"color: {status_color}; font-weight: 600;")
        self.config_warning_label.setStyleSheet(
            "border-radius: 6px; padding: 10px 12px; font-weight: 600;"
            f"color: {warning}; background: #fff7ed; border: 1px solid {warning};"
        )

    def _set_status_message(self, message: str) -> None:
        self.footer_status_label.setText(message)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget#initialization_page {
                background: #eef4fb;
                color: #172033;
                font-family: "Microsoft YaHei", "Segoe UI", Arial;
                font-size: 13px;
            }
            QWidget#header {
                min-height: 56px;
                max-height: 56px;
                background: #f7faff;
                border-bottom: 1px solid #d9e4f2;
            }
            QWidget#body {
                background: #eef4fb;
            }
            QWidget#footer {
                min-height: 40px;
                max-height: 40px;
                background: #f7faff;
                border-top: 1px solid #d9e4f2;
                color: #516075;
            }
            QLabel#logo_badge {
                min-width: 36px;
                max-width: 36px;
                min-height: 36px;
                max-height: 36px;
                border-radius: 0;
                color: #ffffff;
                background: transparent;
                font-weight: 700;
            }
            QLabel#app_title {
                font-size: 18px;
                font-weight: 700;
                color: #172033;
            }
            QPushButton#nav_button,
            QPushButton#joint_debug_menu_button {
                border: 0;
                color: #263750;
                padding: 8px 16px;
                font-weight: 600;
            }
            QLabel#connection_status_label,
            QLabel#alarm_status_label {
                color: #22b573;
                font-weight: 600;
            }
            QLabel#config_status_label,
            QLabel#simulation_status_label {
                color: #f59e0b;
                font-weight: 600;
            }
            QFrame#simulation_card,
            QFrame#settings_card,
            QFrame#guide_card,
            QFrame#scene_card {
                background: #f8fbff;
                border: 1px solid #d5e0ef;
                border-radius: 8px;
            }
            QFrame#scene_card {
                background: #f8fbff;
            }
            QLabel#card_title,
            QLabel#section_title {
                font-size: 16px;
                font-weight: 700;
                color: #162033;
            }
            QFrame#init_prompt_card {
                min-width: 385px;
                max-width: 450px;
                background: rgba(255, 255, 255, 242);
                border: 1px solid #dfe7f2;
                border-radius: 10px;
            }
            QLabel#prompt_title {
                font-size: 18px;
                font-weight: 700;
                color: #172033;
            }
            QLabel#prompt_message_label {
                color: #49566c;
                line-height: 150%;
            }
            QPushButton {
                min-height: 30px;
                border: 1px solid #cdd9eb;
                border-radius: 6px;
                background: #ffffff;
                padding: 5px 12px;
                color: #1d3557;
                font-weight: 600;
            }
            QPushButton:hover {
                border-color: #4f8df7;
                background: #f2f7ff;
            }
            QPushButton#load_model_button,
            QPushButton#load_params_button {
                min-height: 40px;
                color: #0f62d9;
                border: 1px solid #3982ff;
                background: #ffffff;
            }
            QPushButton#open_default_dir_button {
                min-width: 130px;
                color: #44546a;
            }
            QToolButton {
                min-width: 30px;
                min-height: 30px;
                border: 1px solid #d4dfed;
                border-radius: 6px;
                background: #ffffff;
                color: #46566f;
            }
            QToolButton:hover {
                border-color: #4f8df7;
                color: #0f62d9;
            }
            QWidget#view_tool_column,
            QWidget#view_toolbar {
                background: rgba(255, 255, 255, 226);
                border: 1px solid #e0e8f4;
                border-radius: 8px;
            }
            QWidget#status_cards {
                max-height: 265px;
            }
            QFrame#accuracy_card,
            QFrame#health_card {
                min-width: 250px;
                background: rgba(255, 255, 255, 238);
                border: 1px solid #dfe7f2;
                border-radius: 8px;
            }
            QLabel#health_ring_label {
                min-width: 96px;
                min-height: 96px;
                border-radius: 48px;
                border: 10px solid #dfe4ea;
                background: #f8fafc;
                font-size: 15px;
                font-weight: 700;
            }
            QWidget#health_gauge {
                background: transparent;
            }
            QLineEdit,
            QComboBox,
            QSpinBox {
                min-height: 30px;
                border: 1px solid #d4dfed;
                border-radius: 5px;
                background: #ffffff;
                padding: 2px 8px;
                color: #28364d;
            }
            QLabel#config_warning_label {
                color: #c76a00;
                background: #fff7ed;
                border: 1px solid #f5c27b;
                border-radius: 6px;
                padding: 10px 12px;
                font-weight: 600;
            }
            QLabel#recent_project_icon {
                min-width: 50px;
                max-width: 50px;
                min-height: 50px;
                max-height: 50px;
                border: 1px solid #d7e3f5;
                border-radius: 6px;
                background: #eef5ff;
                color: #0f62d9;
                font-size: 22px;
            }
            QPushButton#template_button_1,
            QPushButton#template_button_2,
            QPushButton#template_button_3 {
                min-height: 84px;
                text-align: left;
            }
            QPushButton#calibration_button {
                min-height: 38px;
                color: #ffffff;
                background: #0f62d9;
                border: 1px solid #0953c7;
                font-weight: 700;
            }
            QPushButton#calibration_button:hover {
                background: #2563eb;
            }
            QPushButton#calibration_button:disabled {
                background: #b0c4de;
                border-color: #8fa8c8;
                color: #6b7c93;
            }
            """
        )
