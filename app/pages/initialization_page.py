from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import yaml
from PySide6.QtCore import QObject, QRectF, Qt, QThread, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QDesktopServices, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QCheckBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.widgets.robot_simulation_widget import DEFAULT_JOINT_DEGREES, RobotSimulationWidget
from core.calibration_persistence import (
    record_identification_history,
    save_identification_result,
)
from core.calibration_service import CalibrationResult, CalibrationService, IdentificationOptions

MODEL_SUFFIXES = {".urdf", ".xacro", ".stl", ".dae", ".obj"}
PARAM_SUFFIXES = {".yaml", ".yml", ".json"}
DEFAULT_STATUS_COLORS = {
    "alarm": "#db3444",
    "normal": "#45db34",
    "warning": "#db7734",
}


class IdentificationWorker(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        service: CalibrationService,
        data: dict[str, np.ndarray],
        paths: list[Path],
    ) -> None:
        super().__init__()
        self._service = service
        self._data = data
        self._paths = list(paths)

    @Slot()
    def run(self) -> None:
        try:
            result = self._service.run_identification(
                self._data["joints"],
                self._data["measured_positions"],
                payloads=self._data.get("payloads"),
                directions=self._data.get("directions"),
                dataset_paths=self._paths,
                options=IdentificationOptions(),
            )
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)


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
    ) -> None:
        super().__init__()
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self._open_url = open_url or QDesktopServices.openUrl
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

        # Calibration state
        self._calib_result: CalibrationResult | None = None
        self._calib_data: dict[str, np.ndarray] | None = None
        self._calib_paths: list[Path] = []
        self._identification_thread: QThread | None = None
        self._identification_worker: IdentificationWorker | None = None
        self._identification_progress: QProgressDialog | None = None

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

        # Right-bottom: stacked widget toggles between guide and calibration
        self._right_bottom_stack = QStackedWidget()
        self._right_bottom_stack.setObjectName("right_bottom_stack")
        self._right_bottom_stack.addWidget(self._build_guide_card())
        self._right_bottom_stack.addWidget(self._build_calibration_card())
        body_layout.addWidget(self._right_bottom_stack, 1, 1)

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
        for row, label in enumerate(("位置误差 RMS", "最大误差", "超差阈值", "当前结论"), start=1):
            metrics_layout.addWidget(QLabel(label), row, 0)
            value = QLabel("未初始化" if label == "当前结论" else "--")
            value.setObjectName(f"accuracy_value_{row}")
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            self.accuracy_value_labels.append(value)
            if label == "当前结论":
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

        self.scan_on_start_checkbox = QCheckBox("启动时扫描配置目录")
        self.scan_on_start_checkbox.setObjectName("scan_on_start_checkbox")
        layout.addWidget(self.scan_on_start_checkbox, 4, 1)
        layout.addWidget(QLabel("ⓘ"), 4, 2)

        self.calibration_toggle_btn = QPushButton("▶ 进入标定分析")
        self.calibration_toggle_btn.setObjectName("calibration_button")
        self.calibration_toggle_btn.clicked.connect(self._toggle_calibration_panel)
        self.calibration_toggle_btn.setEnabled(False)
        layout.addWidget(self.calibration_toggle_btn, 5, 1)

        self.config_warning_label = QLabel("⚠ 配置不完整：请加载三维模型与参数文件以完成初始化")
        self.config_warning_label.setObjectName("config_warning_label")
        layout.addWidget(self.config_warning_label, 6, 0, 1, 3)
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

    # ── Calibration card (integrated inline) ─────────────────────────

    def _build_calibration_card(self) -> QWidget:
        card = CardFrame("calibration_card")
        card.setObjectName("calibration_card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 16, 22, 16)
        layout.setSpacing(10)

        title_row = QHBoxLayout()
        title_row.addWidget(self._section_title("📐 参数辨识工具"))
        title_row.addStretch(1)
        self._calib_status_label = QLabel("就绪")
        self._calib_status_label.setObjectName("calib_status_inline_label")
        title_row.addWidget(self._calib_status_label)
        layout.addLayout(title_row)

        # Data loading row
        data_row = QHBoxLayout()
        data_row.setSpacing(8)
        self._load_calib_btn = QPushButton("加载辨识数据 (.pkl)")
        self._load_calib_btn.setObjectName("load_calib_data_button")
        self._load_calib_btn.clicked.connect(self._choose_calib_data)
        data_row.addWidget(self._load_calib_btn)

        self._run_calib_btn = QPushButton("执行 S1 辨识")
        self._run_calib_btn.setObjectName("run_calibration_button")
        self._run_calib_btn.setEnabled(False)
        self._run_calib_btn.clicked.connect(self._run_calibration)
        data_row.addWidget(self._run_calib_btn)
        layout.addLayout(data_row)

        self._calib_data_info = QLabel("尚未加载辨识数据")
        self._calib_data_info.setObjectName("data_info_label")
        self._calib_data_info.setWordWrap(True)
        layout.addWidget(self._calib_data_info)

        # Results grid
        results = QFrame()
        results.setObjectName("metrics_frame")
        rg = QGridLayout(results)
        rg.setContentsMargins(0, 0, 0, 0)
        rg.setHorizontalSpacing(14)
        rg.setVerticalSpacing(6)

        result_labels = [
            ("定位误差 RMS:", "rmse_label", "-- mm"),
            ("最大定位误差:", "max_error_label", "-- mm"),
            ("辨识样本数:", "sample_count_label", "--"),
            ("置信度:", "confidence_label", "100%"),
            ("优化迭代:", "nfev_label", "--"),
            ("辨识状态:", "calib_status_label", "未执行"),
        ]
        for row_idx, (title, obj_name, default) in enumerate(result_labels):
            rg.addWidget(QLabel(title), row_idx, 0)
            value = QLabel(default)
            value.setObjectName(obj_name)
            value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            rg.addWidget(value, row_idx, 1)
        layout.addWidget(results)

        # Error parameter summary
        self._param_summary = QLabel("辨识参数将在执行后显示")
        self._param_summary.setObjectName("param_summary_label")
        self._param_summary.setWordWrap(True)
        layout.addWidget(self._param_summary)

        # Action buttons
        action_row = QHBoxLayout()
        self._save_calib_btn = QPushButton("💾 保存辨识结果 (YAML)")
        self._save_calib_btn.setObjectName("save_calib_button")
        self._save_calib_btn.setEnabled(False)
        self._save_calib_btn.clicked.connect(self._save_calibration)
        action_row.addWidget(self._save_calib_btn)

        self._report_calib_btn = QPushButton("📄 生成精度报告 (HTML)")
        self._report_calib_btn.setObjectName("generate_report_button")
        self._report_calib_btn.setEnabled(False)
        self._report_calib_btn.clicked.connect(self._generate_calibration_report)
        action_row.addWidget(self._report_calib_btn)
        layout.addLayout(action_row)

        layout.addStretch(1)
        return card

    # ── Guide card ────────────────────────────────────────────────────

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
        for text in ("机器人：UR10", "控制器：CB3", "仿真频率：60 Hz"):
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

    # ── Toggle between guide and calibration panels ──────────────────

    def _toggle_calibration_panel(self) -> None:
        self._right_bottom_stack.setCurrentIndex(1)
        self.calibration_toggle_btn.setText("← 返回指南")
        self.calibration_toggle_btn.clicked.disconnect()
        self.calibration_toggle_btn.clicked.connect(self._show_guide_panel)
        self.calibration_toggle_btn.setEnabled(True)
        self._set_status_message("已切换到参数辨识面板")

    def _show_guide_panel(self) -> None:
        self._right_bottom_stack.setCurrentIndex(0)
        self.calibration_toggle_btn.setText("▶ 进入标定分析")
        self.calibration_toggle_btn.clicked.disconnect()
        self.calibration_toggle_btn.clicked.connect(self._toggle_calibration_panel)
        self.calibration_toggle_btn.setEnabled(self.model_loaded)
        self._set_status_message("已返回初始化指南")

    # ── File loading ─────────────────────────────────────────────────

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

    # ── Realtime accuracy refresh ────────────────────────────────────

    def _refresh_realtime_accuracy(self) -> None:
        if not hasattr(self, "accuracy_value_labels") or not self.accuracy_value_labels:
            return
        threshold_mm = float(self.threshold_spin.value()) / 1000.0
        if not self.active_parameters_loaded:
            self.accuracy_value_labels[0].setText("--")
            self.accuracy_value_labels[1].setText("--")
            self.accuracy_value_labels[2].setText(f"{threshold_mm:.3f} mm")
            self.accuracy_value_labels[3].setText("未加载辨识参数")
            self._set_accuracy_alarm_dot("unknown")
            self.health_ring_label.setText("--\n未初始化")
            self.health_gauge.set_status(None, None)
            self._set_alarm_status("unknown")
            if self.health_value_labels:
                self.health_value_labels[0].setText("--")
                self.health_value_labels[1].setText("未加载辨识参数")
                self.health_value_labels[2].setText("--")
            return

        try:
            state = self._calibration_service.compute_predicted_position(
                self.robot_view.joint_degrees,
                joint_unit="degrees",
            )
        except Exception as exc:  # noqa: BLE001
            self.accuracy_value_labels[3].setText(f"计算失败：{exc}")
            self._set_accuracy_alarm_dot("critical")
            self._set_alarm_status("critical")
            return

        is_over_limit = state.rms_mm > threshold_mm
        self.accuracy_value_labels[0].setText(f"{state.rms_mm:.3f} mm")
        self.accuracy_value_labels[1].setText(f"{state.max_error_mm:.3f} mm")
        self.accuracy_value_labels[2].setText(f"{threshold_mm:.3f} mm")
        self.accuracy_value_labels[3].setText("超差" if is_over_limit else "正常")
        self._set_accuracy_alarm_dot("critical" if is_over_limit else "good")
        self.health_ring_label.setText(f"{state.health_score:.0f}\n{state.health_level}")
        self.health_gauge.set_status(state.health_score, state.health_level)
        self._set_alarm_status(state.health_level)
        if self.health_value_labels:
            self.health_value_labels[0].setText(f"{state.confidence:.0f}%")
            self.health_value_labels[1].setText(state.health_level)
            self.health_value_labels[2].setText("实时")

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

    def _set_alarm_status(self, level: str) -> None:
        """Sync footer alarm badge with health level from realtime accuracy."""
        level_lower = (level or "unknown").lower()
        alarm_text = {
            "good": "● 无告警",
            "warning": "⚠ 注意",
            "critical": "● 告警",
        }
        self.alarm_status_label.setText(alarm_text.get(level_lower, "● 未知"))
        color = HealthGaugeWidget.LEVEL_COLORS.get(
            level_lower,
            HealthGaugeWidget.LEVEL_COLORS["unknown"],
        )
        self.alarm_status_label.setStyleSheet(f"color: {color}; font-weight: 600;")

    # ── Calibration actions ──────────────────────────────────────────

    def _choose_calib_data(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "加载辨识数据",
            str(self.project_root / "data"),
            "Calibration data (*.pkl *.pickle);;All files (*)",
        )
        if paths:
            self._load_calib_data([Path(path) for path in paths])

    def _load_calib_data(self, paths: Path | list[Path]) -> None:
        try:
            path_list = [paths] if isinstance(paths, Path) else list(paths)
            data = self._calibration_service.load_identification_data(path_list)
            joints = np.asarray(data["joints"])
            positions = np.asarray(data["measured_positions"])
            self._calib_data = data
            self._calib_paths = path_list
            names = f"{len(path_list)} 个文件: " + ", ".join(path.name for path in path_list[:3])
            if len(path_list) > 3:
                names += f" ... 共 {len(path_list)} 个文件"
            self._calib_data_info.setText(
                f"已加载: {names}\n"
                f"样本数: {len(joints)} | 关节1范围: [{joints[:, 0].min():.3f}, {joints[:, 0].max():.3f}] rad | "
                f"位置范围: X[{positions[:, 0].min():.3f}, {positions[:, 0].max():.3f}]m"
            )
            self._run_calib_btn.setEnabled(True)
            self._calib_status_label.setText("● 数据已加载")
            self._set_status_message(f"辨识数据已加载: {len(path_list)} 个文件，{len(joints)} 个样本")
        except Exception as exc:
            self._calib_data_info.setText(f"加载失败: {exc}")
            self._set_status_message(f"数据加载失败: {exc}")

    def _run_calibration(self) -> None:
        if self._calib_data is None:
            self._set_status_message("请先加载辨识数据")
            return

        self._calib_status_label.setText("● 正在辨识...")
        self._run_calib_btn.setEnabled(False)
        self._identification_progress = QProgressDialog("S1 参数辨识正在运行，请稍候...", "", 0, 0, self)
        self._identification_progress.setObjectName("identification_progress_dialog")
        self._identification_progress.setWindowTitle("参数辨识进度")
        self._identification_progress.setCancelButton(None)
        self._identification_progress.setMinimumDuration(0)
        self._identification_progress.setModal(True)
        self._identification_progress.show()

        thread = QThread(self)
        worker = IdentificationWorker(self._calibration_service, self._calib_data, self._calib_paths)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_identification_finished)
        worker.failed.connect(self._on_identification_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(self._on_identification_thread_finished)
        self._identification_thread = thread
        self._identification_worker = worker
        thread.start()

    @Slot(object)
    def _on_identification_finished(self, result: CalibrationResult) -> None:
        if self._identification_progress is not None:
            self._identification_progress.close()
            self._identification_progress = None
        self._calib_result = result
        self._display_calibration_result(result)
        if result.success:
            saved = self._persist_identification_result(result)
            if saved is not None:
                self._ask_to_apply_identification_result(result, saved)
        self._run_calib_btn.setEnabled(True)

    @Slot(str)
    def _on_identification_failed(self, message: str) -> None:
        if self._identification_progress is not None:
            self._identification_progress.close()
            self._identification_progress = None
        self._calib_status_label.setText(f"● 辨识失败: {message}")
        self._set_status_message(f"辨识失败: {message}")
        self._run_calib_btn.setEnabled(True)

    @Slot()
    def _on_identification_thread_finished(self) -> None:
        self._identification_thread = None
        self._identification_worker = None

    def _display_calibration_result(self, result: CalibrationResult) -> None:
        self.findChild(QLabel, "rmse_label").setText(f"{result.position_error_rmse_mm:.4f} mm")
        self.findChild(QLabel, "max_error_label").setText(f"{result.position_error_max_mm:.4f} mm")
        self.findChild(QLabel, "sample_count_label").setText(str(result.joint_count))
        self.findChild(QLabel, "confidence_label").setText(f"{result.confidence:.0f}%")
        self.findChild(QLabel, "nfev_label").setText(str(result.nfev))

        if result.success:
            self.findChild(QLabel, "calib_status_label").setText("✔ S1 辨识成功")
            self._calib_status_label.setText("● 辨识完成")
        else:
            self.findChild(QLabel, "calib_status_label").setText(f"⚠ {result.message}")

        # Summarize top error parameters
        if result.parameter_values:
            significant = [
                (name, val)
                for name, val in result.parameter_values.items()
                if abs(val) > 1e-10
            ]
            significant.sort(key=lambda x: abs(x[1]), reverse=True)
            top_n = 8
            if significant:
                lines = [
                    "主要辨识参数:",
                    f"  S1 λ: {result.selected_lambda:.3g}",
                    f"  拟合残差 RMSE: {result.rmse_mm:.4f} mm",
                    "  定位误差定义: 预测模型位置 - 名义模型位置",
                ]
                for name, val in significant[:top_n]:
                    param = next(
                        (p for p in result.error_parameters if p.name == name), None
                    )
                    unit = param.unit if param else ""
                    lines.append(f"  {name}: {val:.6f} {unit}")
                if len(significant) > top_n:
                    lines.append(f"  ... 共 {len(significant)} 个非零参数")
                self._param_summary.setText("\n".join(lines))
            else:
                self._param_summary.setText("所有误差参数接近零，模型与测量数据高度一致。")

        self._save_calib_btn.setEnabled(True)
        self._report_calib_btn.setEnabled(True)
        self._set_status_message(
            f"S1 辨识完成: 定位误差RMSE={result.position_error_rmse_mm:.4f}mm, 拟合RMSE={result.rmse_mm:.4f}mm"
        )

    def _persist_identification_result(self, result: CalibrationResult) -> Path | None:
        try:
            yaml_path = self.project_root / "config" / "calibration_result.yaml"
            saved = save_identification_result(
                yaml_path,
                result.parameter_values,
                fit_rmse_mm=result.rmse_mm,
                fit_max_error_mm=result.max_error_mm,
                position_error_rmse_mm=result.position_error_rmse_mm,
                position_error_max_mm=result.position_error_max_mm,
                sample_count=result.joint_count,
                confidence=result.confidence,
                method=result.method,
                selected_lambda=result.selected_lambda,
                dataset_paths=result.dataset_paths,
                cv_scores=result.cv_scores,
                subspace_summary=result.subspace_summary,
                extra_metadata=result.metadata,
            )
            record_identification_history(
                self.project_root / "storage" / "records" / "identification_history.sqlite",
                result_yaml_path=saved,
                method=result.method,
                success=result.success,
                message=result.message,
                sample_count=result.joint_count,
                fit_rmse_mm=result.rmse_mm,
                fit_max_error_mm=result.max_error_mm,
                position_error_rmse_mm=result.position_error_rmse_mm,
                position_error_max_mm=result.position_error_max_mm,
                selected_lambda=result.selected_lambda,
                confidence=result.confidence,
                dataset_paths=result.dataset_paths,
            )
            self._set_status_message(f"S1 辨识结果已保存: {saved.name}，历史已入库")
            return saved
        except Exception as exc:  # noqa: BLE001
            self._set_status_message(f"辨识完成，但持久化失败: {exc}")
            return None

    def _ask_to_apply_identification_result(
        self,
        result: CalibrationResult,
        saved_path: Path,
    ) -> None:
        if not self._should_suggest_model_update(result):
            self._set_status_message("辨识结果已保存；当前指标未触发模型更新建议")
            return

        reply = QMessageBox.question(
            self,
            "应用辨识参数",
            (
                "辨识结果已持久化，当前指标建议更新精度模型。\n"
                f"是否将新参数文件加载到主界面参数文件栏？\n\n{saved_path}"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            self._set_status_message("辨识结果已保存，未加载为当前参数模型")
            return

        self.load_param_file(saved_path)
        self._set_status_message(f"已加载新参数文件: {saved_path.name}")

    def _should_suggest_model_update(self, result: CalibrationResult) -> bool:
        threshold_path = self.project_root / "config" / "thresholds.yaml"
        rms_limit = 0.5
        max_limit = 1.0
        try:
            data = yaml.safe_load(threshold_path.read_text(encoding="utf-8")) or {}
            accuracy = data.get("accuracy", {}) if isinstance(data, dict) else {}
            rms_limit = float(accuracy.get("position_rms_limit_mm", rms_limit))
            max_limit = float(accuracy.get("max_error_limit_mm", max_limit))
        except Exception:
            pass
        improved_fit = result.rmse_mm < result.nominal_to_measured_rmse_mm
        exceeds_limit = (
            result.position_error_rmse_mm > rms_limit
            or result.position_error_max_mm > max_limit
        )
        return bool(improved_fit or exceeds_limit)

    def _save_calibration(self) -> None:
        if self._calib_result is None:
            return
        default_dir = self.project_root / "data" / "calibration"
        default_dir.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存辨识结果",
            str(default_dir / "calibration_result.yaml"),
            "YAML files (*.yaml *.yml)",
        )
        if not path:
            return
        try:
            result = self._calib_result
            saved = save_identification_result(
                path,
                result.parameter_values,
                fit_rmse_mm=result.rmse_mm,
                fit_max_error_mm=result.max_error_mm,
                position_error_rmse_mm=result.position_error_rmse_mm,
                position_error_max_mm=result.position_error_max_mm,
                sample_count=result.joint_count,
                confidence=result.confidence,
                method=result.method,
                selected_lambda=result.selected_lambda,
                dataset_paths=result.dataset_paths,
                cv_scores=result.cv_scores,
                subspace_summary=result.subspace_summary,
                extra_metadata=result.metadata,
            )
            self._set_status_message(f"辨识结果已保存: {saved.name}")
        except Exception as exc:
            self._set_status_message(f"保存失败: {exc}")

    def _generate_calibration_report(self) -> None:
        if self._calib_result is None:
            return
        default_dir = self.project_root / "data" / "reports"
        default_dir.mkdir(parents=True, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "保存精度报告",
            str(default_dir / "calibration_report.html"),
            "HTML files (*.html)",
        )
        if not path:
            return
        try:
            html = self._build_calibration_report_html()
            Path(path).write_text(html, encoding="utf-8")
            self._set_status_message(f"精度报告已生成: {Path(path).name}")
            self._open_url(QUrl.fromLocalFile(str(Path(path).resolve())))
        except Exception as exc:
            self._set_status_message(f"报告生成失败: {exc}")

    def _build_calibration_report_html(self) -> str:
        result = self._calib_result
        if result is None:
            return "<html><body>No calibration result</body></html>"

        nominal = result.nominal_positions
        measured = result.measured_positions
        calibrated = result.calibrated_positions

        rows_html = ""
        for i in range(min(len(measured), 50)):
            pos_err = np.linalg.norm(calibrated[i] - nominal[i]) * 1000.0
            fit_err = np.linalg.norm(calibrated[i] - measured[i]) * 1000.0
            rows_html += (
                f"<tr><td>{i + 1}</td>"
                f"<td>{nominal[i, 0]:.4f}, {nominal[i, 1]:.4f}, {nominal[i, 2]:.4f}</td>"
                f"<td>{calibrated[i, 0]:.4f}, {calibrated[i, 1]:.4f}, {calibrated[i, 2]:.4f}</td>"
                f"<td>{measured[i, 0]:.4f}, {measured[i, 1]:.4f}, {measured[i, 2]:.4f}</td>"
                f"<td>{pos_err:.4f}</td><td>{fit_err:.4f}</td></tr>"
            )

        param_rows = ""
        significant = sorted(
            [(n, v) for n, v in result.parameter_values.items() if abs(v) > 1e-10],
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        for name, val in significant[:20]:
            param_rows += f"<tr><td>{name}</td><td>{val:.8f}</td></tr>"

        return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>机器人参数辨识精度报告</title>
<style>
body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 28px; color: #1f2933; line-height: 1.55; }}
h1, h2, h3 {{ color: #102a43; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }}
th, td {{ border: 1px solid #d9e2ec; padding: 7px 9px; text-align: left; }}
th {{ background: #f0f4f8; }}
.ok {{ color: #0b6b3a; font-weight: 700; }}
.warn {{ color: #9a6700; font-weight: 700; }}
.card {{ background: #f8fafc; border: 1px solid #d9e2ec; border-radius: 8px; padding: 14px 18px; margin: 12px 0; }}
</style>
</head>
<body>
<h1>机器人参数辨识精度报告</h1>

<div class="card">
<h2>概要</h2>
<table>
<tr><th>项目</th><th>结果</th></tr>
<tr><td>辨识状态</td><td class="{'ok' if result.success else 'warn'}">{'成功' if result.success else '未收敛'}</td></tr>
<tr><td>辨识方法</td><td>{result.method}</td></tr>
<tr><td>辨识样本数</td><td>{result.joint_count}</td></tr>
<tr><td>S1 选择 λ</td><td>{result.selected_lambda:.6g}</td></tr>
<tr><td>定位误差 RMSE（预测-名义）</td><td>{result.position_error_rmse_mm:.4f} mm</td></tr>
<tr><td>最大定位误差（预测-名义）</td><td>{result.position_error_max_mm:.4f} mm</td></tr>
<tr><td>拟合残差 RMSE（预测-测量）</td><td>{result.rmse_mm:.4f} mm</td></tr>
<tr><td>最大拟合残差（预测-测量）</td><td>{result.max_error_mm:.4f} mm</td></tr>
<tr><td>优化迭代次数</td><td>{result.nfev}</td></tr>
<tr><td>初始置信度</td><td>{result.confidence:.0f}%</td></tr>
</table>
</div>

<div class="card">
<h2>辨识误差参数（前20项）</h2>
<table>
<tr><th>参数名称</th><th>辨识值</th></tr>
{param_rows if param_rows else '<tr><td colspan="2">所有参数接近零</td></tr>'}
</table>
</div>

<div class="card">
<h2>逐点误差对比（前50点）</h2>
<table>
<tr><th>#</th><th>名义位置 (m)</th><th>预测位置 (m)</th><th>测量位置 (m)</th><th>定位误差 (mm)</th><th>拟合残差 (mm)</th></tr>
{rows_html}
</table>
</div>

<div class="card">
<h2>方法说明</h2>
<ul>
<li>运动学模型: 改进 D-H (Modified Denavit-Hartenberg) 六关节串联机器人</li>
<li>辨识参数: 33 几何参数 (24 MD-H + 6 基座 + 3 工具平移)</li>
<li>优化方法: S1 子空间可辨识性加权 + 交叉验证正则化 + Levenberg-Marquardt</li>
<li>拟合目标: min ||p(q; theta) - y_measured||^2 + 正则项</li>
<li>定位误差定义: p_identified(q) - p_nominal(q)</li>
</ul>
</div>

</body>
</html>"""

    # ── Status refresh ───────────────────────────────────────────────

    def _refresh_status(self) -> None:
        if self.model_loaded and self.params_loaded:
            self.config_status_label.setText("● 已加载")
            self.config_warning_label.setText("● 初始化完成：三维模型与参数文件已加载")
            self.simulation_status_label.setText("● 初始化完成")
            self.prompt_message_label.setText("模型与参数已加载，可进入精度分析流程。")
        elif self.model_loaded:
            self.config_status_label.setText("⚠ 参数未加载")
            self.config_warning_label.setText("⚠ 配置不完整：请继续加载误差参数文件")
            self.simulation_status_label.setText("⚠ 模型已加载，等待参数")
            self.prompt_message_label.setText("已找到机器人模型，请继续加载误差参数文件。")
        elif self.params_loaded:
            self.config_status_label.setText("⚠ 模型未加载")
            self.config_warning_label.setText("⚠ 配置不完整：请继续加载机器人三维模型")
            self.simulation_status_label.setText("⚠ 参数已加载，等待模型")
            self.prompt_message_label.setText("已找到误差参数，请继续加载机器人三维模型。")
        else:
            self.config_status_label.setText("⚠ 未加载")
            self.config_warning_label.setText("⚠ 配置不完整：请加载三维模型与参数文件以完成初始化")
            self.simulation_status_label.setText("⚠ 配置未加载，等待初始化")
            self.prompt_message_label.setText(
                "在当前配置目录中未找到可用的三维模型文件（.urdf/.xacro）\n"
                "或误差参数文件（calib_params.yaml）。\n"
                "请先加载所需文件以继续初始化。"
            )
        self.init_prompt_card.setVisible(not self.model_loaded)
        if hasattr(self, "calibration_toggle_btn"):
            self.calibration_toggle_btn.setEnabled(self.model_loaded)
        self._apply_status_colors()
        self._refresh_realtime_accuracy()

    def _apply_status_colors(self) -> None:
        normal = self.status_colors["normal"]
        warning = self.status_colors["warning"]
        self.connection_status_label.setStyleSheet(f"color: {normal}; font-weight: 600;")
        status_color = normal if self.model_loaded and self.params_loaded else warning
        self.config_status_label.setStyleSheet(f"color: {status_color}; font-weight: 600;")
        self.simulation_status_label.setStyleSheet(f"color: {status_color}; font-weight: 600;")
        self.config_warning_label.setStyleSheet(
            "border-radius: 6px; padding: 10px 12px; font-weight: 600;"
            f"color: {warning}; background: #fff7ed; border: 1px solid {warning};"
        )

    def _set_status_message(self, message: str) -> None:
        self.footer_status_label.setText(message)

    # ── Styles ───────────────────────────────────────────────────────

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
            QFrame#calibration_card,
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
            QLabel#calib_status_inline_label {
                color: #516075;
                font-weight: 600;
            }
            QFrame#metrics_frame {
                background: #f0f4f8;
                border: 1px solid #d9e2ec;
                border-radius: 6px;
                padding: 8px;
            }
            QLabel#data_info_label, QLabel#param_summary_label {
                color: #49566c;
                background: #f8fafc;
                border: 1px solid #e0e8f4;
                border-radius: 4px;
                padding: 8px;
            }
            QPushButton#load_calib_data_button, QPushButton#run_calibration_button {
                min-height: 40px;
                color: #0f62d9;
                border: 1px solid #3982ff;
                background: #ffffff;
            }
            QPushButton#save_calib_button, QPushButton#generate_report_button {
                min-height: 36px;
            }
            QLabel#rmse_label, QLabel#max_error_label {
                font-weight: 700;
            }
            QStackedWidget#right_bottom_stack {
                background: transparent;
            }
            """
        )
