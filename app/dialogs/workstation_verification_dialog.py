from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from core.workstation_verification import (
    JOINT_KEYS,
    POSE_KEYS,
    WorkstationAccuracyPreview,
    WorkstationConfig,
    WorkstationVerificationService,
)


class WorkstationAccuracyPreviewDialog(QDialog):
    """Standalone workstation preview result window."""

    preview_selected = Signal(object)

    def __init__(
        self,
        previews: list[WorkstationAccuracyPreview],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("workstation_accuracy_preview_dialog")
        self.setWindowTitle("加工工位精度预览")
        self.resize(980, 560)
        self._previews = list(previews)
        self._build_ui()
        self._apply_style()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(12)

        title = QLabel("加工工位精度预览")
        title.setObjectName("workstation_preview_title")
        layout.addWidget(title)

        self.preview_table = QTableWidget(0, 7)
        self.preview_table.setObjectName("workstation_preview_table")
        self.preview_table.setHorizontalHeaderLabels(
            ("编号", "输入方式", "配置", "预测误差(mm)", "阈值(mm)", "状态", "健康值")
        )
        self.preview_table.verticalHeader().setVisible(False)
        self.preview_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.preview_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.preview_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.preview_table.itemSelectionChanged.connect(self._emit_selected_preview)
        layout.addWidget(self.preview_table, stretch=1)

        alarm_title = QLabel("告警工位")
        alarm_title.setObjectName("workstation_alarm_title")
        layout.addWidget(alarm_title)

        self.alarm_table = QTableWidget(0, 4)
        self.alarm_table.setObjectName("workstation_alarm_table")
        self.alarm_table.setHorizontalHeaderLabels(("编号", "配置", "预测误差", "阈值"))
        self.alarm_table.verticalHeader().setVisible(False)
        self.alarm_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.alarm_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.alarm_table.setMaximumHeight(160)
        layout.addWidget(self.alarm_table)

        close_button = QPushButton("关闭")
        close_button.setObjectName("close_workstation_preview_button")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button, alignment=Qt.AlignmentFlag.AlignRight)

        self._populate_tables()

    def _populate_tables(self) -> None:
        self.preview_table.setRowCount(0)
        for row, preview in enumerate(self._previews):
            self.preview_table.insertRow(row)
            values = (
                preview.workstation_id,
                "关节角" if preview.workstation.input_type == "joint" else "工位位姿",
                _configuration_summary(preview),
                f"{preview.error_mm:.3f}",
                f"{preview.threshold_mm:.3f}",
                "超差" if preview.over_limit else "正常",
                f"{preview.health_score:.0f} / {preview.health_level}",
            )
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                if preview.over_limit:
                    item.setBackground(QColor("#ffe4e6"))
                elif col in {3, 5, 6}:
                    item.setBackground(QColor("#dcfce7"))
                self.preview_table.setItem(row, col, item)

        alarms = [preview for preview in self._previews if preview.over_limit]
        self.alarm_table.setRowCount(0)
        for row, preview in enumerate(alarms):
            self.alarm_table.insertRow(row)
            values = (
                preview.workstation_id,
                _configuration_summary(preview),
                f"{preview.error_mm:.3f} mm",
                f"{preview.threshold_mm:.3f} mm",
            )
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setBackground(QColor("#ffe4e6"))
                self.alarm_table.setItem(row, col, item)

        if self.preview_table.rowCount() > 0:
            self.preview_table.selectRow(0)

    def _emit_selected_preview(self) -> None:
        row = self.preview_table.currentRow()
        if 0 <= row < len(self._previews):
            self.preview_selected.emit(self._previews[row])

    def _apply_style(self) -> None:
        self.setStyleSheet(_dialog_style())


class WorkstationVerificationDialog(QDialog):
    """Editor window for machining workstation configuration."""

    current_preview_changed = Signal(object)

    def __init__(
        self,
        project_root: str | Path,
        *,
        verification_service: WorkstationVerificationService | None = None,
        threshold_mm_getter: Callable[[], float] | None = None,
        current_joint_degrees_getter: Callable[[], list[float]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("workstation_verification_dialog")
        self.setWindowTitle("加工工位配置")
        self.resize(760, 560)

        self.project_root = Path(project_root).resolve()
        self._service = verification_service or WorkstationVerificationService(self.project_root)
        self._threshold_mm_getter = threshold_mm_getter or (lambda: 0.5)
        self._current_joint_degrees_getter = current_joint_degrees_getter or (lambda: [0.0] * 6)
        self._workstations: list[WorkstationConfig] = []
        self._current_index = -1
        self._loading_form = False
        self.preview_dialog: WorkstationAccuracyPreviewDialog | None = None

        self._build_ui()
        self._load_workstations()
        self._apply_style()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(12)

        title = QLabel("加工工位配置")
        title.setObjectName("workstation_title")
        layout.addWidget(title)

        guide = QLabel("通过下拉框选择工位，选择输入方式后编辑对应参数；精度结果会在独立预览窗口中显示。")
        guide.setObjectName("workstation_guide")
        guide.setWordWrap(True)
        layout.addWidget(guide)

        selector_row = QHBoxLayout()
        selector_row.setSpacing(8)
        selector_row.addWidget(QLabel("当前工位"))
        self.workstation_selector_combo = QComboBox()
        self.workstation_selector_combo.setObjectName("workstation_selector_combo")
        self.workstation_selector_combo.currentIndexChanged.connect(self._select_workstation)
        selector_row.addWidget(self.workstation_selector_combo, stretch=1)

        self.add_button = QPushButton("新增工位")
        self.add_button.setObjectName("add_workstation_button")
        self.add_button.clicked.connect(self._add_default_workstation)
        selector_row.addWidget(self.add_button)

        self.remove_button = QPushButton("删除工位")
        self.remove_button.setObjectName("remove_workstation_button")
        self.remove_button.clicked.connect(self._remove_current_workstation)
        selector_row.addWidget(self.remove_button)
        layout.addLayout(selector_row)

        form = QGridLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)
        form.addWidget(QLabel("工位编号"), 0, 0)
        self.workstation_id_edit = QLineEdit()
        self.workstation_id_edit.setObjectName("workstation_id_edit")
        form.addWidget(self.workstation_id_edit, 0, 1)

        form.addWidget(QLabel("输入方式"), 1, 0)
        self.input_mode_combo = QComboBox()
        self.input_mode_combo.setObjectName("workstation_input_mode_combo")
        self.input_mode_combo.addItem("工位位姿", "pose")
        self.input_mode_combo.addItem("关节角", "joint")
        self.input_mode_combo.currentIndexChanged.connect(self._sync_input_page)
        form.addWidget(self.input_mode_combo, 1, 1)
        layout.addLayout(form)

        self.input_stack = QStackedWidget()
        self.input_stack.setObjectName("workstation_input_stack")
        self.input_stack.addWidget(self._build_pose_page())
        self.input_stack.addWidget(self._build_joint_page())
        layout.addWidget(self.input_stack, stretch=1)

        action_row = QHBoxLayout()
        action_row.addStretch(1)
        self.save_button = QPushButton("保存配置")
        self.save_button.setObjectName("save_workstation_config_button")
        self.save_button.clicked.connect(self.save_configuration)
        action_row.addWidget(self.save_button)

        self.preview_button = QPushButton("精度预览")
        self.preview_button.setObjectName("preview_workstation_accuracy_button")
        self.preview_button.clicked.connect(self.preview_accuracy)
        action_row.addWidget(self.preview_button)

        close_button = QPushButton("关闭")
        close_button.setObjectName("close_workstation_dialog_button")
        close_button.clicked.connect(self.close)
        action_row.addWidget(close_button)
        layout.addLayout(action_row)

        self.status_label = QLabel("配置文件：config/workstations.yaml")
        self.status_label.setObjectName("workstation_status_label")
        layout.addWidget(self.status_label)

    def _build_pose_page(self) -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(10)
        self.pose_spins: list[QDoubleSpinBox] = []
        for index, key in enumerate(POSE_KEYS):
            row = index // 3
            col = (index % 3) * 2
            layout.addWidget(QLabel(key), row, col)
            spin = self._make_spin(0.0, decimals=6, step=0.001, suffix="")
            spin.setObjectName(f"workstation_pose_{key}_spin")
            self.pose_spins.append(spin)
            layout.addWidget(spin, row, col + 1)
        return page

    def _build_joint_page(self) -> QWidget:
        page = QWidget()
        layout = QGridLayout(page)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(10)
        self.joint_spins: list[QDoubleSpinBox] = []
        for index, key in enumerate(JOINT_KEYS):
            row = index // 3
            col = (index % 3) * 2
            layout.addWidget(QLabel(key), row, col)
            spin = self._make_spin(0.0, decimals=3, step=1.0, suffix="°")
            spin.setRange(-360.0, 360.0)
            spin.setObjectName(f"workstation_joint_{key}_spin")
            self.joint_spins.append(spin)
            layout.addWidget(spin, row, col + 1)
        return page

    def _make_spin(
        self,
        value: float,
        *,
        decimals: int,
        step: float,
        suffix: str,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(-10000.0, 10000.0)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        if suffix:
            spin.setSuffix(suffix)
        return spin

    def _load_workstations(self) -> None:
        try:
            self._workstations = self._service.load_workstations()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "读取工位配置失败", str(exc))
            self._workstations = []
        if not self._workstations:
            self._workstations = [self._make_default_workstation("WS-001")]
        self._refresh_selector(0)

    def _make_default_workstation(self, workstation_id: str) -> WorkstationConfig:
        return WorkstationConfig(
            workstation_id,
            "joint",
            joint_degrees=self._current_joint_degrees(),
        )

    def _current_joint_degrees(self) -> tuple[float, float, float, float, float, float]:
        values = [float(value) for value in self._current_joint_degrees_getter()[:6]]
        values += [0.0] * max(0, 6 - len(values))
        return tuple(values[:6])  # type: ignore[return-value]

    def _refresh_selector(self, index: int) -> None:
        self._loading_form = True
        self.workstation_selector_combo.blockSignals(True)
        self.workstation_selector_combo.clear()
        for workstation in self._workstations:
            self.workstation_selector_combo.addItem(workstation.workstation_id)
        self.workstation_selector_combo.setCurrentIndex(max(0, min(index, len(self._workstations) - 1)))
        self.workstation_selector_combo.blockSignals(False)
        self._loading_form = False
        self._load_form(self.workstation_selector_combo.currentIndex())

    def _select_workstation(self, index: int) -> None:
        if self._loading_form:
            return
        self._store_current_form()
        self._load_form(index)

    def _load_form(self, index: int) -> None:
        if not (0 <= index < len(self._workstations)):
            return
        self._current_index = index
        workstation = self._workstations[index]
        pose = workstation.pose or (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        joints = workstation.joint_degrees or self._current_joint_degrees()

        self._loading_form = True
        self.workstation_id_edit.setText(workstation.workstation_id)
        self.input_mode_combo.setCurrentIndex(1 if workstation.input_type == "joint" else 0)
        for spin, value in zip(self.pose_spins, pose, strict=True):
            spin.setValue(float(value))
        for spin, value in zip(self.joint_spins, joints, strict=True):
            spin.setValue(float(value))
        self._loading_form = False
        self._sync_input_page()

    def _store_current_form(self) -> None:
        if self._loading_form or not (0 <= self._current_index < len(self._workstations)):
            return
        workstation_id = self.workstation_id_edit.text().strip() or f"WS-{self._current_index + 1:03d}"
        input_type = str(self.input_mode_combo.currentData() or "pose")
        pose = tuple(float(spin.value()) for spin in self.pose_spins)
        joints = tuple(float(spin.value()) for spin in self.joint_spins)
        self._workstations[self._current_index] = WorkstationConfig(
            workstation_id,
            input_type,
            pose=pose,
            joint_degrees=joints,
        )
        self.workstation_selector_combo.setItemText(self._current_index, workstation_id)

    def _sync_input_page(self) -> None:
        index = 1 if self.input_mode_combo.currentData() == "joint" else 0
        self.input_stack.setCurrentIndex(index)

    def _add_default_workstation(self) -> None:
        self._store_current_form()
        workstation_id = f"WS-{len(self._workstations) + 1:03d}"
        self._workstations.append(self._make_default_workstation(workstation_id))
        self._refresh_selector(len(self._workstations) - 1)

    def _remove_current_workstation(self) -> None:
        if not self._workstations:
            return
        index = max(0, self._current_index)
        self._workstations.pop(index)
        if not self._workstations:
            self._workstations.append(self._make_default_workstation("WS-001"))
        self._refresh_selector(min(index, len(self._workstations) - 1))

    def save_configuration(self) -> None:
        self._store_current_form()
        try:
            saved = self._service.save_workstations(self._workstations)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "保存失败", str(exc))
            return
        self.status_label.setText(f"配置已保存：{saved}")

    def preview_accuracy(self) -> None:
        self._store_current_form()
        try:
            self._service.save_workstations(self._workstations)
            previews = self._service.preview_workstations(
                self._workstations,
                threshold_mm=self._threshold_mm_getter(),
                initial_joint_degrees=self._current_joint_degrees(),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "精度预览失败", str(exc))
            return

        self.preview_dialog = WorkstationAccuracyPreviewDialog(previews, parent=self)
        self.preview_dialog.setModal(False)
        self.preview_dialog.preview_selected.connect(self.current_preview_changed)
        self.preview_dialog.show()
        self.preview_dialog.raise_()
        if previews:
            selected = min(max(self._current_index, 0), len(previews) - 1)
            self.preview_dialog.preview_table.selectRow(selected)
            self.current_preview_changed.emit(previews[selected])
        alarm_count = sum(1 for preview in previews if preview.over_limit)
        self.status_label.setText(
            f"精度预览完成：{len(previews)} 个工位，{alarm_count} 个告警"
        )

    def _apply_style(self) -> None:
        self.setStyleSheet(_dialog_style())


def _configuration_summary(preview: WorkstationAccuracyPreview) -> str:
    if preview.workstation.input_type == "joint":
        values = ", ".join(
            f"{key}={value:.2f}°"
            for key, value in zip(JOINT_KEYS, preview.joint_degrees, strict=True)
        )
        return f"关节角: {values}"
    values = ", ".join(
        f"{key}={value:.4f}"
        for key, value in zip(POSE_KEYS, preview.pose, strict=True)
    )
    return f"位姿: {values}"


def _dialog_style() -> str:
    return """
        QDialog {
            background: #eef4fb;
            color: #172033;
            font-family: "Microsoft YaHei", "Segoe UI", Arial;
            font-size: 13px;
        }
        QLabel#workstation_title,
        QLabel#workstation_preview_title {
            font-size: 18px;
            font-weight: 700;
            color: #172033;
        }
        QLabel#workstation_guide,
        QLabel#workstation_status_label {
            color: #516075;
        }
        QLabel#workstation_alarm_title {
            font-size: 15px;
            font-weight: 700;
        }
        QTableWidget {
            background: #f8fbff;
            alternate-background-color: #f1f6fd;
            border: 1px solid #d5e0ef;
            gridline-color: #d9e4f2;
            selection-background-color: #dbeafe;
            selection-color: #172033;
        }
        QHeaderView::section {
            background: #eaf2fb;
            border: 0;
            border-right: 1px solid #d5e0ef;
            border-bottom: 1px solid #d5e0ef;
            padding: 6px 8px;
            font-weight: 600;
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
        QPushButton#preview_workstation_accuracy_button {
            color: #ffffff;
            background: #0f62d9;
            border-color: #0953c7;
        }
        QLineEdit,
        QDoubleSpinBox,
        QComboBox {
            min-height: 30px;
            border: 1px solid #d4dfed;
            border-radius: 5px;
            background: #ffffff;
            padding: 2px 6px;
        }
    """

