from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QStackedWidget,
    QStyle,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from core.nominal_parameter_service import NominalParameterService


class NominalParameterUpdateDialog(QDialog):
    """Modal editor for nominal robot parameter updates."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        parameter_service: NominalParameterService | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("nominal_parameter_update_dialog")
        self.setWindowTitle("更新名义参数")
        self.resize(760, 620)

        self.project_root = Path(project_root).resolve()
        self._service = parameter_service or NominalParameterService(self.project_root)
        self.result_path: Path | None = None
        self.result_mode: str | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(12)

        title = QLabel("更新名义参数")
        title.setObjectName("nominal_update_title")
        layout.addWidget(title)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(10)
        self.mode_combo = QComboBox()
        self.mode_combo.setObjectName("nominal_update_mode_combo")
        self.mode_combo.addItem("填写完整 YAML", "direct")
        self.mode_combo.addItem("填写参数值", "values")
        self.mode_combo.addItem("导入辨识结果 YAML", "identification")
        self.mode_combo.currentIndexChanged.connect(self._sync_mode_page)
        form.addRow("更新方式", self.mode_combo)
        layout.addLayout(form)

        self.stack = QStackedWidget()
        self.stack.setObjectName("nominal_update_stack")
        self.stack.addWidget(self._build_direct_page())
        self.stack.addWidget(self._build_values_page())
        self.stack.addWidget(self._build_import_page())
        layout.addWidget(self.stack, stretch=1)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.setObjectName("nominal_update_button_box")
        save_button = self.button_box.button(QDialogButtonBox.StandardButton.Save)
        cancel_button = self.button_box.button(QDialogButtonBox.StandardButton.Cancel)
        if save_button is not None:
            save_button.setText("保存更新")
            save_button.setObjectName("nominal_update_save_button")
        if cancel_button is not None:
            cancel_button.setText("取消")
            cancel_button.setObjectName("nominal_update_cancel_button")

        rollback_button = QPushButton("回退上一版本")
        rollback_button.setObjectName("nominal_rollback_button")
        rollback_button.setEnabled(self._service.has_backup())
        rollback_button.clicked.connect(self._rollback)
        self.button_box.addButton(rollback_button, QDialogButtonBox.ButtonRole.ActionRole)
        self.button_box.accepted.connect(self._apply_update)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

        self._sync_mode_page()

    def _build_direct_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.direct_yaml_edit = QPlainTextEdit()
        self.direct_yaml_edit.setObjectName("nominal_direct_yaml_edit")
        self.direct_yaml_edit.setPlainText(self._service.dump_current_yaml())
        layout.addWidget(self.direct_yaml_edit)
        return page

    def _build_values_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        self.values_yaml_edit = QPlainTextEdit()
        self.values_yaml_edit.setObjectName("nominal_values_yaml_edit")
        self.values_yaml_edit.setPlainText(self._service.value_template_yaml())
        layout.addWidget(self.values_yaml_edit)
        return page

    def _build_import_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self.identification_path_edit = QLineEdit()
        self.identification_path_edit.setObjectName("nominal_identification_path_edit")
        self.identification_path_edit.setPlaceholderText("选择 calibration_result.yaml")
        browse_button = QToolButton()
        browse_button.setObjectName("nominal_identification_browse_button")
        browse_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        browse_button.clicked.connect(self._browse_identification_file)

        row.addWidget(self.identification_path_edit, stretch=1)
        row.addWidget(browse_button)
        layout.addLayout(row)
        layout.addStretch(1)
        return page

    def _sync_mode_page(self) -> None:
        self.stack.setCurrentIndex(max(0, self.mode_combo.currentIndex()))

    def _browse_identification_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "导入辨识结果 YAML",
            str(self.project_root / "config"),
            "YAML file (*.yaml *.yml)",
        )
        if path:
            self.identification_path_edit.setText(path)

    def _apply_update(self) -> None:
        try:
            mode = self.mode_combo.currentData()
            if mode == "direct":
                result = self._service.update_direct_yaml(self.direct_yaml_edit.toPlainText())
            elif mode == "values":
                result = self._service.update_values_yaml(self.values_yaml_edit.toPlainText())
            else:
                result = self._service.update_from_identification_file(
                    self.identification_path_edit.text().strip()
                )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "更新失败", str(exc))
            return

        self.result_path = result.nominal_path
        self.result_mode = result.mode
        self.accept()

    def _rollback(self) -> None:
        try:
            self.result_path = self._service.rollback()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "回退失败", str(exc))
            return
        self.result_mode = "rollback"
        self.accept()
