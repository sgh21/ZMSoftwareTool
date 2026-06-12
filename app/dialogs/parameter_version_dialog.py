from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from core.parameter_repository import PARAMETER_KINDS, ParameterFileRepository


KIND_LABELS = {
    "controller_model": "控制模型",
    "identified_model": "辨识模型",
    "camera_monitoring": "相机监控参数",
}


class ParameterVersionDialog(QDialog):
    """Choose the active timestamped parameter-version combination."""

    def __init__(
        self,
        project_root: str | Path,
        *,
        repository: ParameterFileRepository | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.project_root = Path(project_root).resolve()
        self.repository = repository or ParameterFileRepository(self.project_root)
        self.selected_paths: dict[str, Path] = {}
        self._combos: dict[str, QComboBox] = {}
        self.setObjectName("parameter_version_dialog")
        self.setWindowTitle("选择参数版本组合")
        self.resize(680, 360)
        self._build_ui()
        self.refresh_versions()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        title = QLabel("选择参数版本组合")
        title.setObjectName("parameter_version_title")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(1, 1)
        for row, kind in enumerate(PARAMETER_KINDS):
            grid.addWidget(QLabel(KIND_LABELS[kind]), row, 0)
            combo = QComboBox()
            combo.setObjectName(f"{kind}_version_combo")
            combo.setMinimumWidth(420)
            self._combos[kind] = combo
            grid.addWidget(combo, row, 1)

            import_button = QPushButton("导入")
            import_button.setObjectName(f"import_{kind}_button")
            import_button.clicked.connect(lambda _checked=False, k=kind: self._import_version(k))
            grid.addWidget(import_button, row, 2)
        layout.addLayout(grid)

        self.status_label = QLabel("")
        self.status_label.setObjectName("parameter_version_status_label")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        action_row = QHBoxLayout()
        self.select_latest_button = QPushButton("选择最新版本")
        self.select_latest_button.setObjectName("select_latest_parameter_versions_button")
        self.select_latest_button.clicked.connect(self.select_latest_versions)
        action_row.addWidget(self.select_latest_button)
        action_row.addStretch(1)

        cancel_button = QPushButton("取消")
        cancel_button.setObjectName("cancel_parameter_versions_button")
        cancel_button.clicked.connect(self.reject)
        action_row.addWidget(cancel_button)

        apply_button = QPushButton("保存组合")
        apply_button.setObjectName("save_parameter_versions_button")
        apply_button.clicked.connect(self.apply_selection)
        action_row.addWidget(apply_button)
        layout.addLayout(action_row)

    def refresh_versions(self) -> None:
        self.repository.ensure_initial_versions()
        active = self.repository.load_active()
        for kind, combo in self._combos.items():
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("未选择", None)
            versions = self.repository.list_versions(kind)
            for version in versions:
                combo.addItem(_version_label(version.path), str(version.path))
            active_path = active.get(kind)
            if active_path:
                resolved = self.repository.resolve_path(active_path)
                index = combo.findData(str(resolved))
                if index >= 0:
                    combo.setCurrentIndex(index)
            combo.blockSignals(False)

    def select_latest_versions(self) -> None:
        latest = self.repository.select_latest_versions()
        for kind, path in latest.items():
            combo = self._combos[kind]
            index = combo.findData(str(path))
            if index >= 0:
                combo.setCurrentIndex(index)
        self.status_label.setText("已选中各类参数的最新版本")

    def apply_selection(self) -> None:
        active = self.repository.load_active()
        active.setdefault("nominal_robot", "config/nominal_robot.yaml")
        self.selected_paths = {}
        for kind, combo in self._combos.items():
            raw = combo.currentData()
            if raw:
                path = Path(str(raw)).resolve()
                self.repository.load_document(path, expected_kind=kind)
                active[kind] = str(path)
                self.selected_paths[kind] = path
            else:
                active.pop(kind, None)
        self.repository.save_active(active)
        self.accept()

    def _import_version(self, kind: str) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            f"导入{KIND_LABELS[kind]}",
            str(self.project_root / "config"),
            "Parameter files (*.yaml *.yml *.json)",
        )
        if not path:
            return
        try:
            saved = self.repository.import_parameter_file(kind, Path(path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "导入失败", str(exc))
            return
        self.refresh_versions()
        combo = self._combos[kind]
        index = combo.findData(str(saved))
        if index >= 0:
            combo.setCurrentIndex(index)
        self.status_label.setText(f"已导入{KIND_LABELS[kind]}：{saved.name}")


def _version_label(path: Path) -> str:
    return path.name
