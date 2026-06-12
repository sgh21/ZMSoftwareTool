from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import yaml
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QComboBox, QLineEdit, QPlainTextEdit, QPushButton

from app.dialogs.nominal_parameter_dialog import NominalParameterUpdateDialog
from core.calibration_persistence import save_identification_result
from core.nominal_parameter_service import nominal_after_applying_error_parameters


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


def write_nominal(root: Path) -> None:
    path = root / "config" / "nominal_robot.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "nominal_robot": {
                    "base_xyz": [1.0, 2.0, 3.0],
                    "base_rpy": [0.1, 0.2, 0.3],
                    "tool_xyz": [0.0, 0.0, 0.039],
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


def read_nominal(root: Path) -> dict:
    data = yaml.safe_load((root / "config" / "nominal_robot.yaml").read_text(encoding="utf-8"))
    return data["nominal_robot"]


def identified_payload(root: Path, errors: dict[str, float]) -> dict:
    return nominal_after_applying_error_parameters(read_nominal(root), errors)


def test_nominal_update_dialog_values_and_rollback(qapp: QApplication, tmp_path: Path) -> None:
    write_nominal(tmp_path)
    dialog = NominalParameterUpdateDialog(tmp_path)

    child(dialog, QComboBox, "nominal_update_mode_combo").setCurrentIndex(1)
    child(dialog, QPlainTextEdit, "nominal_values_yaml_edit").setPlainText(
        """
nominal_values:
  base_xyz: [0.5, 0, 0]
  mdh:
    a: [0, -0.5, -0.5723, 0, 0, 0]
"""
    )
    QTest.mouseClick(child(dialog, QPushButton, "nominal_update_save_button"), Qt.LeftButton)

    saved = read_nominal(tmp_path)
    assert "base_xyz" not in saved
    assert saved["mdh"]["a"][1] == -0.5

    rollback_dialog = NominalParameterUpdateDialog(tmp_path)
    rollback_button = child(rollback_dialog, QPushButton, "nominal_rollback_button")
    assert rollback_button.isEnabled()
    QTest.mouseClick(rollback_button, Qt.LeftButton)

    restored = read_nominal(tmp_path)
    assert "base_xyz" not in restored
    assert restored["mdh"]["a"][1] == -0.612


def test_nominal_update_dialog_imports_identification_yaml(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    write_nominal(tmp_path)
    result_yaml = tmp_path / "config" / "calibration_result.yaml"
    errors = {"delta_a_2": 0.02, "delta_Btx": 0.25, "delta_Ttz": 0.001}
    save_identification_result(
        result_yaml,
        errors,
        nominal_robot=read_nominal(tmp_path),
        identified_robot=identified_payload(tmp_path, errors),
        fit_rmse_mm=0.1,
        fit_max_error_mm=0.2,
        position_error_rmse_mm=1.0,
        position_error_max_mm=1.5,
        sample_count=10,
    )

    dialog = NominalParameterUpdateDialog(tmp_path)
    child(dialog, QComboBox, "nominal_update_mode_combo").setCurrentIndex(2)
    child(dialog, QLineEdit, "nominal_identification_path_edit").setText(str(result_yaml))
    QTest.mouseClick(child(dialog, QPushButton, "nominal_update_save_button"), Qt.LeftButton)

    saved = read_nominal(tmp_path)
    assert "base_xyz" not in saved
    assert saved["tool_xyz"][2] == pytest.approx(0.039)
    assert saved["mdh"]["a"][1] == pytest.approx(-0.592)
