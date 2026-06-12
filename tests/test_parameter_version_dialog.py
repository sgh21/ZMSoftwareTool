from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication, QComboBox

from app.dialogs.parameter_version_dialog import ParameterVersionDialog
from core.parameter_repository import ParameterFileRepository


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


def test_parameter_version_dialog_selects_latest_versions(
    qapp: QApplication,
    tmp_path: Path,
) -> None:
    repo = ParameterFileRepository(tmp_path)
    repo.create_version(
        "controller_model",
        {"controller_model": {"mdh": {}}},
        timestamp="20260612_120000",
    )
    latest_controller = repo.create_version(
        "controller_model",
        {"controller_model": {"mdh": {}}},
        timestamp="20260612_120001",
    )
    latest_identified = repo.create_version(
        "identified_model",
        {"identification": {"confidence_current": 90.0}},
        timestamp="20260612_120001",
    )
    latest_camera = repo.create_version(
        "camera_monitoring",
        {"model_monitoring": {"camera": {}}},
        timestamp="20260612_120001",
    )

    dialog = ParameterVersionDialog(tmp_path, repository=repo)
    dialog.select_latest_versions()

    assert child(dialog, QComboBox, "controller_model_version_combo").currentData() == str(
        latest_controller
    )
    assert child(dialog, QComboBox, "identified_model_version_combo").currentData() == str(
        latest_identified
    )
    assert child(dialog, QComboBox, "camera_monitoring_version_combo").currentData() == str(
        latest_camera
    )
