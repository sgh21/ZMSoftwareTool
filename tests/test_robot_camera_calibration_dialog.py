"""独立标定窗口回归；使用假后台，不运行真实图像标定。"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QApplication

import app.dialogs.robot_camera_calibration_dialog as dialog_module
from app.dialogs.robot_camera_calibration_dialog import RobotCameraCalibrationDialog
from app.dialogs.robot_position_simulation_dialog import RobotPositionSimulationDialog
from ui_helpers import ready_position_page
from core.services.position_monitoring_service import PositionMonitoringService, write_document


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def dialog(application, tmp_path):
    page = ready_position_page(PositionMonitoringService(root=tmp_path / "application"))
    page.ui_scale = 1.0
    window = RobotCameraCalibrationDialog(page)
    yield window
    window.task = None
    window.close()
    page.close()


def load_dataset(dialog, root, *, missing_record=None):
    runs = []
    for batch in ("B001", "B000", "B002"):
        runs.append({"batch_id": batch, "record": f"{batch}/record.json"})
        if batch != missing_record:
            write_document(root / batch / "record.json", {})
    write_document(root / "parameters.json", {"runs": runs})
    dialog.data_root.setText(str(root))
    dialog._load_dataset()
    return root


def result(output):
    return {
        "report": {
            "batch_id": "B000", "sample_count": 30,
            "camera_calibration": {
                "rms_px": 0.03, "camera_matrix": [[5000, 0, 1536], [0, 5000, 1024], [0, 0, 1]],
                "dist_coeffs": [0, 0, 0, 0, 0],
            },
            "hand_eye_calibration": {"translation_rms_mm": 0.04, "rotation_rms_deg": 0.01},
            "warnings": ["合成回归报告，不是真实测量。"],
        },
        "parameters_path": str(output / "parameters.json"),
        "hand_eye_path": str(output / "hand_eye.json"),
        "report_path": str(output / "calibration_report.json"),
    }


def test_defaults_source_changes_and_missing_records(dialog, tmp_path):
    assert not dialog.run_button.isEnabled()
    dialog.data_root.setText(str(tmp_path / "missing"))
    dialog._load_dataset()
    assert "parameters.json" in dialog.source_status.text()
    assert not dialog.run_button.isEnabled()
    load_dataset(dialog, tmp_path / "data", missing_record="B002")
    assert dialog.batch.currentData() == "B000"
    assert dialog.intrinsic_mode.currentData() == "calibrate"
    assert dialog.distortion_mode.currentData() == "estimate"
    assert dialog.hand_eye_mode.currentData() == "calibrate"
    assert dialog.run_button.isEnabled()
    dialog.batch.setCurrentIndex(dialog.batch.findData("B002"))
    assert "缺少批次记录" in dialog.source_status.text()
    assert not dialog.run_button.isEnabled()
    dialog.batch.setCurrentIndex(dialog.batch.findData("B001"))
    assert dialog.run_button.isEnabled()
    dialog.data_root.setText(str(tmp_path / "other"))
    assert dialog.dataset is None
    assert dialog.batch.count() == 0
    assert not dialog.run_button.isEnabled()


def test_reference_modes_disable_unused_options_and_clear_stale_results(dialog, tmp_path):
    load_dataset(dialog, tmp_path / "data")
    dialog._completed(result(tmp_path / "output"))
    assert dialog.report is not None
    dialog.intrinsic_mode.setCurrentIndex(dialog.intrinsic_mode.findData("reference"))
    assert not dialog.distortion_mode.isEnabled()
    assert dialog.report is None
    assert dialog.output.toPlainText() == ""
    assert dialog.report_path.text() == "输出文件：—"
    dialog.hand_eye_mode.setCurrentIndex(dialog.hand_eye_mode.findData("reference"))
    assert not dialog.method.isEnabled()
    assert dialog.run_button.isEnabled()
    dialog.hand_eye_mode.setCurrentIndex(dialog.hand_eye_mode.findData("calibrate"))
    assert dialog.method.isEnabled()


def test_task_snapshots_inputs_and_keeps_main_state_unchanged(dialog, tmp_path, monkeypatch):
    source = load_dataset(dialog, tmp_path / "data")
    dialog.distortion_mode.setCurrentIndex(dialog.distortion_mode.findData("zero"))
    dialog.hand_eye_mode.setCurrentIndex(dialog.hand_eye_mode.findData("reference"))
    queued = []
    monkeypatch.setattr(dialog_module.QThreadPool, "globalInstance", lambda: SimpleNamespace(start=queued.append))
    captured = {}

    def calculate(dataset, output, **options):
        captured.update({"dataset": dataset, "output": output, **options})
        options["progress"](30, "已检测角点")
        return result(output)

    monkeypatch.setattr(dialog_module, "calibrate_dataset", calculate)
    service = dialog.page.service
    original_parameters = deepcopy(service.parameters)
    original_files = {path: path.read_bytes() for path in service.storage.rglob("*.json")}
    dialog.page.result = {"id": "preserved"}
    dialog._run()
    assert len(queued) == 1
    assert dialog.task is queued[0]
    for widget in (dialog.data_root, dialog.browse_button, dialog.batch, dialog.intrinsic_mode,
                   dialog.distortion_mode, dialog.hand_eye_mode, dialog.method, dialog.close_button,
                   dialog.run_button):
        assert not widget.isEnabled()
    event = QCloseEvent()
    dialog.closeEvent(event)
    assert not event.isAccepted()
    dialog.reject()
    assert "等待完成" in dialog.output.toPlainText()
    dialog._run()
    assert len(queued) == 1
    queued[0].run()
    assert dialog.task is None
    assert dialog.run_button.isEnabled()
    assert dialog.close_button.isEnabled()
    assert not dialog.method.isEnabled()
    assert captured["dataset"] == source.resolve()
    assert captured["batch_id"] == "B000"
    assert captured["intrinsic_mode"] == "calibrate"
    assert captured["distortion_mode"] == "zero"
    assert captured["hand_eye_mode"] == "reference"
    assert captured["output"].parent == service.root / "data" / "processed" / "robot_camera_calibration"
    assert "30 张图像" in dialog.summary.text()
    assert "0.030000 px" in dialog.summary.text()
    assert "一致性残差" in dialog.output.toPlainText()
    assert "合成回归报告" in dialog.output.toPlainText()
    assert "parameters.json" in dialog.report_path.text()
    assert "hand_eye.json" in dialog.report_path.text()
    assert "calibration_report.json" in dialog.report_path.text()
    assert dialog.progress.value() == 100
    assert service.parameters == original_parameters
    assert service.baseline is None
    assert service.current_batch is None
    assert dialog.page.result == {"id": "preserved"}
    assert {path: path.read_bytes() for path in service.storage.rglob("*.json")} == original_files


def test_failure_restores_controls_and_leaves_main_result(dialog, tmp_path, monkeypatch):
    load_dataset(dialog, tmp_path / "data")
    queued = []
    monkeypatch.setattr(dialog_module.QThreadPool, "globalInstance", lambda: SimpleNamespace(start=queued.append))

    def fail(*args, **kwargs):
        raise ValueError("标定图像不足")

    monkeypatch.setattr(dialog_module, "calibrate_dataset", fail)
    dialog.page.result = {"id": "preserved"}
    dialog._run()
    queued[0].run()
    assert dialog.task is None
    assert dialog.run_button.isEnabled()
    assert dialog.close_button.isEnabled()
    assert dialog.method.isEnabled()
    assert dialog.distortion_mode.isEnabled()
    assert "标定图像不足" in dialog.summary.text()
    assert dialog.report is None
    assert dialog.page.result == {"id": "preserved"}


def test_simulation_entry_passes_dataset_to_new_dialog_and_blocks_it_while_busy(dialog, tmp_path, monkeypatch):
    import app.dialogs.robot_position_simulation_dialog as simulation_module

    source = load_dataset(dialog, tmp_path / "data")
    simulation = RobotPositionSimulationDialog(dialog.page)
    simulation.data_root.setText(str(source))
    simulation._load_dataset()
    called = []
    monkeypatch.setattr(RobotCameraCalibrationDialog, "exec", lambda window: called.append(
        (window.page, window.dataset, window.batch.currentData()),
    ))
    monkeypatch.setattr(simulation_module, "fit_dialog", lambda *args: None)
    simulation.calibration_button.click()
    assert called == [(dialog.page, source.resolve(), "B000")]
    simulation.task = object()
    simulation._set_running(True)
    assert not simulation.calibration_button.isEnabled()
    simulation.task = None
    simulation._set_running(False)
    assert simulation.calibration_button.isEnabled()
    simulation.close()
