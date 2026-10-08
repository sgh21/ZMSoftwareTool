"""仿真核验窗口的隐藏控件测试；图像测量通过假服务隔离。"""

from copy import deepcopy
from types import SimpleNamespace

import pytest
from PyQt6.QtGui import QCloseEvent

import app.dialogs.robot_position_simulation_dialog as dialog_module
from app.dialogs.robot_position_simulation_dialog import RobotPositionSimulationDialog
from ui_helpers import ready_position_page
from core.services.position_monitoring_service import PositionMonitoringService, write_document


@pytest.fixture
def dialog(application, tmp_path):
    page = ready_position_page(PositionMonitoringService(root=tmp_path / "application"))
    page.ui_scale = 1.0
    window = RobotPositionSimulationDialog(page)
    yield window
    window.task = None
    window.close()
    page.close()


def dataset(root, *, missing_record=None):
    runs = []
    for batch in ("B000", "B001", "B002"):
        runs.append({"batch_id": batch, "record": f"{batch}/record.json"})
        if batch != missing_record:
            write_document(root / batch / "record.json", {})
    write_document(root / "parameters.json", {"runs": runs})
    return root


def load_dataset(dialog, path):
    dialog.data_root.setText(str(path))
    dialog._load_dataset()


def report():
    summary = {"sample_count": 12}
    for index, prefix in enumerate(("mean", "p95", "max"), 1):
        summary[f"{prefix}_abs_xyz_mm"] = [0.01 * index, 0.02 * index, 0.03 * index]
        summary[f"{prefix}_distance_mm"] = 0.04 * index
        summary[f"{prefix}_rotation_deg"] = 0.05 * index
    return {
        "baseline_batch_id": "B001", "current_batch_id": "B002",
        "measurement_summary": summary,
        "metric_comparison": [{
            "metric": mode, "label": label, "measured": [1.1, 2.2, 3.3, 4.4],
            "truth": [1, 2, 3, 4], "difference": [0.1, 0.2, 0.3, 0.4],
        } for mode, label in (
            ("absolute_change", "绝对定位精度退化"),
            ("repeatability_change", "多方向到位散布退化"),
            ("repeatability", "当前多方向到位散布"),
        )],
    }


def test_dataset_batch_defaults_and_missing_input_are_explicit(dialog, tmp_path):
    assert dialog.data_root.text() == str(dialog.page.service.root / "debug")
    assert not dialog.run_button.isEnabled()
    load_dataset(dialog, tmp_path / "missing")
    assert "parameters.json" in dialog.source_status.text()
    assert not dialog.run_button.isEnabled()
    path = dataset(tmp_path / "data")
    load_dataset(dialog, path)
    assert dialog.baseline_batch.currentData() == "B001"
    assert dialog.current_batch.currentData() == "B002"
    assert dialog.run_button.isEnabled()
    assert dialog.baseline_batch.property("robotInput")
    assert dialog.current_batch.property("robotInput")
    dialog.current_batch.setCurrentIndex(dialog.current_batch.findData("B001"))
    assert "不同批次" in dialog.source_status.text()
    assert not dialog.run_button.isEnabled()


def test_missing_batch_record_disables_measurement(dialog, tmp_path):
    load_dataset(dialog, dataset(tmp_path / "data", missing_record="B002"))
    assert "缺少批次记录" in dialog.source_status.text()
    assert "B002" in dialog.source_status.text()
    assert not dialog.run_button.isEnabled()
    dialog.current_batch.setCurrentIndex(dialog.current_batch.findData("B000"))
    assert dialog.run_button.isEnabled()


def test_background_task_uses_isolated_output_and_renders_tables(dialog, tmp_path, monkeypatch):
    source = dataset(tmp_path / "data")
    load_dataset(dialog, source)
    queued = []
    monkeypatch.setattr(dialog_module.QThreadPool, "globalInstance", lambda: SimpleNamespace(start=queued.append))
    captured = {}

    def calculate(dataset_path, output_root, baseline_id, current_id, progress):
        captured.update({"dataset": dataset_path, "output": output_root,
                         "baseline": baseline_id, "current": current_id})
        progress(55, "读取图像中")
        return {"report": report(), "report_path": str(output_root / "verification.json")}

    monkeypatch.setattr(dialog_module, "run_simulation_check", calculate)
    service = dialog.page.service
    parameters = deepcopy(service.parameters)
    baseline = deepcopy(service.baseline)
    files = {path: path.read_bytes() for path in service.storage.rglob("*.json")}
    dialog._run()
    assert len(queued) == 1
    assert dialog.task is queued[0]
    assert not dialog.run_button.isEnabled()
    assert not dialog.calibration_button.isEnabled()
    assert not dialog.close_button.isEnabled()
    assert not dialog.data_root.isEnabled()
    event = QCloseEvent()
    dialog.closeEvent(event)
    assert not event.isAccepted()
    dialog.reject()
    assert "等待完成" in dialog.progress_note.text()
    queued[0].run()
    assert dialog.task is None
    assert dialog.run_button.isEnabled()
    assert captured["dataset"] == source.resolve()
    assert captured["baseline"] == "B001"
    assert captured["current"] == "B002"
    assert captured["output"].parent == service.root / "data" / "processed" / "robot_position_simulation"
    assert dialog.metric_table.rowCount() == 9
    assert dialog.error_table.rowCount() == 3
    assert dialog.metric_table.item(2, 1).text() == "测量 − 真值"
    assert dialog.metric_table.item(2, 2).text() == "0.100000"
    assert "12 张图像" in dialog.summary.text()
    assert dialog.progress.value() == 100
    assert "verification.json" in dialog.report_path.text()
    assert service.parameters == parameters
    assert service.baseline == baseline
    assert dialog.page.result is None
    assert {path: path.read_bytes() for path in service.storage.rglob("*.json")} == files
    dialog.current_batch.setCurrentIndex(dialog.current_batch.findData("B000"))
    assert dialog.report is None
    assert dialog.metric_table.rowCount() == 0


def test_failure_restores_controls_without_touching_main_result(dialog, tmp_path, monkeypatch):
    load_dataset(dialog, dataset(tmp_path / "data"))
    queued = []
    monkeypatch.setattr(dialog_module.QThreadPool, "globalInstance", lambda: SimpleNamespace(start=queued.append))

    def fail(*args):
        raise ValueError("B002 图片缺失")

    monkeypatch.setattr(dialog_module, "run_simulation_check", fail)
    dialog.page.result = {"id": "keep-main-result"}
    dialog._run()
    queued[0].run()
    assert dialog.task is None
    assert dialog.run_button.isEnabled()
    assert dialog.close_button.isEnabled()
    assert "B002 图片缺失" in dialog.progress_note.text()
    assert dialog.metric_table.rowCount() == 0
    assert dialog.page.result == {"id": "keep-main-result"}


def test_new_batch_is_discovered_from_record_without_parameters_runs_entry(dialog, tmp_path):
    path = dataset(tmp_path / "data")
    write_document(path / "B003" / "record.json", {"batch": "B003", "frames": []})
    load_dataset(dialog, path)
    assert dialog.current_batch.findData("B003") >= 0
    dialog.current_batch.setCurrentIndex(dialog.current_batch.findData("B003"))
    assert dialog.run_button.isEnabled()
    assert "B001 → B003" in dialog.source_status.text()
