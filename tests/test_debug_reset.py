"""调试 reset 的数据边界、参数保留、跨进程命令和实际窗口刷新。"""

import os
import subprocess
import sys
from threading import Event
from time import monotonic

import numpy as np
import pytest
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QDialog

import app.pages.robot_position_page as robot_page
import app.pages.spindle_rotation_page as spindle_page
from app.resources import PROJECT_ROOT
from core.services.position_monitoring_service import PositionMonitoringService
from core.services.position_persistence import read_json, write_document
from core.services.spindle_monitoring_service import SpindleMonitoringService
from debug.reset import DebugSession, reset_usage_data
from ui_helpers import wait_for_page


@pytest.fixture
def root(tmp_path):
    write_document(tmp_path / "config/spindle_monitoring.json",
                   read_json(PROJECT_ROOT / "config/spindle_monitoring.json"))
    return tmp_path


@pytest.mark.parametrize("legacy", [False, True])
def test_reset_removes_usage_and_preserves_parameters_settings_and_sources(root, legacy):
    position = PositionMonitoringService(root)
    position.save_parameters({"hand_eye": np.eye(4).tolist()})
    position.save_settings({"multidirectional_thresholds": {"repeatability": {"X": 0.25}},
                            "processing_points": [{"id": "P1", "x": 1, "y": 2, "z": 3}]})
    state = read_json(position.storage / "state.json")
    state.update(baseline_path="old-baseline", current_batch_path="old-batch", latest_result={"id": "old"})
    if legacy:
        state["parameters"] = position.parameters
        position.parameter_path.unlink()
    write_document(position.storage / "state.json", state)
    spindle = SpindleMonitoringService(root / "storage/spindle_monitoring")
    spindle.set_thresholds(2, 4)
    spindle.state.update(runs={"old": {"manual_label": "healthy"}}, packages=["old"],
                         models=[{"version": "old"}], current_model_version="old", results=["old"])
    spindle.settings["thresholds_model_version"] = "old"
    write_document(spindle.state_path, spindle.state)
    for folder in ("images", "observations", "baselines", "daily", "debug", "history", "logs"):
        write_document(position.storage / folder / "old.json", {"old": True})
    for folder in ("packages", "models", "evaluations"):
        write_document(spindle.root / folder / "old.json", {"old": True})
    for name in ("data/source.json", "debug/scene/output/record.json", "storage/other/keep.json"):
        write_document(root / name, {"keep": True})
    protected = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()
                 and (not path.is_relative_to(position.storage) and not path.is_relative_to(spindle.root)
                      or path.is_relative_to(position.storage / "parameters"))}

    reset_usage_data(root)
    reset_usage_data(root)  # 重复执行仍为空，不丢失第一次保留的设置。

    restored = PositionMonitoringService(root)
    assert restored.parameters == position.parameters
    assert restored.settings == position.settings
    assert restored.baseline is restored.current_batch is restored.latest_result is None
    assert restored.list_history() == restored.list_baselines() == restored.list_logs() == []
    assert {path.name for path in position.storage.iterdir()} == {"parameters", "state.json"}
    spindle = SpindleMonitoringService(spindle.root)
    assert spindle.settings["thresholds"] == {"warning": 2, "fault": 4}
    assert spindle.settings["thresholds_model_version"] is None
    assert spindle.runs == {} and spindle.models == [] and spindle.history() == []
    assert spindle.state["packages"] == [] and spindle.current_model is None
    assert {path.name for path in spindle.root.iterdir()} == {"state.json"}
    assert all(path.read_bytes() == content for path, content in protected.items())


@pytest.mark.parametrize("target", ["data", "storage", "storage/position_monitoring",
                                   "storage/position_monitoring/images", "../outside"])
def test_reset_rejects_unsafe_storage_before_changing_files(root, target):
    config = read_json(root / "config/spindle_monitoring.json")
    config["storage_root"] = target
    write_document(root / "config/spindle_monitoring.json", config)
    write_document(root / "storage/position_monitoring/state.json", {"keep": True})
    before = {path: path.read_bytes() for path in root.rglob("*") if path.is_file()}
    with pytest.raises(ValueError):
        reset_usage_data(root)
    assert {path: path.read_bytes() for path in root.rglob("*") if path.is_file()} == before


def test_unreadable_state_does_not_partially_clear_other_service(root):
    write_document(root / "storage/position_monitoring/state.json", {"keep": True})
    state_path = root / "storage/spindle_monitoring/state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text("broken json", encoding="utf-8")
    with pytest.raises(ValueError):
        reset_usage_data(root)
    assert read_json(root / "storage/position_monitoring/state.json") == {"keep": True}


@pytest.fixture
def session(application, root, monkeypatch):
    position = PositionMonitoringService(root)
    position.save_parameters({"hand_eye": np.eye(4).tolist(), "target_pose_base": np.eye(4).tolist()})
    source = root / "data/batch.json"
    write_document(source, {
        "batch_id": "B001", "length_unit": "mm", "program_id": "test", "target_id": "test",
        "sampling_protocol": "multidirectional", "comparison_status": "simulation",
        "samples": [{"point_id": "P1", "direction_id": str(index), "sample_id": str(index),
                     "ideal_pose": np.eye(4).tolist(), "vision_pose": np.eye(4).tolist()}
                    for index in range(2)],
    })
    position.load_observations(source)
    position.create_baseline()
    monkeypatch.setattr(robot_page, "PositionMonitoringService", lambda **kw: PositionMonitoringService(root, **kw))
    monkeypatch.setattr(spindle_page, "SpindleMonitoringService",
                        lambda: SpindleMonitoringService(root / "storage/spindle_monitoring"))
    session = DebugSession(application, root)
    assert session.execute("show")["ok"]
    wait_for_page(session.window.precision_page.content_stack.widget(0))
    yield session
    wait_for_page(session.window.precision_page.content_stack.widget(0))
    session.server.close()
    session.window.close()
    session.window.deleteLater()
    session.deleteLater()
    QTest.qWait(30)


def test_reset_command_from_another_process_clears_display_and_disk(session, root):
    previous = session.window
    previous.precision_page.tab_bar.setCurrentIndex(1)
    robot = previous.precision_page.content_stack.widget(0)
    assert robot.result is not None and robot.saved_history and robot.trend_chart.history
    parameters = robot.service.parameter_path.read_bytes()
    command = [sys.executable, "-B", "-c",
               "import sys; from PyQt6.QtCore import QCoreApplication; "
               "from debug.reset import send_command; app=QCoreApplication([]); "
               "result=send_command('reset', sys.argv[1]); "
               "sys.exit(0 if result and result['ok'] else 1)", str(root)]
    with subprocess.Popen(command, cwd=PROJECT_ROOT, env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
        deadline = monotonic() + 15
        while process.poll() is None and monotonic() < deadline:
            QTest.qWait(20)
        if process.poll() is None:
            process.kill()
        output = process.communicate()
        assert process.returncode == 0, output
    robot = session.window.precision_page.content_stack.widget(0)
    spindle = session.window.precision_page.content_stack.widget(1)
    wait_for_page(robot)
    assert session.window is not previous
    assert session.window.precision_page.tab_bar.currentIndex() == 1
    assert robot.result is None and robot.saved_history == [] and robot.trend_chart.history == []
    assert all(value.text() == "—" for value in robot.axis_values.values())
    assert robot.process_log.toPlainText() == ""
    assert robot.service.parameter_path.read_bytes() == parameters
    assert spindle.result is None and spindle.run_select.count() == 0
    assert all(value.text() == "—" for value in spindle.result_values.values())
    assert (root / "data/batch.json").is_file()


@pytest.mark.parametrize("page_index", [0, 1])
def test_reset_refuses_active_background_work(session, root, page_index):
    page = session.window.precision_page.content_stack.widget(page_index)
    released = Event()
    before = (root / "storage/position_monitoring/state.json").read_bytes()
    page._run_task("测试任务", lambda _progress: released.wait(5), lambda _result: None)
    try:
        result = session.execute("reset")
        assert not result["ok"] and "任务尚未结束" in result["message"]
        assert (root / "storage/position_monitoring/state.json").read_bytes() == before
    finally:
        released.set()
        deadline = monotonic() + 5
        while page.task is not None and monotonic() < deadline:
            QTest.qWait(10)
        assert page.task is None


def test_reset_refuses_open_dialog(session):
    dialog = QDialog(session.window)
    dialog.setModal(True)
    dialog.show()
    try:
        assert not session.execute("reset")["ok"]
        assert session.window.precision_page.content_stack.widget(0).saved_history
    finally:
        dialog.close()
        dialog.deleteLater()
