"""实际后台线程、界面响应、缓存及重复基准交互；全部使用临时存储。"""

from copy import deepcopy
from threading import Event

import numpy as np
import pytest
from PyQt6.QtCore import QEvent, QThread, QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QComboBox, QDialog, QPushButton

import app.pages.robot_position_page as page_module
from core.services.position_monitoring_service import PositionMonitoringService, write_document
from ui_helpers import ready_position_page, wait_for_page


def import_batch(service, batch_id, radius):
    samples = []
    for index, sign in enumerate((-1, 1)):
        pose = np.eye(4)
        pose[0, 3] = sign * radius
        samples.append({"point_id": "P1", "direction_id": str(index), "sample_id": str(index),
                        "ideal_pose": np.eye(4).tolist(), "vision_pose": pose.tolist()})
    source = service.root / f"{batch_id}.json"
    write_document(source, {"batch_id": batch_id, "length_unit": "mm", "program_id": "fixed",
                            "target_id": "fixed-board", "sampling_protocol": "multidirectional",
                            "comparison_status": "simulation", "samples": samples})
    return service.load_observations(source)


@pytest.fixture
def setup_service(tmp_path):
    service = PositionMonitoringService(tmp_path)
    service.save_parameters({"hand_eye": np.eye(4).tolist(), "target_pose_base": np.eye(4).tolist()})
    import_batch(service, "B001", 0.2)
    first = service.create_baseline()
    import_batch(service, "B003", 0.6)
    last = service.create_baseline()
    return service, first, last


def test_startup_restores_in_worker_while_event_loop_remains_responsive(application, tmp_path, monkeypatch):
    service = PositionMonitoringService(tmp_path, defer_restore=True)
    restore = service.restore
    released = Event()
    ran_in_worker = []

    def delayed_restore(progress):
        ran_in_worker.append(QThread.currentThread() != application.thread())
        progress(-1, "读取定位记录")
        assert released.wait(2)
        return restore(progress)

    monkeypatch.setattr(service, "restore", delayed_restore)
    monkeypatch.setattr(page_module, "PositionMonitoringService", lambda **kwargs: service)
    page = page_module.RobotPositionPage()
    try:
        heartbeat = []
        QTimer.singleShot(0, lambda: heartbeat.append(True))
        QTest.qWait(30)
        assert heartbeat and ran_in_worker == [True]
        assert page.task is not None
        assert page.task_progress.maximum() == 0
        assert not page.show_before_baseline.isEnabled()
        released.set()
        wait_for_page(page)
        assert page.task_progress.value() == 100
        assert page.show_before_baseline.isEnabled()
    finally:
        released.set()
        wait_for_page(page, success=False)
        page.close()


def test_task_completion_survives_source_dialog_destruction(application, tmp_path):
    page = ready_position_page(PositionMonitoringService(tmp_path))
    dialog = QDialog(page)
    dialog_button = QPushButton("确认", dialog)
    released = Event()

    def operation(_progress):
        assert released.wait(2)

    try:
        page._run_task("弹窗发起任务", operation, lambda _result: None)
        assert dialog_button not in [control for control, _enabled in page._busy_controls]
        dialog.deleteLater()
        application.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        released.set()
        wait_for_page(page)
        assert page.task_progress.value() == 100
        assert page._busy_controls == []
        assert page.show_before_baseline.isEnabled()
    finally:
        released.set()
        wait_for_page(page, success=False)
        page.close()


def test_baseline_switch_waits_through_render_and_display_changes_use_cached_history(application, setup_service, monkeypatch):
    service, first, _last = setup_service
    page = ready_position_page(service)
    page.ui_scale = 1.0
    released = Event()
    selected = service.select_baseline
    compare = service.history_comparisons
    history_threads = []
    rendered_busy = []

    def delayed_select(path, progress=None):
        progress(-1, "读取所选基准")
        assert released.wait(2)
        return selected(path, progress)

    def comparisons(*args, **kwargs):
        history_threads.append(QThread.currentThread() != application.thread())
        return compare(*args, **kwargs)

    completed = page._baseline_selected

    def displayed(baseline):
        completed(baseline)
        rendered_busy.append(page.task is not None and page.task_progress.value() == 99
                             and not page.show_before_baseline.isEnabled())

    def choose(dialog):
        combo = dialog.findChild(QComboBox)
        combo.setCurrentIndex(combo.findData(first["path"]))
        next(button for button in dialog.findChildren(QPushButton) if button.text() == "选择").click()
        return dialog.result()

    monkeypatch.setattr(service, "select_baseline", delayed_select)
    monkeypatch.setattr(service, "history_comparisons", comparisons)
    monkeypatch.setattr(page, "_baseline_selected", displayed)
    monkeypatch.setattr(QDialog, "exec", choose)
    try:
        page._select_baseline()
        heartbeat = []
        QTimer.singleShot(0, lambda: heartbeat.append(True))
        QTest.qWait(30)
        assert heartbeat and page.task is not None
        assert page.task_progress.maximum() == 0
        released.set()
        wait_for_page(page)
        assert rendered_busy == [True]
        assert page.task_progress.value() == 100
        assert history_threads == [True]
        assert page.result["batch_id"] == "B003"
        assert page.result["baseline_id"] == first["id"]
        for index in range(3):
            page.result_metric.setCurrentIndex(index)
        page.trend_metric.setCurrentIndex(1)
        page.show_before_baseline.setChecked(True)
        wait_for_page(page)
        assert history_threads == [True]
        assert [row["days"] for row in page.trend_chart.history] == [0, 2]
    finally:
        released.set()
        wait_for_page(page, success=False)
        page.close()


def test_duplicate_baseline_notifies_without_switching_or_writing_another_evaluation(application, setup_service, monkeypatch):
    service, first, last = setup_service
    import_batch(service, "B001", 0.2)
    page = ready_position_page(service)
    saved = deepcopy(service.list_history())
    prompts = []

    def notified(existing):
        assert page.task is None
        prompts.append(existing)

    monkeypatch.setattr(page, "_show_duplicate_baseline", notified)
    try:
        page._create_baseline()
        wait_for_page(page)
        application.processEvents()
        assert [entry["id"] for entry in prompts] == [first["id"]]
        assert service.baseline["id"] == last["id"]
        assert service.current_batch["batch_id"] == "B001"
        assert page.result["batch_id"] == "B003"
        assert page.result["baseline_id"] == last["id"]
        assert service.list_history() == saved
        assert len(service.list_baselines()) == 2
        assert "未新建或切换基准" in page.process_log.toPlainText()
    finally:
        page.close()


def test_failed_history_preference_restores_checkbox_and_controls(application, setup_service, monkeypatch):
    service, _first, last = setup_service
    page = ready_position_page(service)
    history = deepcopy(page.trend_chart.history)
    assert [row["days"] for row in history] == [0]

    def fail_save(_settings, progress=None):
        raise OSError("设置文件无法写入")

    monkeypatch.setattr(service, "save_settings", fail_save)
    try:
        page.show_before_baseline.setChecked(True)
        wait_for_page(page, success=False)
        assert not page.show_before_baseline.isChecked()
        assert page.show_before_baseline.isEnabled()
        assert all(button.isEnabled() for button in page.findChildren(QPushButton))
        assert page.task_progress.value() == 0
        assert "设置文件无法写入" in page.task_progress_note.text()
        assert service.baseline["id"] == last["id"]
        assert page.trend_chart.history == history
    finally:
        page.close()
