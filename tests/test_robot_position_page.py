"""正式 UI 回归：缺失量保持空白、指标切换、历史隔离与异步任务。"""

import json

import numpy as np
import pytest
from PyQt6.QtCore import QEventLoop, QPoint, QRect, QThread, QTimer
from PyQt6.QtWidgets import QApplication, QBoxLayout

from app.pages.robot_position_page import RobotPositionPage
from app.dialogs.robot_position_debug_dialog import RobotPositionDebugDialog
from app.resources import UiScale, load_stylesheet
from core.algorithms.position_monitoring import evaluate_position_monitoring
from core.services.position_monitoring_service import PositionMonitoringService


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


class MemoryService:
    def __init__(self):
        self.parameters = {"version": "p1", "hand_eye": np.eye(4).tolist()}
        self.settings = {"thresholds": {}, "processing_points": []}
        self.baseline = {"id": "b1", "label": "基准", "created_at": "2026-09-28", "batch": {"samples": []}}
        self.current_batch = None
        self.history = []

    def list_history(self, baseline_id=None):
        return [row for row in self.history if baseline_id is None or row["baseline_id"] == baseline_id]


def result(with_reference=False, repeats=1, drift=2):
    initial, current = [], []
    for index in range(repeats):
        before = np.eye(4)
        before[0, 3] = index * 0.1
        after = before.copy()
        after[0, 3] -= drift
        initial.append({"point_id": "P1", "direction_id": "D1", "vision_pose": before.tolist()})
        current.append({"point_id": "P1", "direction_id": "D1", "vision_pose": after.tolist()})
    data = evaluate_position_monitoring(
        initial, current, np.eye(4),
        base_rotations={"P1": np.eye(3)} if with_reference else {},
        initial_errors={"P1": {"D1": [1, 0, 0]}} if with_reference else {},
    )
    return {**data, "id": "r1", "baseline_id": "b1", "parameter_version": "p1",
            "batch_id": "current", "created_at": "2026-09-28", "status": "未设阈值", "alarms": []}


def test_missing_reference_and_single_arrival_do_not_fabricate_metrics(application):
    service = MemoryService()
    page = RobotPositionPage(service)
    page._evaluation_completed(result())
    assert page.axis_values["X"].text() == "—"
    assert page.axis_values["distance"].text() == "2.0000"
    assert page.measured_table.item(0, 5).text() == "1 → 1"
    assert "初始参考末端系 Δp (mm)：[2.0000, 0.0000, 0.0000]" in page._result_details_text()
    page.result_metric.setCurrentIndex(page.result_metric.findData("repeatability"))
    assert page.axis_values["distance"].text() == "—"
    page.result_metric.setCurrentIndex(page.result_metric.findData("absolute"))
    assert page.axis_values["distance"].text() == "—"
    page.close()


def test_metric_switch_and_history_use_matching_reference(application):
    service = MemoryService()
    data = result(with_reference=True, repeats=3)
    service.history = [data, {**data, "baseline_id": "other"}, {**data, "parameter_version": "other"}]
    page = RobotPositionPage(service)
    page._evaluation_completed(data)
    assert page.axis_values["X"].text() == "2.0000"
    assert len(page.trend_chart.history) == 1
    page.result_metric.setCurrentIndex(page.result_metric.findData("absolute"))
    assert page.axis_values["distance"].text() == "3.0000"
    page.result_metric.setCurrentIndex(page.result_metric.findData("absolute_change"))
    assert page.axis_values["distance"].text() == "2.0000"
    page._invalidate_result()
    assert page.axis_values["distance"].text() == "—"
    assert page.measured_table.rowCount() == 0
    page.close()


def test_background_completion_returns_to_ui_thread(application):
    page = RobotPositionPage(MemoryService())
    event_loop = QEventLoop()
    observations = []

    def operation(progress):
        progress(50, "正在计算")
        return QThread.currentThread() != application.thread()

    def completed(ran_in_worker):
        observations.append((ran_in_worker, QThread.currentThread() == application.thread()))
        event_loop.quit()

    page._run_task("测试任务", operation, completed)
    QTimer.singleShot(3000, event_loop.quit)
    event_loop.exec()
    assert observations == [(True, True)]
    assert page.task is None
    assert all(button.isEnabled() for button in page.mutation_buttons)
    page.close()


def test_debug_baseline_and_retest_use_real_service_and_persist_results(application, tmp_path):
    service = PositionMonitoringService(root=tmp_path / "project")
    service.save_parameters({"hand_eye": np.eye(4).tolist()})
    service.save_settings({"thresholds": {"X": 0.2, "Y": None, "Z": None, "distance": None}})
    sources = []
    for period in ("baseline", "current"):
        samples = []
        for index in range(3):
            vision = np.eye(4)
            vision[:3, 3] = [100 + 0.1 * index, 10, 500]
            if period == "current":
                vision[:3, 3] += [-0.4, 0.3, 0]
            samples.append({"point_id": "P1", "direction_id": "D1",
                            "sample_id": str(index), "vision_pose": vision.tolist()})
        document = {
            "batch_id": period, "label": period, "length_unit": "mm",
            "program_id": "same-program", "target_id": "same-fixed-board",
            "comparison_status": "debug_unverified", "samples": samples,
            "base_rotations": {"P1": np.eye(3).tolist()},
        }
        path = tmp_path / f"{period}.json"
        path.write_text(json.dumps(document), encoding="utf-8")
        sources.append(path)
    original_bytes = [path.read_bytes() for path in sources]
    page = RobotPositionPage(service)
    dialog = RobotPositionDebugDialog(page)
    assert [dialog.method.itemText(index) for index in range(dialog.method.count())] == ["PARK", "TSAI"]

    def finish_background_task():
        event_loop = QEventLoop()
        poll = QTimer()
        poll.setInterval(10)
        poll.timeout.connect(lambda: event_loop.quit() if dialog.task is None else None)
        timeout = QTimer()
        timeout.setSingleShot(True)
        timeout.timeout.connect(event_loop.quit)
        poll.start()
        timeout.start(3000)
        event_loop.exec()
        poll.stop()
        timeout.stop()
        assert dialog.task is None, dialog.output.toPlainText()

    dialog.baseline_path.setText(str(sources[0]))
    dialog._build_baseline()
    finish_background_task()
    assert service.baseline is not None, dialog.output.toPlainText()
    assert service.current_batch is None
    assert page.result is None
    assert "旧数据调试基准" in page.baseline_label.text()

    dialog.current_path.setText(str(sources[1]))
    dialog._evaluate()
    finish_background_task()
    assert page.result is not None, dialog.output.toPlainText()
    assert page.result["groups"][0]["drift_base"] == pytest.approx([0.4, -0.3, 0], abs=1e-10)
    assert page.measured_table.item(0, 2).text() == "-0.3000"
    assert page.axis_values["Y"].text() == "0.3000"
    assert "调试比较" in page.conclusion.text()
    assert len(page.result["alarms"]) == 1
    assert len(page.trend_chart.history) == 1
    page.result_metric.setCurrentIndex(page.result_metric.findData("repeatability"))
    assert page.axis_values["distance"].text() != "—"
    page.result_metric.setCurrentIndex(page.result_metric.findData("absolute"))
    assert page.axis_values["distance"].text() == "—"
    reopened = PositionMonitoringService(root=service.root)
    assert reopened.baseline["id"] == service.baseline["id"]
    assert reopened.list_history() == [page.result]
    assert [path.read_bytes() for path in sources] == original_bytes
    dialog.close()
    page.close()


def test_narrow_page_stacks_panels_without_horizontal_clipping(application):
    previous_style = application.styleSheet()
    application.setStyleSheet(load_stylesheet(0.75))
    service = MemoryService()
    data = result(with_reference=True, repeats=3, drift=1234.5678)
    data.update({"batch_id": "c7364c180a1d43f3aca8d861f21590b2",
                 "created_at": "2026-09-28T09:21:40.123456+00:00"})
    service.history = [data]
    page = RobotPositionPage(service)
    try:
        UiScale(page).apply(0.75)
        page._evaluation_completed(data)
        # 模拟 800px 主窗口扣除侧栏后的业务页面宽度。
        page.resize(640, 500)
        page.show()
        application.processEvents()
        application.processEvents()
        assert page.columns.direction() == QBoxLayout.Direction.TopToBottom
        assert page.axis_values["X"].text() == "1234.5678"
        assert page.scroll_area.horizontalScrollBar().maximum() == 0
        assert page.scroll_area.verticalScrollBar().maximum() > 0
        evaluate_button = next(button for button in page.mutation_buttons if button.text() == "评估精度")
        page.scroll_area.ensureWidgetVisible(evaluate_button)
        application.processEvents()
        viewport = page.scroll_area.viewport()
        button_bounds = QRect(evaluate_button.mapTo(viewport, QPoint(0, 0)), evaluate_button.size())
        assert viewport.rect().contains(button_bounds)
        page.resize(1250, 700)
        application.processEvents()
        assert page.columns.direction() == QBoxLayout.Direction.LeftToRight
        assert page.scroll_area.horizontalScrollBar().maximum() == 0
    finally:
        page.close()
        application.setStyleSheet(previous_style)
