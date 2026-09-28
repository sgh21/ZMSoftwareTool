"""正式 UI 回归：缺失量保持空白、指标切换、历史隔离与异步任务。"""

import json
from copy import deepcopy

import numpy as np
import pytest
from PyQt6.QtCore import QEventLoop, QPoint, QRect, QThread, QTimer, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import (
    QApplication, QBoxLayout, QLabel, QLineEdit, QPushButton, QStyle, QStyleOptionComboBox,
    QTableWidget, QTabWidget,
)

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
        self.settings = {"thresholds": {}, "metric_thresholds": {}, "processing_points": []}
        self.baseline = {"id": "b1", "label": "基准", "created_at": "2026-09-28", "batch": {"samples": []}}
        self.current_batch = None
        self.history = []

    def list_history(self, baseline_id=None):
        return [row for row in self.history if baseline_id is None or row["baseline_id"] == baseline_id]

    def save_settings(self, settings):
        self.settings.update(deepcopy(settings))


def result(with_reference=False, repeats=1, drift=2, scatter_scale=2):
    initial, current = [], []
    for index in range(repeats):
        before = np.eye(4)
        before[0, 3] = (index - (repeats - 1) / 2) * 0.1
        after = before.copy()
        after[0, 3] = scatter_scale * before[0, 3] - drift
        initial.append({"point_id": "P1", "direction_id": "D1", "vision_pose": before.tolist()})
        current.append({"point_id": "P1", "direction_id": "D1", "vision_pose": after.tolist()})
    data = evaluate_position_monitoring(
        initial, current, np.eye(4),
        base_rotations={"P1": np.eye(3)} if with_reference else {},
        initial_errors={"P1": {"D1": [1, 0, 0]}} if with_reference else {},
    )
    return {**data, "id": "r1", "baseline_id": "b1", "parameter_version": "p1",
            "batch_id": "current", "created_at": "2026-09-28", "status": "未设阈值", "alarms": [],
            "metric_thresholds": {}}


def test_missing_reference_and_single_arrival_do_not_fabricate_metrics(application):
    service = MemoryService()
    page = RobotPositionPage(service)
    page._evaluation_completed(result())
    assert page._metric_mode() == "repeatability"
    assert page.axis_values["X"].text() == "—"
    assert page.axis_values["distance"].text() == "—"
    assert page.measured_table.item(0, 5).text() == "1"
    assert "初始参考末端系 Δp (mm)：[2.0000, 0.0000, 0.0000]" in page._result_details_text()
    assert "样本" in page.result_hint.text() or "重复" in page.result_hint.text()
    page.result_metric.setCurrentIndex(1)
    assert page.axis_values["distance"].text() == "—"
    page.result_metric.setCurrentIndex(0)
    assert page.axis_values["distance"].text() == "—"
    assert "初始" in page.result_hint.text() and "误差" in page.result_hint.text()
    assert "正常" not in page.conclusion.text()
    page.close()


def test_metric_switch_and_history_use_matching_reference(application):
    service = MemoryService()
    data = result(with_reference=True, repeats=3)
    service.history = [data, {**data, "baseline_id": "other"}, {**data, "parameter_version": "other"}]
    page = RobotPositionPage(service)
    page._evaluation_completed(data)
    assert [page.result_metric.itemData(index) for index in range(page.result_metric.count())] == [
        "absolute_change", "repeatability_change", "repeatability",
    ]
    assert page.result_metric.currentIndex() == 2
    before = deepcopy((data, service.parameters, service.settings, service.baseline, service.current_batch))
    expected = [
        (0, "绝对定位精度退化", [2.0, 0, 0, 2.0]),
        (1, "重复定位精度退化", [0.3, 0, 0, 0.2398717474235544]),
        (2, "当前重复定位精度", [0.6, 0, 0, 0.4797434948471088]),
    ]
    for index, name, values in expected:
        page.result_metric.setCurrentIndex(index)
        assert page._metric_name() == name
        assert name in page.trend_title.text()
        assert name in page.measured_title.text()
        assert name in page.conclusion.text()
        for column, (key, value) in enumerate(zip(("X", "Y", "Z", "distance"), values), start=1):
            assert page.axis_values[key].text() == f"{value:.4f}"
            assert page.measured_table.item(0, column).text() == f"{value:.4f}"
            assert page.trend_chart.history[0][key] == pytest.approx(value)
        assert len(page.trend_chart.history) == 1
        assert page.measured_table.horizontalHeaderItem(4).text() == "空间指标 / mm"
        assert [page.axis_titles[key].text() for key in ("X", "Y", "Z")] == [
            "X 方向", "Y 方向", "Z 方向",
        ]
    assert (data, service.parameters, service.settings, service.baseline, service.current_batch) == before
    page._invalidate_result()
    assert page.axis_values["distance"].text() == "—"
    assert page.measured_table.rowCount() == 0
    page.close()


def test_historical_metrics_use_saved_thresholds_and_ignore_drift_alarms(application):
    service = MemoryService()
    service.settings["metric_thresholds"] = {"repeatability": {"X": 0.1}}
    service.settings["thresholds"] = {"X": 0.01}
    data = result(with_reference=True, repeats=3)
    data.update({
        "status": "超限",
        "alarms": [{"point_id": "P1", "direction_id": "D1", "axis": "X", "value": 2, "threshold": 0.01}],
        "thresholds": {"X": 0.01},
        "metric_thresholds": {
            "repeatability": {"X": 0.7},
            "repeatability_change": {"X": 0.1},
            "absolute_change": {"X": 3.0},
        },
    })
    page = RobotPositionPage(service)
    page._evaluation_completed(data)
    assert page.axis_thresholds["X"].text() == "阈值：0.7000"
    assert "1 项超限" not in page.alarm_status.text()
    assert "定位漂移" not in page.conclusion.text()
    page.result_metric.setCurrentIndex(1)
    assert page.axis_thresholds["X"].text() == "阈值：0.1000"
    assert "1 项超限" in page.alarm_status.text()
    assert "超限" in page.conclusion.text()
    page.result_metric.setCurrentIndex(0)
    assert page.axis_thresholds["X"].text() == "阈值：3.0000"
    assert "1 项超限" not in page.alarm_status.text()
    page.result_metric.setCurrentIndex(2)
    # 旧记录只有漂移阈值时，不能借用当前配置补判历史 RP。
    data.pop("metric_thresholds")
    page._evaluation_completed(data)
    assert "未设置" in page.axis_thresholds["X"].text()
    assert "未设" in page.conclusion.text()
    assert "1 项超限" not in page.alarm_status.text()
    page.close()


def test_threshold_dialog_saves_three_independent_sets_and_blank_fields(application):
    service = MemoryService()
    page = RobotPositionPage(service)
    page.ui_scale = 1.0
    observed = {}

    def fill_and_save():
        dialog = application.activeModalWidget()
        observed["title"] = dialog.windowTitle()
        fields = dialog.findChildren(QLineEdit)
        observed["field_count"] = len(fields)
        tabs = dialog.findChild(QTabWidget)
        observed["tab_count"] = tabs.count()
        observed["selected"] = tabs.currentIndex()
        for mode, value in (("absolute_change", "1.2"), ("repeatability_change", "0.2"),
                            ("repeatability", "0.7")):
            dialog.findChild(QLineEdit, f"threshold_{mode}_X").setText(value)
        next(button for button in dialog.findChildren(QPushButton) if button.text() == "保存").click()

    QTimer.singleShot(0, fill_and_save)
    page._show_settings()
    assert observed == {"title": "定位精度阈值", "field_count": 12, "tab_count": 3, "selected": 2}
    assert service.settings["metric_thresholds"] == {
        "absolute_change": {"X": 1.2, "Y": None, "Z": None, "distance": None},
        "repeatability_change": {"X": 0.2, "Y": None, "Z": None, "distance": None},
        "repeatability": {"X": 0.7, "Y": None, "Z": None, "distance": None},
    }
    assert page.axis_thresholds["X"].text() == "阈值：0.7000"
    page.result_metric.setCurrentIndex(0)
    assert page.axis_thresholds["X"].text() == "阈值：1.2000"
    page.close()


def test_current_repeatability_can_be_evaluated_without_baseline(application, tmp_path):
    service = PositionMonitoringService(root=tmp_path / "project")
    service.save_parameters({"hand_eye": np.eye(4).tolist()})
    samples = []
    for index in range(3):
        vision = np.eye(4)
        vision[0, 3] = index * 0.1
        samples.append({"point_id": "P1", "direction_id": "D1",
                        "sample_id": str(index), "vision_pose": vision.tolist()})
    source = tmp_path / "current.json"
    source.write_text(json.dumps({
        "batch_id": "current", "label": "本次观测", "length_unit": "mm",
        "program_id": "fixed-program", "target_id": "fixed-board", "samples": samples,
        "base_rotations": {"P1": np.eye(3).tolist()},
    }), encoding="utf-8")
    service.load_observations(source)
    page = RobotPositionPage(service)
    page._evaluate()
    event_loop = QEventLoop()
    timer = QTimer()
    timer.timeout.connect(lambda: event_loop.quit() if page.task is None else None)
    timer.start(10)
    QTimer.singleShot(3000, event_loop.quit)
    event_loop.exec()
    timer.stop()
    assert page.task is None
    assert page.result is not None, page.process_log.toPlainText()
    assert service.baseline is None
    assert page.axis_values["X"].text() == "0.3000"
    assert page.axis_values["distance"].text() == "0.2399"
    assert page.measured_table.item(0, 5).text() == "3"
    for index in (0, 1):
        page.result_metric.setCurrentIndex(index)
        assert all(label.text() == "—" for label in page.axis_values.values())
        assert "基准" in page.result_hint.text()
    page.result_metric.setCurrentIndex(2)
    assert page.axis_values["distance"].text() == "0.2399"
    page.close()


def test_evaluation_restores_current_observations_after_viewing_history(application, tmp_path):
    service = MemoryService()
    historical_sample = {"point_id": "P1", "direction_id": "D1", "sample_id": "history-A"}
    current_sample = {"point_id": "P1", "direction_id": "D1", "sample_id": "current-B"}
    snapshot = tmp_path / "history-A.json"
    snapshot.write_text(json.dumps({"samples": [historical_sample]}), encoding="utf-8")
    historical_result = {**result(with_reference=True, repeats=3), "id": "history-A",
                         "batch_id": "history-A", "current_batch_path": str(snapshot)}
    service.history = [historical_result]
    service.current_batch = {"batch_id": "current-B", "samples": [current_sample]}
    page = RobotPositionPage(service)
    page.ui_scale = 1.0
    page._observations_loaded(service.current_batch)
    assert page.observation_sample.currentData() == current_sample

    def select_history():
        dialog = application.activeModalWidget()
        dialog.findChild(QTableWidget).cellDoubleClicked.emit(0, 0)

    QTimer.singleShot(0, select_history)
    page._show_history()
    assert page.observation_sample.currentData() == historical_sample
    assert page.result["batch_id"] == "history-A"

    current_result = {**result(with_reference=True, repeats=3), "id": "current-B", "batch_id": "current-B"}
    page._evaluation_completed(current_result)
    assert page.history_batches is None
    assert page.observation_sample.count() == 1
    assert page.observation_sample.currentData() == current_sample
    assert "current-B" in page.observation_sample.currentText()
    assert service.current_batch["samples"] == [current_sample]
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
    service.save_settings({"metric_thresholds": {
        "repeatability": {"X": 0.2, "Y": None, "Z": None, "distance": None},
    }})
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
    assert page.measured_table.item(0, 1).text() == "0.3000"
    assert page.axis_values["Y"].text() == "0.0000"
    assert "调试比较" in page.conclusion.text()
    assert len(page.result["alarms"]) == 1
    assert len(page.trend_chart.history) == 1
    assert page._metric_mode() == "repeatability"
    assert page.axis_values["distance"].text() != "—"
    page.result_metric.setCurrentIndex(0)
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
        page.result_metric.setCurrentIndex(0)
        # 模拟 800px 主窗口扣除侧栏后的业务页面宽度。
        page.resize(640, 500)
        page.show()
        application.processEvents()
        application.processEvents()
        assert page.columns.direction() == QBoxLayout.Direction.TopToBottom
        assert page.axis_values["X"].text() == "1234.5678"
        assert page.scroll_area.horizontalScrollBar().maximum() == 0
        assert page.scroll_area.verticalScrollBar().maximum() > 0
        combo_option = QStyleOptionComboBox()
        page.result_metric.initStyleOption(combo_option)
        text_bounds = page.result_metric.style().subControlRect(
            QStyle.ComplexControl.CC_ComboBox, combo_option,
            QStyle.SubControl.SC_ComboBoxEditField, page.result_metric,
        )
        for index in range(page.result_metric.count()):
            assert text_bounds.width() >= page.result_metric.fontMetrics().horizontalAdvance(
                page.result_metric.itemText(index)
            )
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


@pytest.mark.parametrize("combo_name", ["result_metric", "trend_metric"])
def test_dropdown_stays_below_centered_row_for_every_selection(application, combo_name):
    previous_style = application.styleSheet()
    application.setStyleSheet(load_stylesheet())
    page = RobotPositionPage(MemoryService())
    page._evaluation_completed(result(with_reference=True, repeats=3))
    combo = getattr(page, combo_name)
    try:
        page.resize(1250, 700)
        page.move(application.primaryScreen().availableGeometry().topLeft() + QPoint(20, 20))
        page.show()
        page.activateWindow()
        assert QTest.qWaitForWindowActive(page)
        application.processEvents()
        application.processEvents()
        title = page.findChild(QLabel, "RobotPageTitle")
        metric_index = page.result_metric.currentIndex()
        point_values = [
            [page.measured_table.item(row, column).text()
             for column in range(page.measured_table.columnCount())]
            for row in range(page.measured_table.rowCount())
        ]
        popup_positions = []
        for index in range(combo.count()):
            combo.setCurrentIndex(index)
            application.processEvents()
            if combo_name == "trend_metric":
                assert page.result_metric.currentIndex() == metric_index
                assert [
                    [page.measured_table.item(row, column).text()
                     for column in range(page.measured_table.columnCount())]
                    for row in range(page.measured_table.rowCount())
                ] == point_values
                assert page.trend_chart.mode == combo.currentData()
                labels = [page.trend_axis_title] + [
                    label for label in page.trend_legend.values() if label.isVisible()
                ]
                expected_axes = {"X", "Y", "Z"} if index == 0 else {"distance"}
                assert {
                    axis for axis, label in page.trend_legend.items() if label.isVisible()
                } == expected_axes
            else:
                labels = [title]
            combo_center = combo.mapToGlobal(combo.rect().center())
            for label in labels:
                label_center = label.mapToGlobal(label.rect().center())
                assert abs(label_center.y() - combo_center.y()) <= 1

            combo.showPopup()
            QTest.qWait(250)
            popup = combo.view().window()
            assert popup.isVisible()
            for row in range(combo.count()):
                item_rect = combo.view().visualRect(combo.model().index(row, 0))
                assert combo.view().viewport().rect().contains(item_rect)
            below = combo.mapToGlobal(QPoint(0, combo.height()))
            assert abs(popup.pos().y() - below.y()) <= 1
            assert abs(popup.pos().x() - below.x()) <= 1
            popup_positions.append((popup.pos().x(), popup.pos().y()))

            QTest.keyClick(combo.view(), Qt.Key.Key_Escape)
            application.processEvents()
            assert not popup.isVisible()
            assert combo.currentIndex() == index
        assert len(set(popup_positions)) == 1
    finally:
        combo.hidePopup()
        page.close()
        application.setStyleSheet(previous_style)
