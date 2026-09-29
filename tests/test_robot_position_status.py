"""定位页时间、设置状态及逐点超限显示的回归测试。"""

from copy import deepcopy
from datetime import datetime

import numpy as np
import pytest
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

from app.pages.robot_position_page import PointEditor
from ui_helpers import ready_position_page, refresh_page
from app.resources import DISPLAY, load_stylesheet
from core.algorithms.position_monitoring import evaluate_position_monitoring
from core.services.position_monitoring_service import METRIC_AXES, METRIC_LABELS


BASELINE_TIME = "2026-09-20T00:00:00+00:00"


class MemoryService:
    def __init__(self):
        self.parameters = {"version": "p1", "hand_eye": np.eye(4).tolist()}
        self.baseline = {
            "id": "b1", "label": "初始基准", "created_at": BASELINE_TIME,
            "parameters": deepcopy(self.parameters), "batch": {"samples": []},
        }
        self.settings = {"metric_thresholds": {}, "processing_points": []}
        self.current_batch = None
        self.history = []
        self.latest_result = None
        self.latest_batch = None
        self.latest_comparison_error = ""
        self.history_comparison_warnings = []
        self.logs = []

    def list_baselines(self, progress=None):
        return []

    def list_logs(self):
        return self.logs

    def compare_latest(self):
        return self.latest_result

    def history_comparisons(self, include_before=False, progress=None):
        return self.history

    def append_log(self, message, level="INFO"):
        entry = {"timestamp": BASELINE_TIME, "level": level, "message": message}
        self.logs.append(entry)
        return entry

    def list_history(self, baseline_id=None):
        return [item for item in self.history
                if baseline_id is None or item["baseline_id"] == baseline_id]

    def save_settings(self, settings, progress=None):
        self.settings.update(deepcopy(settings))


@pytest.fixture(scope="module")
def application():
    app = QApplication.instance() or QApplication([])
    previous_style = app.styleSheet()
    app.setStyleSheet(load_stylesheet())
    yield app
    app.setStyleSheet(previous_style)


@pytest.fixture
def page(application):
    widget = ready_position_page(MemoryService())
    widget.ensurePolished()
    yield widget
    widget.close()


@pytest.fixture
def evaluated():
    initial, current = [], []
    for point in ("P1", "P2", "P3"):
        for offset in (-1, 1):
            before = np.eye(4)
            after = before.copy()
            after[0, 3] = offset if point == "P1" else 0
            common = {"point_id": point, "direction_id": "D1"}
            initial.append({**common, "vision_pose": before.tolist()})
            current.append({**common, "vision_pose": after.tolist()})
    result = evaluate_position_monitoring(
        initial, current, np.eye(4),
        base_rotations={point: np.eye(3) for point in ("P1", "P2", "P3")},
        initial_errors={point: {"D1": [1, 0, 0]} for point in ("P1", "P2", "P3")},
    )
    return {
        **result, "id": "r1", "batch_id": "current", "baseline_id": "b1",
        "baseline_created_at": BASELINE_TIME, "parameter_version": "p1",
        "created_at": "2026-09-22T08:00:00+08:00", "status": "超限",
        "metric_thresholds": {"repeatability": dict.fromkeys(METRIC_AXES, 3)},
    }


def local_time(value):
    return datetime.fromisoformat(value).astimezone().strftime("%Y-%m-%d %H:%M:%S")


def test_time_labels_show_full_local_timestamps_and_current_selected_baseline(page, evaluated):
    assert page.evaluation_time_label.text() == "评价更新时间：—"
    assert page.baseline_time_label.text() == f"基准建立时间：{local_time(BASELINE_TIME)}"
    page.service.baseline.update({"id": "other", "created_at": "2026-09-27T10:00:00+00:00"})
    page._evaluation_completed(evaluated)
    for index in range(page.result_metric.count()):
        page.result_metric.setCurrentIndex(index)
        assert page.evaluation_time_label.text() == f"评价更新时间：{local_time(evaluated['created_at'])}"
        assert page.baseline_time_label.text() == f"基准建立时间：{local_time(page.service.baseline['created_at'])}"


def test_trend_uses_elapsed_days_and_sorts_irregular_intervals_across_timezones(page, evaluated):
    page.service.history = [
        {**evaluated, "id": "late", "created_at": "2026-09-25T00:00:00+00:00"},
        {**evaluated, "id": "early", "created_at": "2026-09-20T12:00:00+00:00"},
        evaluated,
    ]
    refresh_page(page)
    page._evaluation_completed(evaluated)
    assert [record["days"] for record in page.trend_chart.history] == pytest.approx([0.5, 2, 5])
    assert all(record["X"] == pytest.approx(np.sqrt(2)) for record in page.trend_chart.history)


def test_standalone_result_keeps_values_without_inventing_a_baseline_time(page, evaluated):
    evaluated.update({"baseline_id": None, "baseline_created_at": None})
    page.service.baseline = None
    page.service.history = [evaluated]
    refresh_page(page)
    page._evaluation_completed(evaluated)
    assert page.baseline_time_label.text() == "基准建立时间：—"
    assert float(page.axis_values["X"].text()) == pytest.approx(np.sqrt(2), abs=0.0001)
    assert [row["days"] for row in page.trend_chart.history] == [0]
    assert page.trend_chart.baseline_day is None


def test_debug_trend_uses_batch_days_and_latest_evaluation_per_batch(page, evaluated):
    page.service.baseline["batch"]["debug_day_index"] = 0
    records = []
    for day in range(3):
        records.append({
            **deepcopy(evaluated), "id": f"r{day}", "batch_id": f"B{day + 1:03d}",
            "debug_day_index": day, "baseline_debug_day_index": 0,
            "created_at": f"2026-09-28T00:00:0{day}+00:00",
        })
    repeated = {**deepcopy(records[1]), "id": "repeat-B002", "created_at": "2026-09-29T00:00:00+00:00"}
    repeated["summary"]["rp_current"] = 0.123
    page.service.history = [repeated, records[2], records[0], records[1]]
    refresh_page(page)
    page._evaluation_completed(records[2])
    assert [record["days"] for record in page.trend_chart.history] == [0, 1, 2]
    assert page.trend_chart.history[1]["distance"] == pytest.approx(0.123)
    assert page.trend_time_caption.text() == "调试天数"
    assert "B003" in page.evaluation_time_label.toolTip()


def test_observed_time_is_separate_from_evaluation_time_and_explains_fallback(page, evaluated):
    page.service.baseline["batch"]["observed_at"] = BASELINE_TIME
    evaluated.update({
        "observed_at": "2026-09-21T00:00:00+00:00", "baseline_observed_at": BASELINE_TIME,
        "time_source": "captured_at", "baseline_time_source": "captured_at",
    })
    page.service.history = [evaluated]
    refresh_page(page)
    page._evaluation_completed(evaluated)
    assert page.trend_chart.history[0]["days"] == 1
    assert page.trend_time_caption.text() == "采集时间 / 天"
    assert page.evaluation_time_label.text() == f"评价更新时间：{local_time(evaluated['created_at'])}"
    evaluated["time_source"] = "imported_at"
    page._render_result()
    assert page.trend_time_caption.text() == "观测时间 / 天"
    assert "导入时间（缺采集时间）" in page.trend_time_caption.toolTip()


def test_image_progress_updates_one_line_without_filling_saved_logs(page):
    before = list(page.service.logs)
    for image in range(1, 601):
        page._task_progress(round(image / 6), f"已读取 {image}/600")
    assert page.task_progress.value() == 99
    assert page.task_progress_note.text() == "已读取 600/600"
    assert page.service.logs == before
    assert "已读取" not in page.process_log.toPlainText()


def test_settings_lights_follow_configuration_and_point_editor_save(page):
    service = page.service
    service.parameters = {}
    service.baseline = None
    page._refresh_settings()
    assert {key: light.property("state") for key, light in page.status_lights.items()} == {
        "parameter": "missing", "baseline": "missing", "conditions": "missing",
    }
    service.parameters = {"version": "p2", "camera_matrix": np.eye(3).tolist()}
    service.baseline = {"id": "b1", "created_at": BASELINE_TIME,
                        "parameters": {"version": "p1"}, "batch": {"samples": []}}
    service.settings["metric_thresholds"] = {"repeatability": {"X": 0}}
    page._refresh_settings()
    assert all(light.property("state") == "partial" for light in page.status_lights.values())
    service.parameters["hand_eye"] = np.eye(4).tolist()
    service.baseline["parameters"]["version"] = "p2"
    service.settings["metric_thresholds"] = {
        mode: dict.fromkeys(METRIC_AXES, 0) for mode in METRIC_LABELS
    }
    page._refresh_settings()
    assert page.status_lights["parameter"].property("state") == "ready"
    assert page.status_lights["baseline"].property("state") == "ready"
    assert page.status_lights["conditions"].property("state") == "partial"
    editor = PointEditor(page)
    editor._add_point()
    for column in (1, 2, 3):
        editor.table.item(0, column).setText("0")
    editor._apply()
    assert page.status_lights["conditions"].property("state") == "ready"
    assert len(service.settings["processing_points"]) == 1
    assert all(light.toolTip() for light in page.status_lights.values())


def test_one_point_alarm_does_not_mark_passing_summary_red(page, evaluated, application):
    original_color = page.axis_values["X"].palette().color(QPalette.ColorRole.WindowText)
    page.service.settings["metric_thresholds"] = deepcopy(evaluated["metric_thresholds"])
    page._evaluation_completed(evaluated)
    application.processEvents()
    assert float(page.axis_values["X"].text()) < evaluated["metric_thresholds"]["repeatability"]["X"]
    assert page.axis_cards["X"].property("overLimit") is False
    assert all(page.axis_cards[axis].property("overLimit") is False for axis in ("Y", "Z", "distance"))
    assert "1 项超限" in page.alarm_status.text()
    assert "逐点超限" in page.axis_cards["X"].toolTip()
    page.result_metric.setCurrentIndex(0)
    application.processEvents()
    assert all(card.property("overLimit") is False for card in page.axis_cards.values())
    assert page.axis_values["X"].palette().color(QPalette.ColorRole.WindowText) == QColor(DISPLAY["colors"]["action"])
    page.result_metric.setCurrentIndex(2)
    assert page.axis_cards["X"].property("overLimit") is False
    page.service.latest_result = None
    page._refresh_latest_result()
    application.processEvents()
    assert all(card.property("overLimit") is False for card in page.axis_cards.values())
    assert page.axis_values["X"].palette().color(QPalette.ColorRole.WindowText) == original_color


def test_cards_compare_each_displayed_value_to_its_own_current_threshold(page, evaluated):
    evaluated["summary"].update({"axis_3sigma_base": [0.1152, 0.1251, 0.1384], "rp_current": 0.253})
    page.service.settings["metric_thresholds"] = {
        "repeatability": {"X": 0.15, "Y": 0.15, "Z": 0.15, "distance": 0.2},
    }
    page._evaluation_completed(evaluated)
    assert [page.axis_values[key].text() for key in METRIC_AXES] == ["0.1152", "0.1251", "0.1384", "0.2530"]
    assert [page.axis_cards[key].property("overLimit") for key in METRIC_AXES] == [False, False, False, True]
