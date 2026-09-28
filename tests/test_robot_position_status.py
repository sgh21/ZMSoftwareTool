"""定位页时间、设置状态及逐点超限显示的回归测试。"""

from copy import deepcopy
from datetime import datetime
import json

import numpy as np
import pytest
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import QApplication

from app.pages.robot_position_page import PointEditor, RobotPositionPage
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

    def list_history(self, baseline_id=None):
        return [item for item in self.history
                if baseline_id is None or item["baseline_id"] == baseline_id]

    def save_settings(self, settings):
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
    widget = RobotPositionPage(MemoryService())
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


def test_time_labels_show_full_local_timestamps_and_keep_history_reference(page, evaluated):
    assert page.evaluation_time_label.text() == "评价更新时间：—"
    assert page.baseline_time_label.text() == f"基准建立时间：{local_time(BASELINE_TIME)}"
    page.service.baseline.update({"id": "other", "created_at": "2026-09-27T10:00:00+00:00"})
    page._evaluation_completed(evaluated)
    for index in range(page.result_metric.count()):
        page.result_metric.setCurrentIndex(index)
        assert page.evaluation_time_label.text() == f"评价更新时间：{local_time(evaluated['created_at'])}"
        assert page.baseline_time_label.text() == f"基准建立时间：{local_time(BASELINE_TIME)}"


def test_trend_uses_elapsed_days_and_sorts_irregular_intervals_across_timezones(page, evaluated):
    page.service.history = [
        {**evaluated, "id": "late", "created_at": "2026-09-25T00:00:00+00:00"},
        {**evaluated, "id": "early", "created_at": "2026-09-20T12:00:00+00:00"},
        evaluated,
        {**evaluated, "id": "wrong-baseline", "baseline_id": "other"},
        {**evaluated, "id": "wrong-parameters", "parameter_version": "other"},
    ]
    page._evaluation_completed(evaluated)
    assert [record["days"] for record in page.trend_chart.history] == pytest.approx([0.5, 2, 5])
    assert all(record["X"] == pytest.approx(np.sqrt(2)) for record in page.trend_chart.history)


@pytest.mark.parametrize("saved_baseline_id", ["b1", "unrelated", None])
def test_old_history_reads_only_its_matching_saved_baseline_time(page, evaluated, tmp_path, saved_baseline_id):
    evaluated.pop("baseline_created_at")
    baseline_path = tmp_path / "historical_baseline.json"
    evaluated["baseline_path"] = str(baseline_path)
    if saved_baseline_id is not None:
        baseline_path.write_text(json.dumps({
            "id": saved_baseline_id, "created_at": BASELINE_TIME,
        }), encoding="utf-8")
    page.service.baseline.update({"id": "current-baseline", "created_at": "2026-09-27T00:00:00+00:00"})
    page.service.history = [evaluated]
    page._evaluation_completed(evaluated)
    if saved_baseline_id == "b1":
        assert page.baseline_time_label.text() == f"基准建立时间：{local_time(BASELINE_TIME)}"
        assert page.trend_chart.history[0]["days"] == pytest.approx(2)
    else:
        assert page.baseline_time_label.text() == "基准建立时间：—"
        assert page.trend_chart.history == []
        assert "基准" in page.trend_chart.empty_message


def test_standalone_result_keeps_values_without_inventing_a_baseline_time(page, evaluated):
    evaluated.update({"baseline_id": None, "baseline_created_at": None})
    page.service.history = [evaluated]
    page._evaluation_completed(evaluated)
    assert page.baseline_time_label.text() == "基准建立时间：—"
    assert float(page.axis_values["X"].text()) == pytest.approx(np.sqrt(2), abs=0.0001)
    assert page.trend_chart.history == []
    assert "基准" in page.trend_chart.empty_message


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


def test_one_point_over_limit_marks_axis_red_even_when_average_passes(page, evaluated, application):
    original_color = page.axis_values["X"].palette().color(QPalette.ColorRole.WindowText)
    page._evaluation_completed(evaluated)
    application.processEvents()
    assert float(page.axis_values["X"].text()) < evaluated["metric_thresholds"]["repeatability"]["X"]
    assert page.axis_cards["X"].property("overLimit") is True
    assert all(page.axis_cards[axis].property("overLimit") is False for axis in ("Y", "Z", "distance"))
    assert page.axis_values["X"].palette().color(QPalette.ColorRole.WindowText) == QColor(DISPLAY["colors"]["error"])
    page.result_metric.setCurrentIndex(0)
    application.processEvents()
    assert all(card.property("overLimit") is False for card in page.axis_cards.values())
    assert page.axis_values["X"].palette().color(QPalette.ColorRole.WindowText) == QColor(DISPLAY["colors"]["action"])
    page.result_metric.setCurrentIndex(2)
    assert page.axis_cards["X"].property("overLimit") is True
    page._invalidate_result()
    application.processEvents()
    assert all(card.property("overLimit") is False for card in page.axis_cards.values())
    assert page.axis_values["X"].palette().color(QPalette.ColorRole.WindowText) == original_color
