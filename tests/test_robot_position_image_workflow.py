"""图像观测接入后的页面回归；创建隐藏控件，不启动窗口或后台任务。"""

from copy import deepcopy

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication, QFileDialog

from app.dialogs.robot_position_parameters_dialog import RobotPositionParametersDialog
from app.pages.robot_position_page import RobotPositionPage
from core.services.position_monitoring_service import PositionMonitoringService, assess_metric, write_document


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def service(tmp_path):
    service = PositionMonitoringService(tmp_path)
    service.save_parameters({
        "hand_eye": np.eye(4).tolist(),
        "target_pose_base": np.eye(4).tolist(),
        "camera_matrix": [[1000, 0, 320], [0, 1000, 240], [0, 0, 1]],
    })
    return service


def batch(service, positions, *, baseline=False, multidirectional=True):
    """用已完成视觉解算的观测测试页面，PnP 本身由算法测试覆盖。"""
    samples = []
    for index, position in enumerate(positions, start=1):
        vision = np.eye(4)
        vision[0, 3] = -position
        samples.append({
            "point_id": "P001",
            "direction_id": f"D{index:03d}" if multidirectional else "D001",
            "sample_id": str(index),
            "vision_pose": vision.tolist(),
            "ideal_pose": np.eye(4).tolist(),
        })
    identifier = "B000" if baseline else "B001"
    result = {
        "batch_id": identifier, "label": identifier,
        "length_unit": "mm",
        "program_id": "fixed-points", "target_id": "board-1",
        "parameter_version": service.parameters["version"],
        "source_path": str(service.root / f"{identifier}.json"),
        "base_rotations": {"P001": np.eye(3).tolist()},
        "samples": samples,
    }
    if multidirectional:
        result["sampling_protocol"] = "multidirectional"
    path = service.root / f"{identifier}.json"
    write_document(path, result)
    return service.load_observations(path)


def compare(service, positions=(-0.2, 0.2)):
    service.current_batch = batch(service, [0] * len(positions), baseline=True)
    service.create_baseline("初始图像观测")
    service.current_batch = batch(service, positions)
    return service.evaluate()


@pytest.mark.parametrize("entry", ["parameter_dialog", "parameter_file"])
def test_saving_unchanged_parameters_preserves_observations_and_display(
    application, service, monkeypatch, entry,
):
    result = compare(service)
    page = RobotPositionPage(service)
    page.ui_scale = 1.0
    page._evaluation_completed(result)
    current = service.current_batch
    baseline = service.baseline
    version = service.parameters["version"]
    parameter_bytes = service.parameter_path.read_bytes()
    previous_bytes = service.previous_parameter_path.read_bytes()
    shown = page.axis_values["distance"].text()
    try:
        if entry == "parameter_dialog":
            def save_without_opening(dialog):
                dialog._save()
                assert not dialog.error_label.text()
                return dialog.result()

            monkeypatch.setattr(RobotPositionParametersDialog, "exec", save_without_opening)
            page._show_parameters()
        else:
            monkeypatch.setattr(
                QFileDialog, "getOpenFileName",
                lambda *args: (str(service.parameter_path), ""),
            )
            page._load_parameters()
        assert service.parameters["version"] == version
        assert service.parameter_path.read_bytes() == parameter_bytes
        assert service.previous_parameter_path.read_bytes() == previous_bytes
        assert service.current_batch is current
        assert service.baseline is baseline
        assert page.result is result
        assert page.axis_values["distance"].text() == shown == "0.2000"
        assert page.observation_sample.count() == 2
        assert page.measured_table.rowCount() == 1
    finally:
        page.close()


def test_multidirectional_display_and_history_match_actual_point_direction_pairs(
    application, service, monkeypatch,
):
    result = compare(service)
    result.update({
        "baseline_created_at": "2026-09-27T00:00:00+00:00",
        "created_at": "2026-09-28T00:00:00+00:00",
        "baseline_observed_at": "2026-09-27T00:00:00+00:00",
        "observed_at": "2026-09-28T00:00:00+00:00",
    })
    reordered = deepcopy(result)
    reordered["created_at"] = "2026-09-29T00:00:00+00:00"
    reordered["observed_at"] = "2026-09-29T00:00:00+00:00"
    reordered["groups"][0]["directions"].reverse()
    wrong_direction = deepcopy(result)
    wrong_direction["groups"][0]["directions"][1]["direction_id"] = "D003"
    wrong_point = deepcopy(result)
    wrong_point["groups"][0]["point_id"] = "P002"
    for direction in wrong_point["groups"][0]["directions"]:
        direction["point_id"] = "P002"
    other_protocol = {**result, "sampling_protocol": None}
    monkeypatch.setattr(
        service, "list_history",
        lambda: [result, reordered, wrong_direction, wrong_point, other_protocol],
    )
    page = RobotPositionPage(service)
    try:
        page._evaluation_completed(result)
        assert page.result_metric.currentText() == "当前重复定位精度"
        assert "非国标同方向RP" in page.result_hint.toolTip()
        assert page.measured_table.horizontalHeaderItem(5).text() == "接近方向数"
        assert page.measured_table.item(0, 0).text() == "P001 / 多方向"
        assert page.measured_table.item(0, 5).text() == "2"
        assert [record["days"] for record in page.trend_chart.history] == [1, 2]
        assert [record["distance"] for record in page.trend_chart.history] == pytest.approx([0.2, 0.2])
        page.result_metric.setCurrentIndex(1)
        assert page.result_metric.currentText() == "重复定位精度退化"
        assert page.measured_table.item(0, 5).text() == "2 → 2"
        assert [record["days"] for record in page.trend_chart.history] == [1, 2]
        assert page.trend_title.text() == "定位精度趋势"
    finally:
        page.close()


def test_multidirectional_thresholds_are_separate_and_history_keeps_saved_limits(
    application, service,
):
    service.save_settings({
        "metric_thresholds": {"repeatability": {"X": 0.1}},
        "multidirectional_thresholds": {"repeatability": {"X": 2.0}},
    })
    service.current_batch = batch(service, [-0.2, 0.2])
    multidirectional = service.evaluate_current()
    page = RobotPositionPage(service)
    try:
        page._evaluation_completed(multidirectional)
        assert multidirectional["metric_thresholds"]["repeatability"]["X"] == 2.0
        assert page._threshold_key() == "multidirectional_thresholds"
        assert page.axis_cards["X"].property("overLimit") is False
        assert "2.0000" in page.axis_cards["X"].toolTip()

        service.save_settings({"multidirectional_thresholds": {"repeatability": {"X": 0.3}}})
        page._render_result()
        assert page.axis_cards["X"].property("overLimit") is False
        assert "2.0000" in page.axis_cards["X"].toolTip()
        assert service.settings["metric_thresholds"]["repeatability"]["X"] == 0.1

        service.current_batch = batch(service, [-0.2, 0.2], multidirectional=False)
        repeated = service.evaluate_current()
        page._evaluation_completed(repeated)
        assert repeated["metric_thresholds"]["repeatability"]["X"] == 0.1
        assert page._threshold_key() == "metric_thresholds"
        assert page.result_metric.currentText() == "当前重复定位精度"
        assert page.measured_table.horizontalHeaderItem(5).text() == "到达次数"
        assert page.axis_cards["X"].property("overLimit") is True
        assert "0.1000" in page.axis_cards["X"].toolTip()
    finally:
        page.close()


def test_one_direction_absolute_degradation_alarm_survives_point_average(application, service):
    service.save_settings({
        "multidirectional_thresholds": {"absolute_change": {"X": 1.5, "distance": 1.5}},
    })
    result = compare(service, positions=[2.0, 0.0])
    assessment = assess_metric(result, "absolute_change")
    assert result["summary"]["absolute_ap_change"] == 1.0
    assert result["groups"][0]["absolute_axis_change"][0] == 1.0
    assert assessment["status"] == "超限"
    assert [(alarm["point_id"], alarm["direction_id"], alarm["axis"], alarm["value"])
            for alarm in assessment["alarms"]] == [
        ("P001", "D001", "X", 2.0), ("P001", "D001", "distance", 2.0),
    ]
    page = RobotPositionPage(service)
    try:
        page._evaluation_completed(result)
        page.result_metric.setCurrentIndex(0)
        assert page.axis_values["X"].text() == "1.0000"
        assert page.axis_cards["X"].property("overLimit") is True
        assert page.axis_cards["distance"].property("overLimit") is True
        assert page.axis_cards["Y"].property("overLimit") is False
        assert "P001/D001 X：2.0000 > 1.5000 mm" in page.alarm_message.text()
    finally:
        page.close()
