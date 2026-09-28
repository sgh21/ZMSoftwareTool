"""用独立临时目录核对启动恢复、每日趋势和日志；不改主运行状态。"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtGui import QImage
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from app.pages.robot_position_page import RobotPositionPage
from core.services.position_monitoring_service import PositionMonitoringService, read_document, write_document


@pytest.fixture(scope="module")
def application():
    return QApplication.instance() or QApplication([])


def import_batch(service, root, batch_id, radius, *, simulation=True, captured_at=None):
    samples = []
    for index, sign in enumerate((-1, 1), 1):
        pose = np.eye(4)
        pose[0, 3] = -sign * radius
        image_path = root / f"{batch_id}_P001_D{index:03d}.png"
        image = QImage(8, 8, QImage.Format.Format_RGB32)
        image.fill(Qt.GlobalColor.white)
        assert image.save(str(image_path))
        samples.append({
            "point_id": "P001", "direction_id": f"D{index:03d}",
            "sample_id": image_path.stem, "vision_pose": pose.tolist(),
            "ideal_pose": np.eye(4).tolist(), "image_path": str(image_path),
        })
        if captured_at:
            samples[-1]["captured_at"] = captured_at
    path = root / f"{batch_id}.json"
    write_document(path, {
        "batch_id": batch_id, "length_unit": "mm", "program_id": "fixed", "target_id": "fixed-board",
        "sampling_protocol": "multidirectional",
        "comparison_status": "simulation" if simulation else "observed", "samples": samples,
    })
    service.load_observations(path)


def test_restart_restores_last_result_images_three_days_and_logs(application, tmp_path):
    service = PositionMonitoringService(tmp_path / "application")
    service.save_parameters({"hand_eye": np.eye(4).tolist(), "target_pose_base": np.eye(4).tolist()})
    import_batch(service, tmp_path, "B001", 0.2)
    baseline = service.create_baseline()
    import_batch(service, tmp_path, "B002", 0.4)
    service.evaluate()
    import_batch(service, tmp_path, "B003", 0.6)
    result = service.evaluate()
    page = RobotPositionPage(service)
    page.append_log("本轮已完成，可重启查看")
    before_logs = deepcopy(service.list_logs())
    page.close()

    restored = PositionMonitoringService(service.root)
    reopened = RobotPositionPage(restored)
    try:
        assert reopened.result["id"] == result["id"]
        assert reopened.task_progress.value() == 100
        assert reopened.task_progress_note.text() == "已恢复 B003 评估 · 2 个观测"
        assert restored.baseline["id"] == baseline["id"]
        assert reopened.axis_values["distance"].text() == "0.6000"
        assert [row["days"] for row in reopened.trend_chart.history] == [0, 1, 2]
        for mode in ("absolute_change", "repeatability_change", "repeatability"):
            reopened.result_metric.setCurrentIndex(reopened.result_metric.findData(mode))
            assert [row["days"] for row in reopened.trend_chart.history] == [0, 1, 2]
        assert [reopened.result_metric.itemText(index) for index in range(3)] == [
            "绝对定位精度退化", "重复定位精度退化", "当前重复定位精度",
        ]
        assert reopened.observation_sample.count() == 2
        assert "B003" in reopened.observation_sample.currentText()
        assert not reopened.observation_image.pixmap.isNull()
        reopened.observation_source.setCurrentIndex(1)
        assert "B001" in reopened.observation_sample.currentText()
        assert not reopened.observation_image.pixmap.isNull()
        assert "本轮已完成，可重启查看" in reopened.process_log.toPlainText()
        assert restored.list_logs() == before_logs
    finally:
        reopened.close()


def test_restart_keeps_unevaluated_import_and_does_not_show_previous_result(application, tmp_path):
    service = PositionMonitoringService(tmp_path / "application")
    service.save_parameters({"hand_eye": np.eye(4).tolist(), "target_pose_base": np.eye(4).tolist()})
    import_batch(service, tmp_path, "B001", 0.2)
    service.create_baseline()
    import_batch(service, tmp_path, "B002", 0.4)
    service.evaluate()
    import_batch(service, tmp_path, "B003", 0.6)
    reopened = RobotPositionPage(PositionMonitoringService(service.root))
    try:
        assert reopened.result is None
        assert reopened.task_progress_note.text() == "已恢复 B003 观测，待评估"
        assert "B003" in reopened.observation_sample.currentText()
        assert reopened.axis_values["distance"].text() == "—"
        assert [row["days"] for row in reopened.trend_chart.history] == [0, 1]
    finally:
        reopened.close()


def test_restored_logs_scroll_to_latest_after_main_window_layout_and_preserve_user_scroll(application, tmp_path, monkeypatch):
    import app.main_window as window_module

    service = PositionMonitoringService(tmp_path / "application")
    for index in range(50):
        service.append_log(f"历史记录 {index} · " + "较长的观测保存路径/" * 12)
    service.append_log("最后一条：评估完成")
    before_logs = deepcopy(service.list_logs())
    monkeypatch.setattr(window_module, "RobotPositionPage", lambda: RobotPositionPage(service))
    previous_style = application.styleSheet()
    window = window_module.MainWindow()
    try:
        window.resize(1600, 960)
        window.show()
        QTest.qWait(350)
        page = window.precision_page.content_stack.widget(0)
        log = page.process_log
        scrollbar = log.verticalScrollBar()
        assert log.textCursor().atEnd()
        assert scrollbar.maximum() > 0
        assert scrollbar.value() == scrollbar.maximum()
        assert service.list_logs() == before_logs
        scrollbar.setValue(0)
        window.resize(1280, 800)
        QTest.qWait(250)
        assert scrollbar.value() < scrollbar.maximum()
    finally:
        window.close()
        application.setStyleSheet(previous_style)


@pytest.mark.parametrize("simulation", [True, False])
def test_legacy_history_and_new_batch_share_complete_trend_without_hiding_fallback(application, tmp_path, simulation):
    service = PositionMonitoringService(tmp_path / "application")
    service.save_parameters({"hand_eye": np.eye(4).tolist(), "target_pose_base": np.eye(4).tolist()})
    import_batch(service, tmp_path, "B001", 0.2, simulation=simulation, captured_at="2026-09-20T00:00:00+00:00")
    service.create_baseline()
    import_batch(service, tmp_path, "B002", 0.4, simulation=simulation)
    old = service.evaluate()
    daily_path = Path(service._latest_result_ref["path"])
    daily = read_document(daily_path)
    daily["evaluations"] = [row for row in daily["evaluations"] if row["id"] != old["id"]]
    write_document(daily_path, daily)
    old["created_at"] = "2026-09-21T00:00:00+00:00"
    for field in ("observed_at", "time_source", "debug_day_index"):
        old.pop(field, None)
        old.pop(f"baseline_{field}", None)
    snapshot = read_document(old["current_batch_path"])
    for field in ("observed_at", "time_source", "debug_day_index", "imported_at"):
        snapshot.pop(field, None)
    write_document(old["current_batch_path"], snapshot)
    legacy_path = service.storage / "history" / f"{old['id']}.json"
    write_document(legacy_path, old)
    old_bytes = legacy_path.read_bytes()
    import_batch(service, tmp_path, "B003", 0.6, simulation=simulation, captured_at="2026-09-22T00:00:00+00:00")
    service.evaluate()
    reopened = RobotPositionPage(PositionMonitoringService(service.root))
    try:
        assert [row["days"] for row in reopened.trend_chart.history] == [0, 1, 2]
        assert [row["distance"] for row in reopened.trend_chart.history] == pytest.approx([0.2, 0.4, 0.6])
        if simulation:
            assert reopened.trend_time_caption.text() == "调试天数"
        else:
            assert reopened.trend_time_caption.text() == "观测时间 / 天"
            assert "旧记录缺采集和导入时间" in reopened.trend_time_caption.toolTip()
        assert legacy_path.read_bytes() == old_bytes
    finally:
        reopened.close()
