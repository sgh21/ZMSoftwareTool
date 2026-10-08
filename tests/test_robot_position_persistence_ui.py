"""用独立临时目录核对启动恢复、每日趋势和日志；不改主运行状态。"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PyQt6.QtGui import QImage
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QComboBox, QDialog, QFileDialog, QLabel, QLineEdit, QPushButton

from ui_helpers import ready_position_page, refresh_page, wait_for_page
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
    page = ready_position_page(service)
    page.append_log("本轮已完成，可重启查看")
    before_logs = deepcopy(service.list_logs())
    page.close()

    restored = PositionMonitoringService(service.root)
    reopened = ready_position_page(restored)
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


def test_restart_keeps_unevaluated_import_but_cards_show_latest_evaluated_batch(application, tmp_path):
    service = PositionMonitoringService(tmp_path / "application")
    service.save_parameters({"hand_eye": np.eye(4).tolist(), "target_pose_base": np.eye(4).tolist()})
    import_batch(service, tmp_path, "B001", 0.2)
    service.create_baseline()
    import_batch(service, tmp_path, "B002", 0.4)
    service.evaluate()
    import_batch(service, tmp_path, "B003", 0.6)
    reopened = ready_position_page(PositionMonitoringService(service.root))
    try:
        assert reopened.result["batch_id"] == "B002"
        assert reopened.task_progress_note.text() == "已恢复 B003 观测；主卡显示 B002 最新评估"
        assert "B003" in reopened.observation_sample.currentText()
        assert reopened.axis_values["distance"].text() == "0.4000"
        assert [row["days"] for row in reopened.trend_chart.history] == [0, 1]
    finally:
        reopened.close()


def test_baseline_selection_moves_zero_day_but_never_changes_latest_cards(application, tmp_path, monkeypatch):
    service = PositionMonitoringService(tmp_path / "application")
    service.save_parameters({"hand_eye": np.eye(4).tolist(), "target_pose_base": np.eye(4).tolist()})
    import_batch(service, tmp_path, "B001", 0.2)
    first = service.create_baseline()
    import_batch(service, tmp_path, "B002", 0.4)
    service.evaluate()
    import_batch(service, tmp_path, "B003", 0.6)
    service.evaluate()
    page = ready_position_page(service)
    try:
        assert not page.show_before_baseline.isChecked()
        page.ui_scale = 1.0
        last = service.create_baseline()
        refresh_page(page)
        page._baseline_created(last)
        page.result_metric.setCurrentIndex(1)
        assert page.result["batch_id"] == "B003"
        assert page.axis_values["distance"].text() == "0.0000"
        assert [row["days"] for row in page.trend_chart.history] == [0]
        assert page.trend_chart.baseline_day == 0
        saved = deepcopy(service.list_history())
        page.show_before_baseline.setChecked(True)
        wait_for_page(page)
        assert [row["days"] for row in page.trend_chart.history] == [-2, -1, 0]
        assert [row["distance"] for row in page.trend_chart.history] == pytest.approx([-0.4, -0.2, 0])
        assert not page.axis_cards["distance"].property("overLimit")

        def choose_first(dialog):
            choice = dialog.findChild(QComboBox)
            assert choice.count() == 2
            assert choice.currentData() == last["path"]
            choice.setCurrentIndex(choice.findData(first["path"]))
            next(button for button in dialog.findChildren(QPushButton) if button.text() == "选择").click()
            return dialog.result()

        monkeypatch.setattr(QDialog, "exec", choose_first)
        page._select_baseline()
        wait_for_page(page)
        assert page.result["batch_id"] == "B003"
        assert page.result["baseline_id"] == first["id"]
        assert page.axis_values["distance"].text() == "0.4000"
        assert page.trend_chart.baseline_day == 0
        assert service.list_history() == saved

        older = service.load_observations(first["batch"]["saved_path"])
        page._observations_loaded(older)
        page._evaluation_completed(service.evaluate())
        assert page.result["batch_id"] == "B003"
        assert "B001" in page.observation_sample.currentText()
        assert "主卡仍显示最新已评估观测 B003" in page.process_log.toPlainText()
        reopened = ready_position_page(PositionMonitoringService(service.root))
        try:
            assert reopened.result["batch_id"] == "B003"
            assert reopened.result["baseline_id"] == first["id"]
            assert reopened.show_before_baseline.isChecked()
            assert [row["days"] for row in reopened.trend_chart.history] == [0, 1, 2]
        finally:
            reopened.close()
    finally:
        page.close()


@pytest.mark.parametrize("simulation", [True, False])
def test_middle_baseline_starts_at_zero_and_earlier_reevaluation_keeps_latest_batch(application, tmp_path, simulation):
    service = PositionMonitoringService(tmp_path / "application")
    service.save_parameters({"hand_eye": np.eye(4).tolist(), "target_pose_base": np.eye(4).tolist()})
    service.save_settings({"multidirectional_thresholds": {"repeatability": {"distance": 0.1}}})
    capture_times = [
        "2026-09-20T00:00:00+00:00", "2026-09-20T12:00:00+00:00",
        "2026-09-21T18:00:00+00:00", "2026-09-23T00:00:00+00:00",
        "2026-09-25T06:00:00+00:00", "2026-09-28T00:00:00+00:00",
    ]
    baselines = []
    for index, captured_at in enumerate(capture_times, 3):
        import_batch(service, tmp_path, f"B{index:03d}", 0.2 * (index - 2),
                     simulation=simulation, captured_at=captured_at)
        if index in (3, 4):
            baselines.append(service.create_baseline())
        else:
            service.evaluate()
    page = ready_position_page(service)
    expected_days = [0, 1, 2, 3, 4] if simulation else [0, 1.25, 2.5, 4.75, 7.5]
    try:
        assert page.result["batch_id"] == "B008"
        assert page.result["baseline_id"] == baselines[1]["id"]
        assert page.trend_chart.baseline_day == 0
        assert [row["days"] for row in page.trend_chart.history] == pytest.approx(expected_days)
        saved = deepcopy(service.list_history())
        page.show_before_baseline.setChecked(True)
        wait_for_page(page)
        assert [row["days"] for row in page.trend_chart.history] == pytest.approx(
            [-1 if simulation else -0.5, *expected_days]
        )
        assert service.list_history() == saved

        service.current_batch = deepcopy(baselines[0]["batch"])
        earlier = service.current_batch
        page._observations_loaded(earlier)
        page._evaluation_completed(service.evaluate())
        assert page.result["batch_id"] == "B008"
        assert page.axis_values["distance"].text() == "1.2000"
        messages = service.list_logs()
        assert any("批次 B003 已评估并保存；综合阈值判定：超限。"
                   "主卡仍显示最新已评估观测 B008。" in entry["message"] for entry in messages)
        assert not any(entry["level"] == "WARN" for entry in messages)
        reopened = ready_position_page(PositionMonitoringService(service.root))
        try:
            assert reopened.result["batch_id"] == "B008"
            assert reopened.trend_chart.baseline_day == 0
            assert reopened.show_before_baseline.isChecked()
            assert [row["days"] for row in reopened.trend_chart.history] == pytest.approx(
                [-1 if simulation else -0.5, *expected_days]
            )
        finally:
            reopened.close()
    finally:
        page.close()


def test_current_threshold_settings_update_details_export_and_preserve_history(application, tmp_path, monkeypatch):
    service = PositionMonitoringService(tmp_path / "application")
    service.save_parameters({"hand_eye": np.eye(4).tolist(), "target_pose_base": np.eye(4).tolist()})
    import_batch(service, tmp_path, "B001", 0.2)
    service.create_baseline()
    import_batch(service, tmp_path, "B003", 0.6)
    service.evaluate()
    saved = deepcopy(service.list_history())
    page = ready_position_page(service)
    page.ui_scale = 1.0
    try:
        def set_threshold(dialog):
            dialog.findChild(QLineEdit, "threshold_repeatability_distance").setText("0.5")
            next(button for button in dialog.findChildren(QPushButton) if button.text() == "保存").click()
            return dialog.result()

        monkeypatch.setattr(QDialog, "exec", set_threshold)
        page._show_settings()
        wait_for_page(page)
        assert page.axis_cards["distance"].property("overLimit")
        assert page.result["metric_thresholds"]["repeatability"]["distance"] == 0.5
        exported = tmp_path / "live_comparison.json"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(exported), ""))

        def export_details(dialog):
            assert any("当前阈值" in label.text() and "空间 0.5000" in label.text()
                       for label in dialog.findChildren(QLabel))
            next(button for button in dialog.findChildren(QPushButton) if button.text() == "导出结果").click()
            dialog.accept()
            return dialog.result()

        monkeypatch.setattr(QDialog, "exec", export_details)
        page._open_result_details(page.result)
        assert read_document(exported)["metric_thresholds"]["repeatability"]["distance"] == 0.5
        assert service.list_history() == saved
    finally:
        page.close()


def test_parameter_change_keeps_latest_view_and_incompatible_baseline_has_visible_reason(application, tmp_path, monkeypatch):
    service = PositionMonitoringService(tmp_path / "application")
    service.save_parameters({"hand_eye": np.eye(4).tolist(), "target_pose_base": np.eye(4).tolist()})
    import_batch(service, tmp_path, "B001", 0.2)
    service.create_baseline()
    import_batch(service, tmp_path, "B003", 0.6)
    service.evaluate()
    page = ready_position_page(service)
    try:
        parameters = deepcopy(service.parameters)
        parameters["hand_eye"][0][3] = 10
        parameter_path = tmp_path / "changed_parameters.json"
        write_document(parameter_path, parameters)
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(parameter_path), ""))
        page._load_parameters()
        wait_for_page(page)
        assert page.result["batch_id"] == "B003"
        assert page.axis_values["distance"].text() == "0.6000"
        import_batch(service, tmp_path, "B002", 0.4)
        selected = service.create_baseline()
        refresh_page(page)
        page._baseline_created(selected)
        assert page.result["batch_id"] == "B003"
        assert page.axis_values["distance"].text() == "0.6000"
        assert page.result["comparison_error"] in page.result_hint.text()
        assert selected["id"] in page.baseline_time_label.toolTip()
        assert page.trend_chart.baseline_day == 0
        page.result_metric.setCurrentIndex(0)
        assert page.axis_values["distance"].text() == "—"
        assert page.result["comparison_error"] in page.alarm_message.text()
    finally:
        page.close()


def test_restored_logs_scroll_to_latest_after_main_window_layout_and_preserve_user_scroll(application, tmp_path, monkeypatch):
    import app.main_window as window_module

    service = PositionMonitoringService(tmp_path / "application")
    for index in range(50):
        service.append_log(f"历史记录 {index} · " + "较长的观测保存路径/" * 12)
    service.append_log("最后一条：评估完成")
    before_logs = deepcopy(service.list_logs())
    monkeypatch.setattr(window_module, "RobotPositionPage", lambda: ready_position_page(service))
    previous_style = application.styleSheet()
    window = window_module.MainWindow()
    try:
        window.resize(1600, 960)
        window.show()
        page = window.precision_page.content_stack.widget(0)
        log = page.process_log
        scrollbar = log.verticalScrollBar()
        # 全套 Qt 测试共享 QApplication，首次全局换样式可能触发多轮布局。
        for _ in range(20):
            QTest.qWait(100)
            if (not window.scale_timer.isActive() and log.textCursor().atEnd()
                    and scrollbar.maximum() > 0 and scrollbar.value() == scrollbar.maximum()):
                break
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
    reopened = ready_position_page(PositionMonitoringService(service.root))
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
