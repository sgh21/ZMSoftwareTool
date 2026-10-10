"""窝深调试、理论设置、误差报警与历史联动回归。"""

from copy import deepcopy
from pathlib import Path
from statistics import fmean
from threading import Event, get_ident

import pytest
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QComboBox, QDialog, QLabel, QLineEdit, QPushButton, QTableWidget

from app.pages import feed_depth_page as page_module
from app.pages.feed_depth_page import FeedDepthPage, FeedDepthTrendChart
from app.resources import DISPLAY, UiScale, load_stylesheet
from tests.ui_helpers import wait_for_page


pytestmark = pytest.mark.usefixtures("styled_application")


def click_debug_action(page, application, action_text):
    clicked = []

    def click_action():
        dialog = application.activeModalWidget()
        if isinstance(dialog, QDialog):
            button = next((item for item in dialog.findChildren(QPushButton)
                           if item.text() == action_text), None)
            if button is not None:
                clicked.append(button.text())
                button.click()
            else:
                dialog.reject()

    QTimer.singleShot(0, click_action)
    page.debug_button.click()
    assert clicked == [action_text]


def change_settings(page, application, *, mode="uniform", depth="1.500",
                    lower="-0.05", upper="0.05", action="应用设置"):
    inspected = {}

    def edit_dialog():
        dialog = application.activeModalWidget()
        try:
            inspected["title"] = dialog.windowTitle()
            selector = dialog.findChild(QComboBox)
            selector.setCurrentIndex(selector.findData(mode))
            for name, value in (
                ("feedDepthTheory", depth), ("feedDepthErrorLower", lower),
                ("feedDepthErrorUpper", upper),
            ):
                dialog.findChild(QLineEdit, name).setText(value)
            button = next(item for item in dialog.findChildren(QPushButton) if item.text() == action)
            button.click()
        except Exception as error:
            inspected["error"] = repr(error)
            if isinstance(dialog, QDialog):
                dialog.reject()

    QTimer.singleShot(0, edit_dialog)
    page.theory_button.click()
    assert inspected == {"title": "理论窝深与报警设置"}
    wait_for_page(page)
    application.processEvents()


def read_table_dialog(application, show_dialog):
    inspected = {}

    def inspect():
        dialog = application.activeModalWidget()
        try:
            table = dialog.findChild(QTableWidget)
            inspected["rows"] = [
                [(table.item(row, column).text(), table.item(row, column).foreground().color())
                 for column in range(table.columnCount())]
                for row in range(table.rowCount())
            ]
        except Exception as error:
            inspected["error"] = repr(error)
        finally:
            if isinstance(dialog, QDialog):
                dialog.accept()

    QTimer.singleShot(0, inspect)
    show_dialog()
    assert "error" not in inspected, inspected
    return inspected["rows"]


@pytest.fixture
def page(application, tmp_path):
    widget = FeedDepthPage(simulation_root=tmp_path / "simulation", settings_path=tmp_path / "settings.json")
    widget.ui_scale = 1.0
    widget.resize(1500, 1000)
    widget.show()
    application.processEvents()
    yield widget
    wait_for_page(widget, success=False)
    widget.close()


def test_simulation_runs_in_background_and_links_selected_batch(page, application, monkeypatch):
    main_thread = get_ident()
    worker_threads = []
    release_generation = Event()
    generate = page_module.generate_feed_depth_simulation

    def tracked_generate(*args, **kwargs):
        worker_threads.append(get_ident())
        release_generation.wait(5)
        return generate(*args, **kwargs)

    monkeypatch.setattr(page_module, "generate_feed_depth_simulation", tracked_generate)
    assert not page.findChildren(QComboBox) and not page.findChildren(QLineEdit)
    assert all("模拟" not in button.text() for button in page.findChildren(QPushButton))
    try:
        click_debug_action(page, application, "生成90天模拟数据")
        assert page.task is not None
        assert not any(control.isEnabled() for control in (
            page.choose_button, page.debug_button, page.theory_button,
        ))
    finally:
        release_generation.set()
    wait_for_page(page)
    application.processEvents()
    assert worker_threads and worker_threads[0] != main_thread
    assert len(page.simulation["history"]) == 90
    assert page.progress.value() == 100
    assert "模拟数据" in page.source_badge.text()

    chart = page.trend_charts["bias"]
    selected_index = 34
    assert len(chart.points) == len(page.trend_charts["spread"].points) == 90
    position = dict(chart.points)[selected_index].toPoint()
    QTest.mouseClick(chart, Qt.MouseButton.LeftButton, pos=position)
    assert page.selected_batch_index == selected_index
    assert all(chart.selected_index == selected_index for chart in page.trend_charts.values())
    batch_id = page.simulation["history"][selected_index]["batch_id"]
    batch_rows = [row for row in page.simulation["rows"] if row["batch_id"] == batch_id]
    assert page.result_values["count"].text() == str(len(batch_rows))
    assert page.result_values["mean"].text() == f"{fmean(row['actual_depth_mm'] for row in batch_rows):.3f}"

    history_dialog = {}

    def choose_history_batch():
        dialog = application.activeModalWidget()
        if isinstance(dialog, QDialog):
            table = dialog.findChild(QTableWidget)
            history_dialog["row_count"] = table.rowCount()
            history_dialog["column_count"] = table.columnCount()
            table.selectRow(11)
            button = next((item for item in dialog.findChildren(QPushButton)
                           if item.text() == "查看批次"), None)
            if button is not None:
                button.click()
            else:
                dialog.reject()

    QTimer.singleShot(0, choose_history_batch)
    page._show_history()
    application.processEvents()
    assert history_dialog == {"row_count": 90, "column_count": 6}
    assert page.selected_batch_index == 11
    assert all(len(chart.history) == len(chart.points) == 90 for chart in page.trend_charts.values())
    batch_id = page.simulation["history"][11]["batch_id"]
    batch_rows = [row for row in page.simulation["rows"] if row["batch_id"] == batch_id]
    assert page.result_values["mean"].text() == f"{fmean(row['actual_depth_mm'] for row in batch_rows):.3f}"
    details = {}

    def inspect_details():
        dialog = application.activeModalWidget()
        if isinstance(dialog, QDialog):
            try:
                table = dialog.findChild(QTableWidget)
                details["row_count"] = table.rowCount()
                details["first_depth"] = table.item(0, 3).text()
                details["source"] = table.item(0, 5).text()
                details["labels"] = " ".join(label.text() for label in dialog.findChildren(QLabel))
            finally:
                dialog.accept()

    QTimer.singleShot(0, inspect_details)
    page._show_hole_details()
    assert details["row_count"] == 30
    assert details["first_depth"] == f"{batch_rows[0]['actual_depth_mm']:.3f}"
    assert details["source"] == "模拟数据"
    assert batch_id in details["labels"] and "非真实测量" in details["labels"]


def test_exit_simulation_restores_settings_and_keeps_csv(page, application, tmp_path):
    change_settings(page, application, depth="1.7")
    page.selected_file = str(tmp_path / "selected-measurements.csv")
    click_debug_action(page, application, "生成90天模拟数据")
    wait_for_page(page)
    assert page.result_values["mean"].property("overLimit") is True
    csv_path = Path(page.simulation["csv_path"])
    csv_content = csv_path.read_bytes()
    click_debug_action(page, application, "退出调试模式")
    application.processEvents()
    assert page.simulation is None
    assert page.evaluation is None
    assert all(not chart.history and not chart.points for chart in page.trend_charts.values())
    assert all(page.result_values[key].text() == "—" for key in ("count", "mean", "variance"))
    assert page.result_values["theory"].text() == "1.700"
    assert page.depth_mode == "uniform" and page.uniform_depth == 1.7
    assert "1.7" in page.theory_button.toolTip()
    assert page.error_lower_mm == -0.05 and page.error_upper_mm == 0.05
    assert not page.result_values["mean"].property("overLimit")
    assert page.result_values["mean"].palette().color(QPalette.ColorRole.WindowText) != QColor(DISPLAY["colors"]["error"])
    assert page.choose_button.toolTip() == page.selected_file
    assert page.selected_batch_index == -1
    assert page.theory_button.isEnabled()
    assert page.import_button.isEnabled() and not page.calculate_button.isEnabled()
    assert page.calculate_button.text() == "评估精度"
    assert page.progress.value() == 0
    assert csv_path.read_bytes() == csv_content


def test_theory_and_error_settings_refresh_history_and_alarm_colors(page, application):
    click_debug_action(page, application, "生成90天模拟数据")
    wait_for_page(page)
    assert page.theory_button.isEnabled()
    source = deepcopy(page.simulation)
    csv_before = Path(source["csv_path"]).read_bytes()
    mean = page.result_values["mean"]
    normal_color = mean.palette().color(QPalette.ColorRole.WindowText)
    red = QColor(DISPLAY["colors"]["error"])
    original_mean = mean.text()
    original_variance = page.result_values["variance"].text()

    change_settings(page, application)
    assert mean.property("overLimit") is True
    assert mean.palette().color(QPalette.ColorRole.WindowText) == red
    detail_rows = read_table_dialog(application, page._show_hole_details)
    flagged = [row for row in detail_rows if row[5][0] == "模拟数据 · 超限"]
    normal = [row for row in detail_rows if row[5][0] == "模拟数据"]
    assert flagged and normal
    assert all(row[3][1] == row[4][1] == red for row in flagged)
    assert all(row[3][1] != red and row[4][1] != red for row in normal)
    history_rows = read_table_dialog(application, page._show_history)
    assert len(history_rows) == 90
    assert history_rows[-1][2][1] == history_rows[-1][3][1] == red
    assert history_rows[0][2][1] != red and history_rows[0][3][1] != red

    page._show_batch(0)
    application.processEvents()
    assert not mean.property("overLimit")
    assert mean.palette().color(QPalette.ColorRole.WindowText) == normal_color
    page._show_batch(89)
    change_settings(page, application, depth="1.400")
    assert page.selected_batch_index == 89
    assert page.result_values["theory"].text() == "1.400"
    assert mean.text() == original_mean
    assert page.result_values["variance"].text() == original_variance
    assert not mean.property("overLimit")
    assert mean.palette().color(QPalette.ColorRole.WindowText) == normal_color
    assert all(len(chart.history) == len(chart.points) == 90 for chart in page.trend_charts.values())
    assert [row["bias_mm"] for row in page.trend_charts["bias"].history] == pytest.approx(
        [row["bias_mm"] + 0.1 for row in source["history"]]
    )
    assert [row["stddev_mm"] for row in page.trend_charts["spread"].history] == pytest.approx(
        [row["stddev_mm"] for row in source["history"]]
    )

    change_settings(page, application, mode="per_hole", depth="1.400")
    assert page.result_values["theory"].text() == "1.500"
    assert mean.property("overLimit") is True
    assert page.simulation == source
    assert Path(source["csv_path"]).read_bytes() == csv_before

    change_settings(page, application, mode="per_hole", lower="", upper="")
    assert page.error_lower_mm is None and page.error_upper_mm is None
    assert not mean.property("overLimit")
    assert mean.palette().color(QPalette.ColorRole.WindowText) == normal_color
    assert all(not row["alarm"] for row in page.evaluation["rows"])
    assert all(not record["alarm"] for record in page.evaluation["history"])
    detail_rows = read_table_dialog(application, page._show_hole_details)
    assert all(row[5][0] == "模拟数据" for row in detail_rows)
    assert all(row[3][1] != red and row[4][1] != red for row in detail_rows)


def test_settings_persist_without_data_and_cancel_keeps_previous_values(page, application, tmp_path):
    assert page.error_lower_mm is None and page.error_upper_mm is None
    change_settings(page, application, depth="1.650", lower="-0.02", upper="0.03")
    settings_path = tmp_path / "settings.json"
    saved = settings_path.read_bytes()
    assert page.uniform_depth == 1.65
    assert page.error_lower_mm == -0.02 and page.error_upper_mm == 0.03
    assert page.simulation is None and page.evaluation is None
    assert all(page.result_values[key].text() == "—" for key in ("count", "mean", "variance"))
    assert all(not chart.history for chart in page.trend_charts.values())
    assert not page.result_values["mean"].property("overLimit")

    change_settings(page, application, depth="2.000", lower="-0.1", upper="0.1", action="取消")
    assert page.uniform_depth == 1.65
    assert page.error_lower_mm == -0.02 and page.error_upper_mm == 0.03
    assert settings_path.read_bytes() == saved

    reopened = FeedDepthPage(simulation_root=tmp_path / "reopened", settings_path=settings_path)
    try:
        assert reopened.depth_mode == "uniform" and reopened.uniform_depth == 1.65
        assert reopened.error_lower_mm == -0.02 and reopened.error_upper_mm == 0.03
        assert "1.65" in reopened.theory_button.toolTip()
        assert reopened.result_values["theory"].text() == "1.650"
        assert reopened.simulation is None and reopened.evaluation is None
        assert all(reopened.result_values[key].text() == "—" for key in ("count", "mean", "variance"))
    finally:
        reopened.close()


def test_settings_save_failure_keeps_previous_settings_and_results(page, application, tmp_path, monkeypatch):
    change_settings(page, application)
    click_debug_action(page, application, "生成90天模拟数据")
    wait_for_page(page)
    previous_evaluation = deepcopy(page.evaluation)
    previous_cards = {key: label.text() for key, label in page.result_values.items()}
    previous_color = page.result_values["mean"].palette().color(QPalette.ColorRole.WindowText)
    previous_tooltip = page.theory_button.toolTip()
    settings_path = tmp_path / "settings.json"
    saved = settings_path.read_bytes()

    def fail_save(*args, **kwargs):
        raise OSError("设置文件保存失败")

    monkeypatch.setattr(page_module, "save_feed_depth_settings", fail_save)
    change_settings(page, application, depth="1.400", lower="-0.1", upper="0.1")

    assert page.depth_mode == "uniform" and page.uniform_depth == 1.5
    assert page.error_lower_mm == -0.05 and page.error_upper_mm == 0.05
    assert settings_path.read_bytes() == saved
    assert page.evaluation == previous_evaluation
    assert page.selected_batch_index == 89
    assert {key: label.text() for key, label in page.result_values.items()} == previous_cards
    assert page.theory_button.toolTip() == previous_tooltip
    assert page.result_values["mean"].property("overLimit") is True
    assert page.result_values["mean"].palette().color(QPalette.ColorRole.WindowText) == previous_color
    assert all(chart.history == previous_evaluation["history"] for chart in page.trend_charts.values())
    assert all(control.isEnabled() for control in (
        page.choose_button, page.debug_button, page.theory_button,
    ))
    assert "设置文件保存失败" in page.process_log.toPlainText()
    assert "设置未应用" in page.task_progress_note.text()
    assert page.progress.value() == 0


def test_generation_failure_restores_controls(page, monkeypatch):
    def fail_generation(*args, **kwargs):
        raise OSError("演示磁盘不可写")

    monkeypatch.setattr(page_module, "generate_feed_depth_simulation", fail_generation)
    page._toggle_simulation()
    wait_for_page(page, success=False)
    assert page.simulation is None
    assert page.progress.value() == 0
    assert "演示磁盘不可写" in page.process_log.toPlainText()
    assert all(control.isEnabled() for control in (
        page.choose_button, page.debug_button, page.theory_button,
    ))
    assert page.import_button.isEnabled() and not page.calculate_button.isEnabled()


def test_empty_page_import_is_available_and_disabled_evaluation_shows_hint(page):
    from ui_helpers import assert_disabled_tooltip

    assert page.import_button.isEnabled()
    assert page.calculate_button.text() == "评估精度"
    assert "导入数据" in page.calculate_button.toolTip()
    assert_disabled_tooltip(page.calculate_button)


def test_import_evaluate_and_failed_replacement_keep_original_data(page, application, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QFileDialog

    path = tmp_path / "measured.csv"
    path.write_text("hole_id,actual_depth_mm,theoretical_depth_mm\n001,1.4,1.5\n002,1.6,1.5\n", encoding="utf-8")
    original = path.read_bytes()
    selected = [str(path)]
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (selected[0], ""))
    page.import_button.click()
    assert not page.import_button.isEnabled()
    assert "等待" in page.import_button.toolTip()
    wait_for_page(page)
    assert page.progress.value() == 100
    assert page.import_button.isEnabled() and page.calculate_button.isEnabled()
    assert page.result_values["count"].text() == "2"
    assert page.result_values["mean"].text() == "1.500"
    assert page.source_badge.text() == "导入数据"
    assert page.trend_charts["bias"].points == []  # 未提供时间与工况，不造历史趋势。
    page.calculate_button.click()
    assert not page.calculate_button.isEnabled()
    wait_for_page(page)
    previous = deepcopy(page.evaluation)
    bad = tmp_path / "bad.csv"
    bad.write_text("hole_id,actual_depth_mm\n001,nan\n", encoding="utf-8")
    selected[0] = str(bad)
    page.import_button.click()
    wait_for_page(page, success=False)
    assert page.evaluation == previous
    assert page.import_button.isEnabled() and page.calculate_button.isEnabled()
    assert "缺少字段" in page.process_log.toPlainText()
    assert path.read_bytes() == original
    selected[0] = ""
    page.choose_button.click()
    assert page.evaluation == previous


def test_import_worker_keeps_ui_responsive_and_releases_controls_on_failure(page, application, monkeypatch, tmp_path):
    path = tmp_path / "input.csv"
    page.selected_file = str(path)
    released = Event()
    worker = []
    main_thread = get_ident()

    def read(*args):
        worker.append(get_ident() != main_thread)
        assert released.wait(3)
        raise ValueError("文件格式有误")

    monkeypatch.setattr(page_module, "load_feed_depth_data", read)
    try:
        page.import_button.click()
        heartbeat = []
        QTimer.singleShot(0, lambda: heartbeat.append(True))
        QTest.qWait(30)
        assert heartbeat and worker == [True]
        assert not page.import_button.isEnabled()
    finally:
        released.set()
        wait_for_page(page, success=False)
    assert page.import_button.isEnabled() and not page.calculate_button.isEnabled()


def test_trends_only_compare_matching_groups_with_known_time_and_condition(application):
    chart = FeedDepthTrendChart("bias_mm", "等待可比历史批次")
    chart.resize(600, 200)
    common = {"count": 2, "bias_mm": .01, "row_id": "R1", "theoretical_depth_mm": 1.5,
              "condition_id": "C1", "is_simulated": False, "measured_at": "2026-10-10"}
    chart.set_history([common, {**common, "row_id": "R2"}, {**common, "condition_id": ""},
                       {**common, "measured_at": ""}, {**common, "is_simulated": True},
                       {**common, "measured_at": "2026-10-09"}])
    chart.selected_index = 0
    chart.show()
    application.processEvents()
    assert [index for index, _point in chart.points] == [5, 0]


def test_multiple_imports_group_details_and_restore_after_simulation(page, application, tmp_path):
    header = "hole_id,actual_depth_mm,theoretical_depth_mm,row_id,measured_at,condition_id\n"
    first = tmp_path / "batch1.csv"
    first.write_text(header + "001,1.4,1.5,R1,2026-10-09,C1\n002,,1.5,R1,2026-10-09,C1\n"
                     "001,2.1,2.0,R2,2026-10-09,C1\n", encoding="utf-8")
    page.selected_file = str(first)
    page.import_button.click()
    wait_for_page(page)
    assert len(page.evaluation["history"]) == 2
    page._show_batch(0)
    rows = read_table_dialog(application, page._show_hole_details)
    assert [row[1][0] for row in rows] == ["001", "002"]
    assert rows[1][3][0] == "—" and rows[1][5][0] == "导入数据 · 缺测"

    second = tmp_path / "batch2.csv"
    second.write_text(header + "001,1.6,1.5,R1,2026-10-10,C1\n", encoding="utf-8")
    page.selected_file = str(second)
    page.import_button.click()
    wait_for_page(page)
    application.processEvents()
    assert len(page.evaluation["history"]) == 3
    assert [index for index, _point in page.trend_charts["bias"].points] == [0, 2]
    original = deepcopy(page.measurements)
    page._toggle_simulation()
    wait_for_page(page)
    assert page.simulation is not None
    page._toggle_simulation()
    wait_for_page(page)
    assert page.simulation is None and page.measurements == original
    assert len(page.evaluation["history"]) == 3
    assert page.source_badge.text() == "导入数据"


def test_standard_deviation_chart_excludes_single_hole_batches(application):
    chart = FeedDepthTrendChart("stddev_mm", "样本不足")
    chart.resize(600, 200)
    chart.show()
    chart.set_history([
        {"measured_at": "2026-07-11T09:00:00+08:00", "condition_id": "C1", "count": 30, "stddev_mm": 0.006},
        {"measured_at": "2026-07-12T09:00:00+08:00", "condition_id": "C1", "count": 1, "stddev_mm": 0.0},
        {"measured_at": "2026-07-13T09:00:00+08:00", "condition_id": "C1", "count": 30, "stddev_mm": 0.008},
    ])
    application.processEvents()
    assert [index for index, _point in chart.points] == [0, 2]
    chart.close()


@pytest.mark.parametrize(("depth", "lower", "upper", "message"), [
    ("nan", "-0.05", "0.05", "理论窝深"),
    ("1.5", "-0.05", "", "同时填写"),
    ("1.5", "0.05", "-0.05", "下限不能大于上限"),
])
def test_invalid_settings_can_be_corrected_in_the_same_dialog(page, application, depth, lower, upper, message):
    inspected = []

    def edit():
        dialog = application.activeModalWidget()
        try:
            mode = dialog.findChild(QComboBox)
            mode.setCurrentIndex(mode.findData("uniform"))
            for name, value in (("feedDepthTheory", depth), ("feedDepthErrorLower", lower), ("feedDepthErrorUpper", upper)):
                dialog.findChild(QLineEdit, name).setText(value)
            apply = next(button for button in dialog.findChildren(QPushButton) if button.text() == "应用设置")
            apply.click()
            error = next(label for label in dialog.findChildren(QLabel) if label.property("validationError"))
            inspected.append((dialog.isVisible(), page.task is None, error.text()))
            assert message in error.text()
            assert not page.settings_path.exists()
            for name, value in (("feedDepthTheory", "1.6"), ("feedDepthErrorLower", "-0.02"), ("feedDepthErrorUpper", "0.03")):
                dialog.findChild(QLineEdit, name).setText(value)
            apply.click()
        finally:
            if dialog.isVisible():
                dialog.reject()

    QTimer.singleShot(0, edit)
    page.theory_button.click()
    wait_for_page(page)
    assert inspected and inspected[0][:2] == (True, True)
    assert page.uniform_depth == 1.6
    assert page.error_lower_mm == -.02 and page.error_upper_mm == .03


def test_repeated_simulation_and_file_selection_preserve_exported_inputs(page, application, monkeypatch, tmp_path):
    from PyQt6.QtWidgets import QFileDialog

    click_debug_action(page, application, "生成90天模拟数据")
    wait_for_page(page)
    first = Path(page.simulation["csv_path"])
    contents = first.read_bytes()
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("", ""))
    page.choose_button.click()
    assert page.simulation is not None and first.read_bytes() == contents
    click_debug_action(page, application, "退出调试模式")
    click_debug_action(page, application, "生成90天模拟数据")
    wait_for_page(page)
    second = Path(page.simulation["csv_path"])
    assert second != first and second.read_bytes() == contents
    selected = tmp_path / "真实文件仅选择.csv"
    selected.write_text("unchanged", encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(selected), ""))
    page.choose_button.click()
    assert page.simulation is not None and page.evaluation is not None
    assert page.selected_file == str(selected)
    assert page.import_button.isEnabled() and page.calculate_button.isEnabled()
    assert first.read_bytes() == second.read_bytes() == contents
    assert selected.read_text(encoding="utf-8") == "unchanged"


@pytest.mark.parametrize(("width", "height", "scale"), [(640, 400, .75), (1500, 1000, 1), (1600, 1000, 2)])
def test_supported_scale_keeps_panels_and_log_inside_viewport(page, application, width, height, scale):
    metrics = UiScale(page)
    try:
        page.resize(width, height)
        page.ui_scale = scale
        metrics.apply(scale)
        application.setStyleSheet(load_stylesheet(scale))
        QTest.qWait(30)
        assert page.widget().width() <= page.viewport().width()
        assert page.process_log.geometry().bottom() < page.progress.geometry().top()
        assert page.progress.geometry().bottom() < page.task_progress_note.geometry().top()
    finally:
        page.ui_scale = 1
        application.setStyleSheet(load_stylesheet())


def test_background_settings_completion_after_page_deletion_does_not_touch_widgets(application, tmp_path, monkeypatch):
    import sys
    from PyQt6 import sip
    from PyQt6.QtCore import QCoreApplication, QEvent, QThreadPool

    page = FeedDepthPage(simulation_root=tmp_path / "simulation", settings_path=tmp_path / "settings.json")
    entered, release = Event(), Event()
    errors = []
    save = page_module.save_feed_depth_settings

    def gated(*args):
        entered.set()
        if not release.wait(5):
            raise TimeoutError("设置保存测试门未释放")
        return save(*args)

    monkeypatch.setattr(page_module, "save_feed_depth_settings", gated)
    monkeypatch.setattr(sys, "excepthook", lambda kind, value, trace: errors.append(str(value)))
    try:
        page._apply_settings({"depth_mode": "uniform", "uniform_depth": 1.6, "error_lower_mm": None, "error_upper_mm": None})
        assert entered.wait(2)
        page.deleteLater()
        QCoreApplication.sendPostedEvents(page, QEvent.Type.DeferredDelete)
        assert sip.isdeleted(page)
    finally:
        release.set()
        assert QThreadPool.globalInstance().waitForDone(5000)
    QTest.qWait(30)
    assert not errors
    assert (tmp_path / "settings.json").exists()
