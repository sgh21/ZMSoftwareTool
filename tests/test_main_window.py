"""三页切换、缩放和后台任务关闭约束；所有存储使用隔离目录。"""

from threading import Event

import pytest
from PyQt6.QtCore import QPoint, QRect, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QPushButton

from app import main_window
from app.pages import feed_depth_page
from app.pages.feed_depth_page import FeedDepthPage
from app.pages.robot_position_page import RobotPositionPage
from app.pages.spindle_rotation_page import SpindleRotationPage
from app.resources import DISPLAY
from core.services.position_monitoring_service import PositionMonitoringService
from core.services.spindle_monitoring_service import SpindleMonitoringService
from tests.ui_helpers import wait_for_page


@pytest.fixture
def window(application, tmp_path, monkeypatch):
    monkeypatch.setattr(main_window, "RobotPositionPage", lambda: RobotPositionPage(
        PositionMonitoringService(tmp_path / "robot")))
    monkeypatch.setattr(main_window, "SpindleRotationPage", lambda: SpindleRotationPage(
        SpindleMonitoringService(tmp_path / "spindle")))
    monkeypatch.setattr(main_window, "FeedDepthPage", lambda: FeedDepthPage(
        simulation_root=tmp_path / "feed_simulation", settings_path=tmp_path / "feed_settings.json"))
    original_style = application.styleSheet()
    window = main_window.MainWindow()
    window.show()
    wait_for_page(window.precision_page.content_stack.widget(0))
    application.processEvents()
    yield window
    for index in range(window.precision_page.content_stack.count()):
        wait_for_page(window.precision_page.content_stack.widget(index), success=False)
    window.close()
    application.setStyleSheet(original_style)


@pytest.mark.parametrize("page_index", [0, 1, 2])
def test_close_waits_for_background_task(window, monkeypatch, page_index):
    stack = window.precision_page.content_stack
    page = stack.widget(page_index)
    finished = Event()
    messages = []
    monkeypatch.setattr(main_window.QMessageBox, "information", lambda *args: messages.append(args[2]))
    if page_index == 2:
        save = feed_depth_page.save_feed_depth_settings

        def save_after_release(*args):
            assert finished.wait(5)
            return save(*args)

        monkeypatch.setattr(feed_depth_page, "save_feed_depth_settings", save_after_release)
        page._apply_settings(page._current_settings())
    else:
        page._run_task("后台处理", lambda _progress: finished.wait(5), lambda _result: None)
    window.precision_page.tab_bar.setCurrentIndex((page_index + 1) % 3)
    try:
        QTest.mouseClick(window.close_button, Qt.MouseButton.LeftButton)
        assert window.isVisible()
        assert window.precision_page.tab_bar.currentIndex() == page_index
        assert "等待进度完成" in messages[-1]
    finally:
        finished.set()
        wait_for_page(page)
    QTest.mouseClick(window.close_button, Qt.MouseButton.LeftButton)
    assert not window.isVisible()


def test_switching_three_tabs_and_back_preserves_each_pages_state(window):
    stack = window.precision_page.content_stack
    robot, spindle, feed = [stack.widget(index) for index in range(3)]
    robot.result_metric.setCurrentIndex(0)
    robot.trend_metric.setCurrentIndex(1)
    spindle.signal_select.setCurrentIndex(3)
    feed._toggle_simulation()
    wait_for_page(feed)
    feed._show_batch(17)
    cards = {key: label.text() for key, label in feed.result_values.items()}
    output = feed.simulation["csv_path"]
    tab_bar = window.precision_page.tab_bar
    sequence = [1, 2, 0, 2, 1, 0]

    def assert_state():
        assert robot.result_metric.currentIndex() == 0
        assert robot.trend_metric.currentIndex() == 1
        assert spindle.signal_select.currentIndex() == 3
        assert feed.selected_batch_index == 17
        assert {key: label.text() for key, label in feed.result_values.items()} == cards
        assert feed.simulation["csv_path"] == output
        assert len(feed.trend_charts["bias"].history) == 90

    for index in sequence:
        QTest.mouseClick(tab_bar, Qt.MouseButton.LeftButton, pos=tab_bar.tabRect(index).center())
        assert stack.currentIndex() == index
        assert_state()
    for index in reversed([0, *sequence][:-1]):
        QTest.mouseClick(window.back_button, Qt.MouseButton.LeftButton)
        assert stack.currentIndex() == index
        assert_state()
    assert not window.back_button.isEnabled()


def assert_page_actions_reachable(window):
    stack = window.precision_page.content_stack
    for index in range(stack.count()):
        window.precision_page.tab_bar.setCurrentIndex(index)
        QTest.qWait(20)
        page = stack.widget(index)
        scroll = page.scroll_area if isinstance(page, RobotPositionPage) else page
        assert scroll.widget().width() <= scroll.viewport().width(), (index, window.size(), window.ui_scale)
        if index == 0:
            actions = [next(button for button in page.findChildren(QPushButton) if button.text() == text)
                       for text in ("导入观测", "评估精度")]
        elif index == 1:
            actions = [page.daily_button, page.evaluate_button]
        else:
            actions = [page.choose_button, page.calculate_button]
        for button in actions:
            scroll.ensureWidgetVisible(button)
            QTest.qWait(10)
            bounds = QRect(button.mapTo(scroll.viewport(), QPoint()), button.size())
            assert button.isVisible() and scroll.viewport().rect().contains(bounds), (index, button.text(), bounds)


def test_resizing_back_and_forth_keeps_three_pages_and_actions_reachable(window):
    for width, height in ((800, 500), (1280, 800), (1920, 1080), (1280, 800), (800, 500)):
        window.resize(width, height)
        QTest.qWait(130)
        assert_page_actions_reachable(window)


def test_maximum_ui_scale_keeps_all_three_pages_reachable(window, monkeypatch):
    window.resize(1920, 1080)
    monkeypatch.setitem(DISPLAY["scaling"], "minimum", 2.0)
    window._update_scale()
    QTest.qWait(130)
    assert window.ui_scale == 2
    assert_page_actions_reachable(window)
