"""窗口关闭须等待另一选项卡的后台处理完成。"""

from threading import Event

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest

from app import main_window
from app.pages.robot_position_page import RobotPositionPage
from app.pages.spindle_rotation_page import SpindleRotationPage
from core.services.position_monitoring_service import PositionMonitoringService
from core.services.spindle_monitoring_service import SpindleMonitoringService
from tests.ui_helpers import wait_for_page


@pytest.mark.parametrize("page_index", [0, 1])
def test_close_waits_for_background_task(application, tmp_path, monkeypatch, page_index):
    monkeypatch.setattr(main_window, "RobotPositionPage", lambda: RobotPositionPage(
        PositionMonitoringService(tmp_path)))
    monkeypatch.setattr(main_window, "SpindleRotationPage", lambda: SpindleRotationPage(
        SpindleMonitoringService(tmp_path / "spindle")))
    window = main_window.MainWindow()
    window.show()
    stack = window.precision_page.content_stack
    wait_for_page(stack.widget(0))
    page = stack.widget(page_index)
    finished = Event()
    messages = []
    monkeypatch.setattr(main_window.QMessageBox, "information", lambda *args: messages.append(args[2]))
    page._run_task("后台处理", lambda _progress: finished.wait(5), lambda _result: None)
    window.precision_page.tab_bar.setCurrentIndex(2)
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
