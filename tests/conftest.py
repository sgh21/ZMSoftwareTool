"""Qt 回归共用一个 QApplication，按模块恢复界面样式。"""

import pytest
from PyQt6.QtCore import QEvent
from PyQt6.QtWidgets import QApplication

from app.resources import load_stylesheet


@pytest.fixture(scope="session")
def qt_application():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def application(qt_application):
    yield qt_application
    for widget in qt_application.topLevelWidgets():
        widget.close()
        widget.deleteLater()
    qt_application.sendPostedEvents(None, QEvent.Type.DeferredDelete)


@pytest.fixture(scope="module")
def styled_application(qt_application):
    previous_style = qt_application.styleSheet()
    qt_application.setStyleSheet(load_stylesheet())
    yield qt_application
    qt_application.setStyleSheet(previous_style)
