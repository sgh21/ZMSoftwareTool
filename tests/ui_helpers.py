"""Qt 回归共用的后台任务等待，不使用主运行目录。"""

from time import monotonic

from PyQt6.QtTest import QTest

from app.pages.robot_position_page import RobotPositionPage


def wait_for_page(page, timeout=5, *, success=True):
    deadline = monotonic() + timeout
    while page.task is not None and monotonic() < deadline:
        QTest.qWait(10)
    assert page.task is None, page.task_progress_note.text()
    if success:
        assert not page.task_progress_note.text().startswith("处理失败"), page.task_progress_note.text()


def ready_position_page(service):
    page = RobotPositionPage(service)
    wait_for_page(page)
    return page


def refresh_page(page):
    """测试直接操作服务之后，模拟页面业务操作结束时的后台视图刷新。"""
    page._run_task("刷新测试视图", lambda _progress: None,
                   lambda _result: page._refresh_latest_result(), refresh=True, log=False)
    wait_for_page(page)
