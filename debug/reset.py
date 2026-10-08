"""仅供开发调试：清空使用数据并重新创建窗口，正式入口不加载本模块。"""

import argparse
from base64 import urlsafe_b64encode
import json
from pathlib import Path
import shutil

from PyQt6.QtCore import QObject
from PyQt6.QtNetwork import QLocalServer, QLocalSocket
from PyQt6.QtWidgets import QApplication, QWidget

from app.main_window import MainWindow
from app.resources import PROJECT_ROOT, load_stylesheet
from core.services.position_persistence import read_json, write_document
from main import create_application


def reset_usage_data(root=PROJECT_ROOT):
    """仅删除本工程 storage 内的业务数据，保留参数和设置。"""
    root = Path(root).resolve()
    storage = root / "storage"
    position = storage / "position_monitoring"
    config = read_json(root / "config" / "spindle_monitoring.json")
    spindle = root / config["storage_root"]
    # 删除前一次性核对两处真实路径，拒绝外部目录、重叠目录和重定向。
    targets = (position, spindle)
    for path in (storage, *targets):
        if path.resolve() != path.absolute() or not path.resolve().is_relative_to(root):
            raise ValueError(f"reset 不允许重定向或工程外目录：{path}")
    if (spindle == storage or not spindle.is_relative_to(storage)
            or spindle.is_relative_to(position) or position.is_relative_to(spindle)):
        raise ValueError("主轴存储必须位于 storage 内，且与定位存储互不包含")

    states = [read_json(path / "state.json") if (path / "state.json").exists() else {}
              for path in targets]
    removals = [child for path in targets if path.exists() for child in path.iterdir()
                if child.name not in ("state.json", ".gitkeep")
                and child != position / "parameters"]
    for child in removals:
        if not child.resolve().is_relative_to(child.parent.resolve()):
            raise ValueError(f"reset 不允许删除重定向的数据路径：{child}")

    # 老版本参数可能仍在 state.json 中，先迁移再清除状态引用。
    parameter_path = position / "parameters" / "current.json"
    if not parameter_path.exists() and "parameters" in states[0]:
        write_document(parameter_path, states[0]["parameters"])
    write_document(position / "state.json", {
        "settings": states[0].get("settings", {}),
        "baseline_path": None, "current_batch_path": None, "latest_result": None,
    })
    write_document(spindle / "state.json", {
        "schema_version": 1,
        "settings": {**states[1].get("settings", {"thresholds": config["thresholds"]}),
                     "thresholds_model_version": None},
        "runs": {}, "packages": [], "models": [], "current_model_version": None,
        "results": [],
    })
    for child in removals:
        if child.is_dir() and not child.is_symlink() and not child.is_junction():
            shutil.rmtree(child)
        elif child.is_junction():
            child.rmdir()
        else:
            child.unlink()


def server_name(root=PROJECT_ROOT):
    # 路径区分不同检出目录，不使用摘要或固定的全局服务名。
    encoded = urlsafe_b64encode(str(Path(root).resolve()).casefold().encode()).decode()
    return "SoftwareTools-debug-" + encoded.rstrip("=")


def send_command(command, root=PROJECT_ROOT):
    """返回已有调试窗口的执行结果；无监听窗口时返回 None。"""
    socket = QLocalSocket()
    socket.connectToServer(server_name(root))
    if not socket.waitForConnected(500):
        return None
    socket.write((command + "\n").encode())
    socket.flush()
    while not socket.canReadLine():
        if not socket.waitForReadyRead(30000):
            raise OSError("调试窗口未返回结果，请查看窗口状态；本次不重复执行 reset")
    result = json.loads(bytes(socket.readLine()).decode())
    socket.disconnectFromServer()
    return result


class DebugSession(QObject):
    def __init__(self, application, root=PROJECT_ROOT):
        super().__init__(application)
        self.root = Path(root).resolve()
        self.window = None
        self.server = QLocalServer(self)
        if not self.server.listen(server_name(self.root)):
            raise OSError(f"无法启动调试命令入口：{self.server.errorString()}")
        self.server.newConnection.connect(self._connect)
        application.aboutToQuit.connect(self.server.close)

    def _connect(self):
        socket = self.server.nextPendingConnection()
        socket.disconnected.connect(socket.deleteLater)
        socket.readyRead.connect(lambda: self._receive(socket))
        self._receive(socket)

    def _receive(self, socket):
        if socket.canReadLine():
            result = self.execute(bytes(socket.readLine()).decode().strip())
            socket.write((json.dumps(result, ensure_ascii=False) + "\n").encode())
            socket.disconnectFromServer()

    def execute(self, command):
        if command not in ("show", "reset"):
            return {"ok": False, "message": "仅支持 reset 指令"}
        if command == "reset":
            if self.window and any(getattr(widget, "task", None) is not None
                                   for widget in self.window.findChildren(QWidget)):
                return {"ok": False, "message": "当前任务尚未结束，请完成后再输入 reset"}
            if QApplication.activeModalWidget() or QApplication.activePopupWidget():
                return {"ok": False, "message": "请先关闭弹窗或菜单，再输入 reset"}
            try:
                reset_usage_data(self.root)
            except (OSError, ValueError) as error:
                return {"ok": False, "message": f"reset 未完成：{error}"}
            self._rebuild_window()
        elif self.window is None:
            self._rebuild_window()
        self.window.show()
        self.window.raise_()
        self.window.activateWindow()
        return {"ok": True, "message": "reset 完成：已清空使用数据，保留参数和阈值，界面已刷新"
                if command == "reset" else "调试窗口已打开；另开终端执行 python -m debug reset"}

    def _rebuild_window(self):
        previous = self.window
        QApplication.instance().setStyleSheet(load_stylesheet())
        self.window = MainWindow()
        self.window.setWindowTitle(self.window.windowTitle() + " [DEBUG]")
        if previous:
            self.window.restoreGeometry(previous.saveGeometry())
            self.window.precision_page.tab_bar.setCurrentIndex(
                previous.precision_page.tab_bar.currentIndex())
        self.window.show()
        if previous:
            previous.close()
            previous.deleteLater()


def main(argv=None):
    parser = argparse.ArgumentParser(description="独立调试入口；reset 清空使用数据，保留参数和阈值")
    parser.add_argument("command", nargs="?", choices=("reset",), help="重置数据并刷新调试窗口")
    args = parser.parse_args(argv)
    application = create_application()
    command = args.command or "show"
    try:
        result = send_command(command)
        if result is not None:
            print(result["message"])
            return 0 if result["ok"] else 1
        session = DebugSession(application)
        result = session.execute(command)
    except (OSError, ValueError) as error:
        print(f"调试命令失败：{error}")
        return 1
    print(result["message"], flush=True)
    return application.exec() if result["ok"] else 1
