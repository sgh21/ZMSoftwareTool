from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMainWindow

from app.pages.initialization_page import InitializationPage


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ZMSoftware Robot Digital Twin Tool")
        logo_path = Path(__file__).resolve().parent / "resources" / "app_logo.png"
        if logo_path.exists():
            self.setWindowIcon(QIcon(str(logo_path)))
        self.resize(1440, 810)
        self.setMinimumSize(1180, 700)
        self.initialization_page = InitializationPage()
        self.setCentralWidget(self.initialization_page)
