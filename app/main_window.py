from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMainWindow, QVBoxLayout, QWidget


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("ZMSoftware Robot Digital Twin Tool")
        self.resize(1280, 720)

        placeholder = QLabel("Robot digital twin workspace")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addWidget(placeholder)
        self.setCentralWidget(central)
