from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMainWindow, QStackedWidget

from app.pages.calibration_page import CalibrationPage
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

        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)

        self.initialization_page = InitializationPage(
            navigate_to_calibration=self.show_calibration_page,
        )
        self._stack.addWidget(self.initialization_page)

        self.calibration_page: CalibrationPage | None = None

    def show_calibration_page(self) -> None:
        if self.calibration_page is None:
            self.calibration_page = CalibrationPage()
            self._stack.addWidget(self.calibration_page)
        # Sync joint angles from initialization page to calibration page
        joint_names = self.initialization_page.current_joint_names
        if joint_names and hasattr(self.initialization_page, "robot_view"):
            joint_degrees = list(self.initialization_page.robot_view.joint_degrees)
            self.calibration_page.set_joint_angles(joint_degrees)
        self._stack.setCurrentWidget(self.calibration_page)

    def show_initialization_page(self) -> None:
        self._stack.setCurrentWidget(self.initialization_page)

    def load_identification_parameter_file(self, path: str | Path) -> None:
        self.initialization_page.load_param_file(Path(path))
        self.show_initialization_page()
