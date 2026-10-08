"""精度监控桌面程序入口。"""

import sys

from PyQt6.QtCore import QLibraryInfo, QTranslator
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from app.main_window import MainWindow
from app.resources import DISPLAY, load_icon, load_stylesheet


def create_application() -> QApplication:
    application = QApplication(sys.argv)
    application.setApplicationName(DISPLAY["title"])
    translator = QTranslator(application)
    translator.load("qtbase_zh_CN", QLibraryInfo.path(QLibraryInfo.LibraryPath.TranslationsPath))
    application.installTranslator(translator)
    application.setStyle("Fusion")
    font = QFont(DISPLAY["font_family"], 10)
    font.setStyleStrategy(
        QFont.StyleStrategy.PreferAntialias | QFont.StyleStrategy.NoSubpixelAntialias
    )
    application.setFont(font)
    application.setWindowIcon(
        load_icon("branding/application.svg", DISPLAY["colors"]["text"], 32)
    )
    application.setStyleSheet(load_stylesheet())
    return application


def main() -> int:
    application = create_application()
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    sys.exit(main())
