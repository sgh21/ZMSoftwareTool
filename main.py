from __future__ import annotations

import sys

# PyVista must initialize before Qt/PySide when pyvistaqt is used later.
import pyvista as _pyvista  # noqa: F401
from PySide6.QtWidgets import QApplication

from app.main_window import MainWindow


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
