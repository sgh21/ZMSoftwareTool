"""集中读取配置、样式和图标，资源路径不依赖启动目录。"""

import json
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESOURCE_ROOT = PROJECT_ROOT / "resources"
DISPLAY = json.loads((PROJECT_ROOT / "config/display.json").read_text(encoding="utf-8"))


def load_stylesheet() -> str:
    stylesheet = (RESOURCE_ROOT / "styles/light.qss").read_text(encoding="utf-8")
    stylesheet = stylesheet.replace("@font_family@", DISPLAY["font_family"])
    for name, color in DISPLAY["colors"].items():
        stylesheet = stylesheet.replace(f"@{name}@", color)
    return stylesheet


def load_icon(name: str, color: str, size: int = 24) -> QIcon:
    path = RESOURCE_ROOT / "icons" / name
    # 从同一个 SVG 生成不同状态的颜色，替换资源时不必维护多份图标。
    renderer = QSvgRenderer(str(path))
    icon = QIcon()
    for scale in (1, 2, 3):
        pixmap = QPixmap(size * scale, size * scale)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        renderer.render(painter)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
        painter.fillRect(pixmap.rect(), QColor(color))
        painter.end()
        pixmap.setDevicePixelRatio(scale)
        for mode in (QIcon.Mode.Normal, QIcon.Mode.Active, QIcon.Mode.Selected):
            icon.addPixmap(pixmap, mode)
    return icon
