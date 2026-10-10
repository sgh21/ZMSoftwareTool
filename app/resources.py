"""集中读取配置、样式和图标，资源路径不依赖启动目录。"""

import json
import re

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import QAbstractButton, QLabel, QLayout, QPushButton, QTabBar, QWidget

from core.runtime_paths import bundle_root


PROJECT_ROOT = bundle_root()
RESOURCE_ROOT = PROJECT_ROOT / "resources"
DISPLAY = json.loads((PROJECT_ROOT / "config/display.json").read_text(encoding="utf-8"))


def make_note(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("robotNote", True)
    label.setWordWrap(True)
    return label


def make_button(text: str, callback, primary: bool = False) -> QPushButton:
    button = QPushButton(text)
    button.setProperty("robotAction", True)
    button.setProperty("primary", primary)
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    button.clicked.connect(callback)
    return button


def set_status_light(light, state, details, status):
    if light.property("state") != state:
        light.setProperty("state", state)
        light.style().unpolish(light)
        light.style().polish(light)
        light.update()
    light.setToolTip(f"{status}\n{details}")
    light.setAccessibleDescription(f"{status}；{details}")


def load_stylesheet(scale: float = 1.0) -> str:
    stylesheet = (RESOURCE_ROOT / "styles/light.qss").read_text(encoding="utf-8")
    stylesheet = stylesheet.replace("@font_family@", DISPLAY["font_family"])
    stylesheet = stylesheet.replace("@resource_root@", RESOURCE_ROOT.as_posix())
    for name, color in DISPLAY["colors"].items():
        stylesheet = stylesheet.replace(f"@{name}@", color)
    return re.sub(
        r"(\d+)px", lambda match: f"{round(int(match[1]) * scale)}px", stylesheet
    )


class UiScale:
    """记录设计尺寸，每次从原值缩放；保留原生控件与当前页面状态。"""

    def __init__(self, root: QWidget, include_widget_sizes: bool = True) -> None:
        self.layouts = []
        self.spacers = []
        self.bounds = []
        self.icons = []
        self.pictures = []
        for layout in root.findChildren(QLayout):
            margins = layout.contentsMargins()
            self.layouts.append(
                (
                    layout,
                    (margins.left(), margins.top(), margins.right(), margins.bottom()),
                    layout.spacing(),
                )
            )
            for index in range(layout.count()):
                spacer = layout.itemAt(index).spacerItem()
                if spacer is not None:
                    self.spacers.append(
                        (spacer, spacer.sizeHint(), spacer.sizePolicy())
                    )
        if not include_widget_sizes:
            return
        for widget in root.findChildren(QWidget):
            minimum, maximum = widget.minimumSize(), widget.maximumSize()
            if minimum != QSize(0, 0) or maximum != QSize(16777215, 16777215):
                self.bounds.append((widget, minimum, maximum))
            if isinstance(widget, (QAbstractButton, QTabBar)):
                self.icons.append((widget, widget.iconSize()))
            if isinstance(widget, QLabel):
                pixmap = widget.pixmap()
                if pixmap is not None and not pixmap.isNull():
                    self.pictures.append((widget, pixmap))

    def apply(self, scale: float) -> None:
        for layout, margins, spacing in self.layouts:
            layout.setContentsMargins(*(round(value * scale) for value in margins))
            if spacing >= 0:
                layout.setSpacing(round(spacing * scale))
        for spacer, size, policy in self.spacers:
            spacer.changeSize(
                round(size.width() * scale),
                round(size.height() * scale),
                policy.horizontalPolicy(),
                policy.verticalPolicy(),
            )
        for widget, minimum, maximum in self.bounds:
            widget.setMinimumSize(
                round(minimum.width() * scale), round(minimum.height() * scale)
            )
            widget.setMaximumSize(
                *(
                    16777215 if value == 16777215 else round(value * scale)
                    for value in (maximum.width(), maximum.height())
                )
            )
        for widget, size in self.icons:
            widget.setIconSize(
                QSize(round(size.width() * scale), round(size.height() * scale))
            )
        for widget, pixmap in self.pictures:
            widget.setPixmap(
                pixmap.scaled(
                    round(pixmap.width() * scale),
                    round(pixmap.height() * scale),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        for layout, _, _ in self.layouts:
            layout.invalidate()


def fit_dialog(dialog: QWidget, width: int, height: int | None = None) -> None:
    """沿用主窗口比例；未指定高度的表单按内容收紧。"""
    scale = dialog.parentWidget().window().ui_scale
    UiScale(dialog, include_widget_sizes=False).apply(scale)
    available = dialog.screen().availableGeometry().size()
    width = min(round(width * scale), available.width())
    layout = dialog.layout()
    layout.activate()
    if height is None:
        height = layout.totalHeightForWidth(width) if layout.hasHeightForWidth() else dialog.sizeHint().height()
    else:
        height = round(height * scale)
    dialog.resize(QSize(width, height).boundedTo(available))


def load_icon(name: str, color: str, size: int = 24) -> QIcon:
    path = RESOURCE_ROOT / "icons" / name
    # 从同一个 SVG 生成不同状态的颜色，替换资源时不必维护多份图标。
    renderer = QSvgRenderer(str(path))
    icon = QIcon()
    # 同时提供放大后的逻辑尺寸，避免大窗口中只有图标仍停留在原始大小。
    for extent in (size, round(size * DISPLAY["scaling"]["maximum"])):
        for ratio in (1, 2, 3):
            pixmap = QPixmap(extent * ratio, extent * ratio)
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            renderer.render(painter)
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceIn
            )
            painter.fillRect(pixmap.rect(), QColor(color))
            painter.end()
            pixmap.setDevicePixelRatio(ratio)
            for mode in (QIcon.Mode.Normal, QIcon.Mode.Active, QIcon.Mode.Selected):
                icon.addPixmap(pixmap, mode)
    return icon
