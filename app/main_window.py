"""精度监控主窗口、三类精度选项卡和导航。"""

from PyQt6.QtCore import QDateTime, QEvent, QRect, QSize, Qt, QTimer
from PyQt6.QtGui import QMouseEvent, QPaintEvent, QPainter, QPixmap, QResizeEvent
from PyQt6.QtWidgets import (
    QButtonGroup,
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizeGrip,
    QStackedWidget,
    QStyle,
    QStyleOptionTab,
    QStylePainter,
    QTabBar,
    QVBoxLayout,
    QWidget,
)

from app.pages.robot_position_page import RobotPositionPage
from app.pages.spindle_rotation_page import SpindleRotationPage
from app.resources import DISPLAY, RESOURCE_ROOT, UiScale, load_icon, load_stylesheet


PRECISION_TABS = (
    ("机器人末端定位精度", "precision/robot-position.svg"),
    ("主轴回转精度", "precision/spindle-rotation.svg"),
    ("主轴轴向进给精度", "precision/axial-feed.svg"),
)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle(DISPLAY["title"])
        self.ui_scale = 1.0
        self.scale_timer = QTimer(self)
        self.scale_timer.setSingleShot(True)
        self.scale_timer.setInterval(80)
        self.scale_timer.timeout.connect(self._update_scale)
        available = self.screen().availableGeometry()
        self.resize(QSize(*DISPLAY["window_size"]).boundedTo(available.size()))
        self.setMinimumSize(QSize(*DISPLAY["minimum_size"]).boundedTo(available.size()))
        self.move(available.center() - self.rect().center())

        root = QWidget()
        root.setObjectName("ApplicationRoot")
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.header = self._create_header()
        layout.addWidget(self.header)

        body = QHBoxLayout()
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(0)
        layout.addLayout(body, 1)

        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(DISPLAY["sidebar_width"])
        navigation = QVBoxLayout(sidebar)
        navigation.setContentsMargins(12, 20, 12, 16)
        self.monitor_button = QPushButton("  精度监控")
        self.monitor_button.setObjectName("MonitorNavigation")
        self.monitor_button.setIcon(
            load_icon(
                "navigation/precision-monitor.svg", DISPLAY["colors"]["navigation"]
            )
        )
        self.monitor_button.setIconSize(QSize(24, 24))
        self.monitor_button.setCheckable(True)
        self.navigation_group = QButtonGroup(self)
        self.navigation_group.addButton(self.monitor_button)
        self.monitor_button.setChecked(True)
        self.monitor_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.monitor_button.setFixedHeight(40)
        self.navigation_marker = QFrame(self.monitor_button)
        self.navigation_marker.setObjectName("NavigationMarker")
        self.navigation_marker.setFixedSize(3, 16)
        self.navigation_marker.move(0, 12)
        self.navigation_marker.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self.monitor_button.toggled.connect(self.navigation_marker.setVisible)
        self.navigation_marker.setVisible(self.monitor_button.isChecked())
        navigation.addWidget(self.monitor_button)
        navigation.addStretch()
        body.addWidget(sidebar)

        self.precision_page = PrecisionMonitorPage()
        body.addWidget(self.precision_page, 1)
        self._tab_history = [self.precision_page.tab_bar.currentIndex()]
        self.precision_page.tab_bar.currentChanged.connect(self._remember_tab)
        self.size_grip = QSizeGrip(self)
        self.size_grip.setFixedSize(16, 16)
        self.ui_metrics = UiScale(self)
        self._update_scale()
        # 建立原生窗口后监听换屏；系统 DPI 仍交给 Qt 自动处理。
        self.winId()
        self.windowHandle().screenChanged.connect(self._fit_screen)

    def _fit_screen(self) -> None:
        available = self.screen().availableGeometry()
        self.setMinimumSize(QSize(*DISPLAY["minimum_size"]).boundedTo(available.size()))
        if not self.isMaximized():
            self.resize(self.size().boundedTo(available.size()))
            self.move(
                max(
                    available.left(),
                    min(self.x(), available.right() - self.width() + 1),
                ),
                max(
                    available.top(),
                    min(self.y(), available.bottom() - self.height() + 1),
                ),
            )
        self.scale_timer.start()

    def _update_scale(self) -> None:
        settings = DISPLAY["scaling"]
        width, height = settings["reference_size"]
        scale = round(
            max(
                settings["minimum"],
                min(settings["maximum"], self.width() / width, self.height() / height),
            ),
            2,
        )
        if scale == self.ui_scale:
            return
        self.ui_scale = scale
        self.setUpdatesEnabled(False)
        QApplication.instance().setStyleSheet(load_stylesheet(scale))
        self.ui_metrics.apply(scale)
        self.navigation_marker.move(0, round(12 * scale))
        self.size_grip.move(
            self.width() - self.size_grip.width(),
            self.height() - self.size_grip.height(),
        )
        self.setUpdatesEnabled(True)

    def _create_header(self) -> QFrame:
        header = TitleBar(self)
        header.setObjectName("ApplicationHeader")
        header.setFixedHeight(54)
        row = QHBoxLayout(header)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        self.back_button = self._header_button("返回上一页", "navigation/back.svg")
        self.back_button.setEnabled(False)
        self.back_button.clicked.connect(self._go_back)
        row.addWidget(self.back_button)
        row.addSpacing(16)
        row.addWidget(ReferenceLogo())
        row.addStretch()

        self.clock_label = QLabel()
        self.clock_label.setObjectName("Clock")
        self.clock_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.clock_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        row.addWidget(self.clock_label)
        row.addSpacing(24)
        self.minimize_button = self._header_button("最小化", "window/minimize.svg")
        self.maximize_button = self._header_button("最大化", "window/maximize.svg")
        self.close_button = self._header_button("关闭", "window/close.svg")
        self.close_button.setObjectName("CloseWindowButton")
        self.minimize_button.clicked.connect(self.showMinimized)
        self.maximize_button.clicked.connect(self.toggle_maximized)
        self.close_button.clicked.connect(self.close)
        for button in (self.minimize_button, self.maximize_button, self.close_button):
            row.addWidget(button)
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(1000)
        self._update_clock()
        return header

    def _header_button(self, label: str, icon_path: str) -> QPushButton:
        button = QPushButton()
        button.setProperty("windowControl", True)
        button.setToolTip(label)
        button.setAccessibleName(label)
        button.setIcon(load_icon(icon_path, DISPLAY["colors"]["navigation"]))
        button.setIconSize(QSize(20, 20))
        button.setFixedSize(44, 54)
        return button

    def _remember_tab(self, index: int) -> None:
        if index != self._tab_history[-1]:
            self._tab_history.append(index)
        self.back_button.setEnabled(len(self._tab_history) > 1)

    def _go_back(self) -> None:
        self._tab_history.pop()
        self.precision_page.tab_bar.setCurrentIndex(self._tab_history[-1])

    def toggle_maximized(self) -> None:
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if event.type() == QEvent.Type.WindowStateChange:
            icon_name = "restore.svg" if self.isMaximized() else "maximize.svg"
            label = "还原" if self.isMaximized() else "最大化"
            self.maximize_button.setIcon(
                load_icon(f"window/{icon_name}", DISPLAY["colors"]["navigation"])
            )
            self.maximize_button.setToolTip(label)
            self.maximize_button.setAccessibleName(label)
            self.size_grip.setVisible(not self.isMaximized())

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        self.size_grip.move(
            self.width() - self.size_grip.width(),
            self.height() - self.size_grip.height(),
        )
        self.scale_timer.start()

    def _update_clock(self) -> None:
        self.clock_label.setText(
            QDateTime.currentDateTime().toString("yyyy-MM-dd  HH:mm:ss")
        )


class TitleBar(QFrame):
    """单行窗口标题栏，移动交给操作系统处理。"""

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.window().windowHandle().startSystemMove()

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.window().toggle_maximized()


class ReferenceLogo(QWidget):
    """直接显示参考截图的品牌区域，保留原始图片。"""

    def __init__(self) -> None:
        super().__init__()
        branding = DISPLAY["branding"]
        self.pixmap = QPixmap(str(RESOURCE_ROOT / branding["image"]))
        self.source_rect = QRect(*branding["source_rect"])
        self.setFixedSize(*branding["display_size"])
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawPixmap(self.rect(), self.pixmap, self.source_rect)


class PrecisionMonitorPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("PrecisionMonitorPage")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.tab_bar = PrecisionTabBar()
        self.tab_bar.setObjectName("PrecisionTabs")
        self.tab_bar.setAccessibleName("精度类型")
        self.tab_bar.setExpanding(True)
        self.tab_bar.setDrawBase(False)
        self.tab_bar.setUsesScrollButtons(False)
        self.tab_bar.setElideMode(Qt.TextElideMode.ElideNone)
        self.tab_bar.setIconSize(QSize(28, 28))
        self.tab_bar.setCursor(Qt.CursorShape.PointingHandCursor)
        self.tab_icons = []
        self.content_stack = QStackedWidget()
        self.content_stack.setObjectName("PrecisionContent")

        for index, (title, icon_path) in enumerate(PRECISION_TABS):
            inactive = load_icon(icon_path, DISPLAY["colors"]["inactive"], 28)
            active = load_icon(icon_path, DISPLAY["colors"]["accent"], 28)
            self.tab_icons.append((inactive, active))
            self.tab_bar.addTab(inactive, title)
            if index == 0:
                page = RobotPositionPage()
            elif index == 1:
                page = SpindleRotationPage()
            else:
                page = self._create_placeholder(title)
            self.content_stack.addWidget(page)

        layout.addWidget(self.tab_bar)
        content = QVBoxLayout()
        content.setContentsMargins(16, 16, 16, 16)
        content.addWidget(self.content_stack)
        layout.addLayout(content, 1)

        self.tab_bar.currentChanged.connect(self._select_tab)
        self._select_tab(0)

    def _select_tab(self, index: int) -> None:
        self.content_stack.setCurrentIndex(index)
        for tab_index, (inactive, active) in enumerate(self.tab_icons):
            self.tab_bar.setTabIcon(
                tab_index, active if tab_index == index else inactive
            )

    @staticmethod
    def _create_placeholder(title: str) -> QFrame:
        panel = QFrame()
        panel.setObjectName("PlaceholderPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        heading = QFrame()
        heading.setObjectName("PanelHeading")
        heading_layout = QHBoxLayout(heading)
        heading_layout.setContentsMargins(16, 0, 16, 0)
        marker = QFrame()
        marker.setObjectName("HeadingMarker")
        marker.setFixedSize(3, 15)
        heading_layout.addWidget(marker)
        heading_layout.addSpacing(4)
        heading_layout.addWidget(QLabel(title))
        heading_layout.addStretch()
        layout.addWidget(heading)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setSpacing(14)
        center_layout.addStretch()
        placeholder_icon = QLabel()
        placeholder_icon.setObjectName("PendingIllustration")
        placeholder_icon.setFixedSize(64, 64)
        placeholder_icon.setPixmap(
            load_icon(
                "common/pending.svg", DISPLAY["colors"]["placeholder"], 32
            ).pixmap(QSize(32, 32), 3.0)
        )
        placeholder_icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_layout.addWidget(placeholder_icon, 0, Qt.AlignmentFlag.AlignHCenter)
        message = QLabel("待开发")
        message.setObjectName("PlaceholderMessage")
        message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        center_layout.addWidget(message)
        center_layout.addStretch()
        layout.addWidget(center, 1)
        return panel


class PrecisionTabBar(QTabBar):
    """保留 Qt 原生选项卡行为，将图标和文字作为整体居中。"""

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QStylePainter(self)
        for index in range(self.count()):
            option = QStyleOptionTab()
            self.initStyleOption(option, index)
            painter.drawControl(QStyle.ControlElement.CE_TabBarTabShape, option)
            label_rect = QRect(option.rect)
            label_rect.setWidth(self.tabSizeHint(index).width())
            label_rect.moveCenter(option.rect.center())
            option.rect = label_rect
            painter.drawControl(QStyle.ControlElement.CE_TabBarTabLabel, option)
