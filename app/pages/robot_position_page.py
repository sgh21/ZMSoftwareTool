"""机器人定位监控：按用户约定的区域 1—8 组织结果、设置和日志。"""

from math import isfinite

from PyQt6.QtCore import QDateTime, QPointF, QRectF, QSize, Qt
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.resources import DISPLAY, fit_dialog, load_icon


PREDICTION_COLUMNS = ["点位", "X / mm", "Y / mm", "Z / mm", "距离 / mm", "判定"]


class RobotPositionPage(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.processing_points = []
        self.setObjectName("RobotPositionPage")
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(8)
        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("RobotPageScroll")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        canvas = QWidget()
        canvas.setObjectName("RobotPageCanvas")
        self.scroll_area.setWidget(canvas)
        layout = QVBoxLayout(canvas)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(8)
        columns = QHBoxLayout()
        columns.setSpacing(14)
        columns.addWidget(self._results_view(), 13)
        columns.addWidget(self._settings_sidebar(), 7)
        layout.addLayout(columns, 1)
        page_layout.addWidget(self.scroll_area, 1)
        self.append_log("页面已就绪，等待手眼参数、基准及观测数据。")

    def append_log(self, message: str, level: str = "INFO") -> None:
        timestamp = QDateTime.currentDateTime().toString("HH:mm:ss")
        self.process_log.appendPlainText(f"{timestamp} [{level}] {message}")

    @staticmethod
    def _note(text: str) -> QLabel:
        label = QLabel(text)
        label.setProperty("robotNote", True)
        label.setWordWrap(True)
        return label

    @staticmethod
    def _button(text: str, callback, primary: bool = False) -> QPushButton:
        button = QPushButton(text)
        button.setProperty("robotAction", True)
        button.setProperty("primary", primary)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(callback)
        return button

    def _planned_button(
        self, text: str, description: str, primary=False
    ) -> QPushButton:
        message = f"{text}：{description}".replace("\n", " ")
        button = self._button(
            text,
            lambda: self.append_log(message, "WARN"),
            primary,
        )
        button.setToolTip(description)
        return button

    def _results_view(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("RobotResults")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)
        summary = QFrame()
        summary.setObjectName("Region1")
        body = QVBoxLayout(summary)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(12)
        title = QLabel("机器人定位精度")
        title.setObjectName("RobotPageTitle")
        body.addWidget(title)
        body.addWidget(self._note("本次评估：—    基准批次：—"))

        metrics = QHBoxLayout()
        metrics.setSpacing(12)
        self.axis_values = {}
        for axis, title in (
            ("X", "X 方向"),
            ("Y", "Y 方向"),
            ("Z", "Z 方向"),
            ("distance", "距离"),
        ):
            metric = QFrame()
            metric.setProperty("axisMetric", True)
            metric.setProperty("axis", axis)
            content = QVBoxLayout(metric)
            content.setContentsMargins(10, 12, 10, 12)
            content.setSpacing(4)
            name = QLabel(title)
            name.setProperty("axisTitle", True)
            name.setAlignment(Qt.AlignmentFlag.AlignCenter)
            content.addWidget(name)
            value_row = QHBoxLayout()
            value_row.setSpacing(6)
            value_row.addStretch()
            value = QLabel("—")
            value.setProperty("axisValue", True)
            self.axis_values[axis] = value
            value_row.addWidget(value)
            value_row.addWidget(QLabel("mm"), 0, Qt.AlignmentFlag.AlignBottom)
            value_row.addStretch()
            content.addLayout(value_row)
            threshold = self._note("阈值：未设置")
            threshold.setAlignment(Qt.AlignmentFlag.AlignCenter)
            content.addWidget(threshold)
            metrics.addWidget(metric, 1)
        body.addLayout(metrics)

        self.conclusion = QLabel("尚未评估 · 等待基准与本次观测")
        self.conclusion.setObjectName("PositionConclusion")
        self.conclusion.setWordWrap(True)
        body.addWidget(self.conclusion)
        layout.addWidget(summary)
        layout.addWidget(self._trend_view(), 1)

        self.detail_tabs = QTabWidget()
        self.detail_tabs.setObjectName("RobotDetailTabs")
        self.detail_tabs.setProperty("region", 3)
        self.detail_tabs.setAccessibleName("定位监控结果详情")
        self.detail_tabs.addTab(self._point_results(), "加工点位预测")
        self.detail_tabs.addTab(self._observation_view(), "观测图像")
        self.detail_tabs.addTab(self._alarm_view(), "报警与维护")
        layout.addWidget(self.detail_tabs, 1)
        return panel

    def _trend_view(self) -> QWidget:
        area = QWidget()
        area.setObjectName("Region2")
        body = QVBoxLayout(area)
        body.setContentsMargins(0, 2, 0, 0)
        body.setSpacing(8)
        title = QLabel("定位变化趋势")
        title.setProperty("robotSectionTitle", True)
        body.addWidget(title)
        self.trend_metric = QComboBox()
        self.trend_metric.setProperty("robotInput", True)
        self.trend_metric.setAccessibleName("趋势纵轴指标")
        self.trend_metric.addItem("XYZ 分量", "xyz")
        self.trend_metric.addItem("位置距离", "distance")
        legend = QHBoxLayout()
        self.trend_axis_title = self._note("XYZ 定位精度 / mm")
        legend.addWidget(self.trend_axis_title)
        legend.addStretch()
        self.trend_legend = {}
        for axis, name in (("X", "X"), ("Y", "Y"), ("Z", "Z"), ("distance", "距离")):
            label = QLabel(f"━ {name}")
            label.setProperty("axis", axis)
            label.setVisible(axis != "distance")
            self.trend_legend[axis] = label
            legend.addWidget(label)
        legend.addWidget(self.trend_metric)
        body.addLayout(legend)
        self.trend_chart = PositionTrendChart()
        body.addWidget(self.trend_chart, 1)
        self.trend_metric.currentIndexChanged.connect(self._change_trend_metric)
        caption = self._note("评估时间 / 批次")
        caption.setAlignment(Qt.AlignmentFlag.AlignRight)
        body.addWidget(caption)
        return area

    def _change_trend_metric(self) -> None:
        distance = self.trend_metric.currentData() == "distance"
        self.trend_axis_title.setText(
            "位置距离 / mm" if distance else "XYZ 定位精度 / mm"
        )
        for axis, label in self.trend_legend.items():
            label.setVisible((axis == "distance") == distance)
        self.trend_chart.mode = self.trend_metric.currentData()
        self.trend_chart.update()
        self.append_log(f"趋势指标已切换为{self.trend_metric.currentText()}。")

    def _point_results(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 10, 0, 0)
        layout.setSpacing(8)
        self.point_table = self._table(PREDICTION_COLUMNS)
        self.point_table.setMinimumHeight(114)
        self.point_table.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored
        )
        layout.addWidget(self.point_table, 1)
        footer = QHBoxLayout()
        message = self._note("加工点位预测结果待评估")
        message.setWordWrap(False)
        footer.addWidget(message)
        footer.addStretch()
        footer.addWidget(self._button("历史记录", self._show_history, True))
        footer.addWidget(self._button("展开明细", self._show_points, True))
        layout.addLayout(footer)
        return page

    def _observation_view(self) -> QWidget:
        page = QWidget()
        body = QVBoxLayout(page)
        body.setContentsMargins(0, 10, 0, 0)
        body.setSpacing(8)
        row = QHBoxLayout()
        self.observation_source = QComboBox()
        self.observation_source.setProperty("robotInput", True)
        self.observation_source.setAccessibleName("观测图像来源")
        self.observation_source.addItems(["本次观测", "基准观测"])
        row.addWidget(self.observation_source)
        row.addStretch()
        row.addWidget(self._note("有效点位：— / —"))
        body.addLayout(row)
        image_area = QFrame()
        image_area.setObjectName("ObservationImage")
        image_area.setMinimumHeight(116)
        image_layout = QVBoxLayout(image_area)
        image_layout.addStretch()
        icon = QLabel()
        icon.setPixmap(
            load_icon("robot/camera.svg", DISPLAY["colors"]["inactive"], 32).pixmap(
                QSize(32, 32), 3.0
            )
        )
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_layout.addWidget(icon)
        self.image_message = QLabel("尚无本次标志点图像")
        self.image_message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        image_layout.addWidget(self.image_message)
        image_layout.addStretch()
        body.addWidget(image_area, 1)
        self.observation_source.currentIndexChanged.connect(
            lambda index: self.image_message.setText(
                "尚无基准标志点图像" if index else "尚无本次标志点图像"
            )
        )
        return page

    def _alarm_view(self) -> QWidget:
        page = QWidget()
        body = QVBoxLayout(page)
        body.setContentsMargins(0, 14, 0, 0)
        body.addWidget(QLabel("报警状态：尚未评估"))
        body.addWidget(self._note("超限方向：—    最近报警：—"))
        body.addWidget(
            self._note(
                "完成评估后显示超限项和维护建议。\n维护完成后，通过复测确认定位变化。"
            )
        )
        body.addStretch()
        actions = QHBoxLayout()
        actions.addWidget(self._button("报警记录", self._show_alarms, True))
        actions.addWidget(self._button("维护记录", self._show_maintenance, True))
        actions.addStretch()
        body.addLayout(actions)
        return page

    def _settings_sidebar(self) -> QFrame:
        panel = QFrame()
        panel.setObjectName("RobotSettings")
        panel.setMinimumWidth(332)
        body = QVBoxLayout(panel)
        body.setContentsMargins(16, 16, 16, 16)
        body.setSpacing(14)
        title = QLabel("检测设置")
        title.setObjectName("RobotSettingsTitle")
        body.addWidget(title)

        hand_eye = self._settings_section(body, 4, "机器人手眼参数")
        hand_eye.addWidget(
            self._note("参数状态：未加载\n变换关系：相机 → 末端（TCP）\n参数版本：—")
        )
        actions = QHBoxLayout()
        actions.addWidget(
            self._planned_button(
                "加载手眼参数",
                "手眼参数读取尚未接入，未执行加载。后续读取相机到被评估末端（TCP）的固定变换及参数版本。",
            )
        )
        actions.addWidget(
            self._planned_button(
                "查看手眼参数",
                "当前尚未加载手眼参数。后续查看相机到末端（TCP）的旋转、平移和参数版本。",
            )
        )
        hand_eye.addLayout(actions)

        baseline = self._settings_section(body, 5, "测量基准")
        baseline.addWidget(self._note("基准批次：未建立\n采集时间：—"))
        actions = QHBoxLayout()
        actions.addWidget(
            self._planned_button(
                "建立基准",
                "待实现：保存初始视觉观测、测量点编号及手眼参数版本，并关联末端在基坐标系下的初始朝向。",
            )
        )
        actions.addWidget(
            self._planned_button(
                "选择基准",
                "待实现：选择历史基准批次。基准与本次观测应采用相同拍摄程序、指令位姿及固定靶标。",
            )
        )
        baseline.addLayout(actions)

        conditions = self._settings_section(body, 7, "判定与点位设置")
        conditions.addWidget(self._note("X / Y / Z / 距离阈值：未设置"))
        self.point_count_label = self._note("加工点位：0 个")
        conditions.addWidget(self.point_count_label)
        actions = QHBoxLayout()
        actions.addWidget(self._button("阈值设置", self._show_settings))
        actions.addWidget(self._button("点位管理", self._show_point_editor))
        conditions.addLayout(actions)

        log_area = QFrame()
        log_area.setObjectName("Region8")
        log_layout = QVBoxLayout(log_area)
        log_layout.setContentsMargins(0, 0, 0, 0)
        heading = QHBoxLayout()
        title = QLabel("运行日志")
        title.setProperty("robotSectionTitle", True)
        heading.addWidget(title)
        heading.addStretch()
        self.process_log = QPlainTextEdit()
        self.process_log.setObjectName("RobotProcessLog")
        self.process_log.setReadOnly(True)
        self.process_log.setMinimumHeight(110)
        self.process_log.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Ignored
        )
        self.process_log.setPlaceholderText("操作过程、提示与错误将在这里显示")
        log_layout.addLayout(heading)
        log_layout.addWidget(self.process_log, 1)
        body.addWidget(log_area, 1)
        body.addWidget(self._action_bar())
        return panel

    def _action_bar(self) -> QFrame:
        bar = QFrame()
        bar.setObjectName("Region6")
        bar.setAccessibleName("定位监控操作栏")
        actions = QHBoxLayout(bar)
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(4)
        actions.addWidget(
            self._planned_button(
                "采集靶标",
                "相机接口尚未接入，未开始采集。后续按与基准相同的程序、固定位姿拍摄靶标。",
                True,
            ),
            1,
        )
        actions.addWidget(
            self._planned_button(
                "导入观测",
                "观测数据接口尚未接入，未执行导入。后续关联基准、拍摄程序及评估批次。",
                True,
            ),
            1,
        )
        actions.addWidget(self._button("清空日志", self.process_log.clear, True), 1)
        actions.addWidget(
            self._planned_button(
                "评估精度",
                "评估算法尚未接入，未生成结果。后续将视觉观测误差映射为末端定位误差，"
                "并预测加工点位的 XYZ 及距离指标。",
                True,
            ),
            1,
        )
        return bar

    def _settings_section(
        self, parent: QVBoxLayout, number: int, title: str
    ) -> QVBoxLayout:
        group = QFrame()
        group.setObjectName(f"Region{number}")
        group.setProperty("settingsGroup", True)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 12)
        layout.setSpacing(8)
        label = QLabel(title)
        label.setProperty("robotSectionTitle", True)
        layout.addWidget(label)
        parent.addWidget(group)
        return layout

    @staticmethod
    def _table(columns: list[str]) -> QTableWidget:
        table = QTableWidget(0, len(columns))
        table.setProperty("robotTable", True)
        table.setHorizontalHeaderLabels(columns)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().hide()
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setAlternatingRowColors(True)
        table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        return table

    @staticmethod
    def _fill_table(table: QTableWidget, rows) -> None:
        table.setRowCount(len(rows))
        for row, values in enumerate(rows):
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row, column, item)

    def _prediction_rows(self) -> list[list[str]]:
        return [
            [point["id"], "—", "—", "—", "—", "待预测"]
            for point in self.processing_points
        ]

    def _show_point_editor(self) -> None:
        dialog = PointEditor(self)
        fit_dialog(dialog, 820, 460)
        dialog.exec()

    def _show_records(self, title: str, columns: list[str], hint: str, rows=()) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.addWidget(self._note(hint))
        table = self._table(columns)
        self._fill_table(table, rows)
        layout.addWidget(table, 1)
        layout.addWidget(
            self._note("预测结果待评估" if rows else "暂无记录 · 数据接入后显示")
        )
        layout.addWidget(
            self._button("关闭", dialog.reject), 0, Qt.AlignmentFlag.AlignRight
        )
        fit_dialog(dialog, 820, 420)
        dialog.exec()

    def _show_points(self) -> None:
        self._show_records(
            "加工点位预测明细",
            PREDICTION_COLUMNS,
            "根据当前评估结果预测各加工点位的定位精度；不表示已在这些点位完成实测。",
            self._prediction_rows(),
        )

    def _show_history(self) -> None:
        self._show_records(
            "检测历史",
            ["测量批次", "采集时间", "基准版本", "手眼参数版本", "评估结论"],
            "回看各期检测及其手眼参数、基准来源。",
        )

    def _show_alarms(self) -> None:
        self._show_records(
            "报警记录",
            ["报警时间", "测量批次", "超限方向", "偏移 / 阈值", "处理状态"],
            "评估超出阈值后，记录超限项及处理过程。",
        )

    def _show_maintenance(self) -> None:
        self._show_records(
            "维护记录",
            ["维护时间", "关联报警", "维护内容", "处理人员", "复测批次"],
            "关联检查、维修和复测结果；本轮只展示记录结构。",
        )

    def _show_settings(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("定位精度阈值")
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        layout.addWidget(self._note("布局预览：以下输入仅供查看，关闭后不保存。"))
        form = QFormLayout()
        form.setSpacing(12)
        for name, placeholder in (
            ("X 方向阈值 / mm", "未设置"),
            ("Y 方向阈值 / mm", "未设置"),
            ("Z 方向阈值 / mm", "未设置"),
            ("距离阈值 / mm", "未设置"),
        ):
            field = QLineEdit()
            field.setProperty("robotInput", True)
            field.setPlaceholderText(placeholder)
            field.setAccessibleName(name)
            form.addRow(name, field)
        layout.addLayout(form)
        layout.addWidget(self._note("XYZ 默认按机器人基坐标系表达，距离为位置偏移量。"))
        layout.addStretch()
        layout.addWidget(
            self._button("关闭", dialog.reject), 0, Qt.AlignmentFlag.AlignRight
        )
        fit_dialog(dialog, 540, 380)
        dialog.exec()


class PointEditor(QDialog):
    """加工点位的界面草稿，应用后仅保存在当前会话。"""

    def __init__(self, page: RobotPositionPage) -> None:
        super().__init__(page)
        self.page = page
        self.setWindowTitle("加工点位管理")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)
        layout.addWidget(
            page._note("双击修改编号和理论坐标，可先留空坐标。配置仅保留在当前会话。")
        )
        actions = QHBoxLayout()
        for title, callback in (
            ("新增点位", self._add_point),
            ("删除点位", self._delete_points),
            ("上移", lambda: self._move_point(-1)),
            ("下移", lambda: self._move_point(1)),
            ("顺序编号", self._renumber),
        ):
            actions.addWidget(page._button(title, callback))
        actions.addStretch()
        layout.addLayout(actions)
        self.table = page._table(["编号", "理论 X / mm", "理论 Y / mm", "理论 Z / mm"])
        self.table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self.table.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        page._fill_table(
            self.table,
            [
                [point["id"]]
                + [
                    "" if point[axis] is None else f"{point[axis]:g}"
                    for axis in ("x", "y", "z")
                ]
                for point in page.processing_points
            ],
        )
        layout.addWidget(self.table, 1)
        self.error_label = page._note("")
        self.error_label.setProperty("validationError", True)
        layout.addWidget(self.error_label)
        footer = QHBoxLayout()
        footer.addWidget(
            page._note("加工点位用于预测，与拍摄靶标的观测点位分别管理。"), 1
        )
        footer.addWidget(page._button("取消", self.reject))
        footer.addWidget(page._button("应用", self._apply, True))
        layout.addLayout(footer)

    def _add_point(self) -> None:
        identifiers = {
            self.table.item(row, 0).text() for row in range(self.table.rowCount())
        }
        number = 1
        while f"P{number:03}" in identifiers:
            number += 1
        row = self.table.rowCount()
        self.table.insertRow(row)
        for column, value in enumerate((f"P{number:03}", "", "", "")):
            self.table.setItem(row, column, QTableWidgetItem(value))
        self.table.selectRow(row)
        self.page.append_log(f"点位草稿已新增 P{number:03}，等待应用。")

    def _delete_points(self) -> None:
        rows = sorted(
            {item.row() for item in self.table.selectionModel().selectedRows()},
            reverse=True,
        )
        if not rows:
            self.page.append_log("删除点位：请先选择点位。", "WARN")
            return
        for row in rows:
            self.table.removeRow(row)
        self.page.append_log(f"点位草稿已删除 {len(rows)} 个点位，等待应用。")

    def _move_point(self, offset: int) -> None:
        row = self.table.currentRow()
        target = row + offset
        if row < 0 or not 0 <= target < self.table.rowCount():
            return
        for column in range(self.table.columnCount()):
            item = self.table.takeItem(row, column)
            other = self.table.takeItem(target, column)
            self.table.setItem(row, column, other)
            self.table.setItem(target, column, item)
        self.table.selectRow(target)
        self.page.append_log("点位草稿的顺序已调整，等待应用。")

    def _renumber(self) -> None:
        for row in range(self.table.rowCount()):
            self.table.item(row, 0).setText(f"P{row + 1:03}")
        self.page.append_log("点位草稿已按当前行顺序重新编号，等待应用。")

    def _apply(self) -> None:
        points = []
        identifiers = set()
        for row in range(self.table.rowCount()):
            identifier = self.table.item(row, 0).text().strip()
            if not identifier or identifier in identifiers:
                self._report_error(f"第 {row + 1} 行编号为空或重复，请修改编号。")
                return
            identifiers.add(identifier)
            point = {"id": identifier}
            for column, axis in enumerate(("x", "y", "z"), 1):
                text = self.table.item(row, column).text().strip()
                try:
                    value = float(text) if text else None
                except ValueError:
                    self._report_error(
                        f"点位 {identifier} 的 {axis.upper()} 坐标不是有效数字。"
                    )
                    return
                if value is not None and not isfinite(value):
                    self._report_error(
                        f"点位 {identifier} 的 {axis.upper()} 坐标必须是有限数字。"
                    )
                    return
                point[axis] = value
            points.append(point)
        self.page.processing_points = points
        self.page.point_count_label.setText(f"加工点位：{len(points)} 个")
        self.page._fill_table(self.page.point_table, self.page._prediction_rows())
        self.page.append_log(
            f"加工点位已应用：{len(points)} 个，仅保留在当前会话；预测结果待评估。"
        )
        self.accept()

    def _report_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.page.append_log(message, "ERROR")


class PositionTrendChart(QWidget):
    """只绘制传入的历史结果，不计算视觉误差、距离或精度指标。"""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumHeight(150)
        self.mode = "xyz"
        self.history = []
        self.setAccessibleName("多次评估的定位精度趋势图")

    def set_history(self, records: list[dict]) -> None:
        """每条记录含 label、X、Y、Z、distance；未计算的指标为 None。"""
        self.history = records
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self.font())
        colors = DISPLAY["colors"]
        painter.fillRect(self.rect(), QColor(colors["background"]))
        metrics = painter.fontMetrics()
        line_height = metrics.height()
        margin = metrics.horizontalAdvance("-0.000") + line_height / 2
        plot = QRectF(
            margin,
            line_height / 2,
            self.width() - margin - line_height,
            self.height() - line_height * 2,
        )
        painter.setPen(QPen(QColor(colors["border"]), 1, Qt.PenStyle.DashLine))
        for step in range(5):
            y = plot.top() + plot.height() * step / 4
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        painter.setPen(QPen(QColor(colors["placeholder"]), 1))
        painter.drawLine(plot.bottomLeft(), plot.topLeft())
        painter.drawLine(plot.bottomLeft(), plot.bottomRight())
        axes = ("distance",) if self.mode == "distance" else ("X", "Y", "Z")
        values = [
            record[axis]
            for record in self.history
            for axis in axes
            if record.get(axis) is not None
        ]
        if not values:
            painter.setPen(QColor(colors["inactive"]))
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, "暂无历史评估记录")
            return
        low, high = min(0, min(values)), max(0, max(values))
        if low == high:
            high = low + 1
        painter.setPen(QColor(colors["inactive"]))
        for step in range(5):
            value = high - (high - low) * step / 4
            y = plot.top() + plot.height() * step / 4
            painter.drawText(
                QRectF(0, y - line_height / 2, margin - 6, line_height),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                f"{value:.3f}",
            )
        for index, alignment in (
            (0, Qt.AlignmentFlag.AlignLeft),
            (-1, Qt.AlignmentFlag.AlignRight),
        ):
            painter.drawText(
                QRectF(plot.left(), plot.bottom() + 2, plot.width(), line_height),
                alignment | Qt.AlignmentFlag.AlignVCenter,
                str(self.history[index]["label"]),
            )
        color_keys = {
            "X": "action",
            "Y": "axis_y",
            "Z": "axis_z",
            "distance": "distance",
        }
        for axis in axes:
            painter.setPen(
                QPen(QColor(colors[color_keys[axis]]), max(1, line_height / 9))
            )
            path = QPainterPath()
            connected = False
            for index, record in enumerate(self.history):
                value = record.get(axis)
                if value is None:
                    connected = False
                    continue
                x = (
                    plot.center().x()
                    if len(self.history) == 1
                    else plot.left() + plot.width() * index / (len(self.history) - 1)
                )
                point = QPointF(
                    x, plot.bottom() - plot.height() * (value - low) / (high - low)
                )
                if connected:
                    path.lineTo(point)
                else:
                    path.moveTo(point)
                connected = True
                painter.drawEllipse(point, line_height / 8, line_height / 8)
            painter.drawPath(path)
