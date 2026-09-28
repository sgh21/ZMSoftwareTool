"""相机、手眼及棋盘参数的数值表单；持久化仍由定位服务负责。"""

from copy import deepcopy
from math import isfinite

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QScrollArea, QTabWidget, QVBoxLayout, QWidget,
)

from core.algorithms.pose_fields import rotation_from_rpy_degrees, rotation_to_rpy_degrees


DISTORTION_NAMES = (
    ("k1", "径向畸变 k1"), ("k2", "径向畸变 k2"),
    ("p1", "切向畸变 p1"), ("p2", "切向畸变 p2"),
    ("k3", "径向畸变 k3"), ("k4", "径向畸变 k4"),
    ("k5", "径向畸变 k5"), ("k6", "径向畸变 k6"),
    ("s1", "薄棱镜系数 s1"), ("s2", "薄棱镜系数 s2"),
    ("s3", "薄棱镜系数 s3"), ("s4", "薄棱镜系数 s4"),
    ("tau_x", "传感器倾斜 τx / rad"), ("tau_y", "传感器倾斜 τy / rad"),
)
HAND_EYE_FIELDS = ("tx", "ty", "tz", "roll", "pitch", "yaw")
REFERENCE_FIELDS = ("reference_roll", "reference_pitch", "reference_yaw")
TARGET_FIELDS = ("target_tx", "target_ty", "target_tz", "target_roll", "target_pitch", "target_yaw")


class RobotPositionParametersDialog(QDialog):
    def __init__(self, service, parent=None):
        super().__init__(parent)
        self.service = service
        self.original = deepcopy(service.parameters)
        self.fields = {}
        self.original_values = {}
        self.setWindowTitle("视觉与手眼参数")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(14)
        self.tabs = QTabWidget()
        self.tabs.setObjectName("RobotParameterTabs")
        layout.addWidget(self.tabs, 1)
        self._camera_tab()
        self._hand_eye_tab()
        self._board_tab()
        self.original_text = {name: field.text() for name, field in self.fields.items()}
        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setProperty("validationError", True)
        layout.addWidget(self.error_label)
        actions = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        for role, text in ((QDialogButtonBox.StandardButton.Save, "保存参数"),
                           (QDialogButtonBox.StandardButton.Cancel, "取消")):
            button = actions.button(role)
            button.setText(text)
            button.setProperty("robotAction", True)
            button.setProperty("primary", role == QDialogButtonBox.StandardButton.Save)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
        actions.accepted.connect(self._save)
        actions.rejected.connect(self.reject)
        layout.addWidget(actions)

    def _tab(self, title, *, scroll=False):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        if scroll:
            container = QScrollArea()
            container.setWidgetResizable(True)
            container.setFrameShape(QFrame.Shape.NoFrame)
            container.setWidget(page)
            self.tabs.addTab(container, title)
        else:
            self.tabs.addTab(page, title)
        return layout

    @staticmethod
    def _note(layout, text):
        label = QLabel(text)
        label.setProperty("robotNote", True)
        label.setWordWrap(True)
        layout.addWidget(label)

    def _section(self, layout, title, note):
        label = QLabel(title)
        label.setProperty("robotSectionTitle", True)
        layout.addWidget(label)
        self._note(layout, note)

    def _fields(self, layout, entries):
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(12)
        widgets = []
        for index, (name, title, value) in enumerate(entries):
            row, column = index // 2, (index % 2) * 2
            label = QLabel(title)
            field = QLineEdit("" if value is None else format(float(value), ".12g"))
            field.setProperty("robotInput", True)
            field.setObjectName(f"parameter_{name}")
            field.setAccessibleName(title)
            field.setMinimumWidth(100)
            label.setBuddy(field)
            self.fields[name] = field
            self.original_values[name] = value
            grid.addWidget(label, row, column)
            grid.addWidget(field, row, column + 1)
            widgets.append((label, field))
        grid.setColumnStretch(1, 1)
        grid.setColumnStretch(3, 1)
        layout.addLayout(grid)
        return widgets

    def _camera_tab(self):
        layout = self._tab("相机参数", scroll=True)
        self._section(layout, "相机内参", "fx、fy 为焦距，cx、cy 为主点位置，单位为像素。未标定时四项全部留空。")
        camera = self.original.get("camera_matrix")
        self._fields(layout, [
            (name, title, camera[row][column] if camera is not None else None)
            for name, title, row, column in (
                ("fx", "水平焦距 fx / px", 0, 0), ("fy", "垂直焦距 fy / px", 1, 1),
                ("cx", "主点横坐标 cx / px", 0, 2), ("cy", "主点纵坐标 cy / px", 1, 2),
            )
        ])
        self._section(layout, "镜头畸变", "按相机标定结果填写，常用为 5 项。")
        row = QHBoxLayout()
        row.addWidget(QLabel("畸变参数项数"))
        self.distortion_count = QComboBox()
        self.distortion_count.setProperty("robotInput", True)
        self.distortion_count.setAccessibleName("畸变参数项数")
        for count in (0, 4, 5, 8, 12, 14):
            self.distortion_count.addItem("不使用畸变" if count == 0 else f"{count} 项", count)
        row.addWidget(self.distortion_count)
        row.addStretch()
        layout.addLayout(row)
        distortion = self.original.get("dist_coeffs", [])
        self.distortion_widgets = self._fields(layout, [
            (name, title, distortion[index] if index < len(distortion) else 0)
            for index, (name, title) in enumerate(DISTORTION_NAMES)
        ])
        self.distortion_count.setCurrentIndex(self.distortion_count.findData(len(distortion)))
        self.distortion_count.currentIndexChanged.connect(self._show_distortion_fields)
        self._show_distortion_fields()
        layout.addStretch()

    def _show_distortion_fields(self):
        count = self.distortion_count.currentData()
        for index, widgets in enumerate(self.distortion_widgets):
            for widget in widgets:
                widget.setVisible(index < count)

    def _hand_eye_tab(self):
        layout = self._tab("手眼参数")
        self._note(layout, "相机相对于被评估末端的安装位姿。未标定时六项全部留空。")
        if self.original.get("end_frame"):
            self._note(layout, f"当前被评估末端坐标系：{self.original['end_frame']}")
        hand_eye = self.original.get("hand_eye")
        translation = [None] * 3
        rpy = [None] * 3
        if hand_eye is not None:
            transform = np.asarray(hand_eye)
            translation = transform[:3, 3]
            rpy = rotation_to_rpy_degrees(transform[:3, :3])
        self._section(layout, "平移", "相机原点在被评估末端坐标系中的位置，单位为毫米。")
        self._fields(layout, [(name, f"{axis} 方向 / mm", value)
                              for name, axis, value in zip(HAND_EYE_FIELDS[:3], "XYZ", translation)])
        self._section(layout, "旋转", "Roll 绕 X 轴，Pitch 绕 Y 轴，Yaw 绕 Z 轴，单位为度。")
        self._fields(layout, [(name, f"{name.title()} / °", value)
                              for name, value in zip(HAND_EYE_FIELDS[3:], rpy)])
        self._note(layout, "按固定坐标轴 X → Y → Z 顺序旋转。")
        layout.addStretch()

    def _board_tab(self):
        layout = self._tab("标定板参数", scroll=True)
        self._section(layout, "标定板几何", "选择实际使用的板型，按标定板尺寸与编码填写。")
        row = QHBoxLayout()
        row.addWidget(QLabel("标定板类型"))
        self.board_type = QComboBox()
        self.board_type.setProperty("robotInput", True)
        self.board_type.setAccessibleName("标定板类型")
        self.board_type.addItem("普通棋盘", "checkerboard")
        self.board_type.addItem("ChArUco", "charuco")
        self.board_type.setCurrentIndex(self.board_type.findData(self.original.get("board_type", "checkerboard")))
        row.addWidget(self.board_type)
        row.addStretch()
        layout.addLayout(row)
        self.checkerboard_panel = QWidget()
        checkerboard_layout = QVBoxLayout(self.checkerboard_panel)
        checkerboard_layout.setContentsMargins(0, 0, 0, 0)
        checkerboard_layout.setSpacing(12)
        self._note(checkerboard_layout, "填写内部交点的列数和行数，不是方格数量。")
        columns, rows = self.original.get("board_grid", [None, None])
        self._fields(checkerboard_layout, [
            ("board_columns", "内角点列数", columns), ("board_rows", "内角点行数", rows),
        ])
        layout.addWidget(self.checkerboard_panel)
        self.charuco_panel = QWidget()
        charuco_layout = QVBoxLayout(self.charuco_panel)
        charuco_layout.setContentsMargins(0, 0, 0, 0)
        charuco_layout.setSpacing(12)
        self._note(charuco_layout, "ChArUco 填写方格总数；方格列数、行数均包含外缘方格。")
        charuco = self.original.get("charuco", {})
        columns, rows = charuco.get("squares_xy", [None, None])
        self._fields(charuco_layout, [
            ("charuco_columns", "方格列数", columns), ("charuco_rows", "方格行数", rows),
            ("marker_length", "标记边长 / mm", charuco.get("marker_length_mm")),
        ])
        dictionary_row = QHBoxLayout()
        dictionary_row.addWidget(QLabel("标记字典"))
        self.charuco_dictionary = QComboBox()
        self.charuco_dictionary.setProperty("robotInput", True)
        self.charuco_dictionary.setAccessibleName("ChArUco 标记字典")
        dictionaries = [f"DICT_{bits}X{bits}_{count}" for bits in (4, 5, 6, 7)
                        for count in (50, 100, 250, 1000)]
        dictionary = charuco.get("dictionary", "DICT_5X5_1000")
        if dictionary not in dictionaries:
            dictionaries.append(dictionary)
        self.charuco_dictionary.addItems(dictionaries)
        self.charuco_dictionary.setCurrentText(dictionary)
        dictionary_row.addWidget(self.charuco_dictionary)
        dictionary_row.addStretch()
        charuco_layout.addLayout(dictionary_row)
        if charuco.get("marker_ids") is not None or "legacy_pattern" in charuco:
            self._note(charuco_layout, "已加载的标记编号与图案版本随参数保留。")
        layout.addWidget(self.charuco_panel)
        self._fields(layout, [
            ("square_size", "单个方格边长 / mm", self.original["square_size_mm"]),
            ("reprojection_limit", "重投影误差上限 / px", self.original.get("max_reprojection_error_px")),
        ])
        self.fields["reprojection_limit"].setPlaceholderText("留空表示不设上限")
        self._section(layout, "参考朝向（可选）", "普通棋盘相对于相机的 RPY 角，用于保持角点方向一致；未指定时三项全部留空。")
        rotation = self.original.get("reference_rotation")
        rpy = rotation_to_rpy_degrees(rotation) if rotation is not None else [None] * 3
        self._fields(layout, [(name, f"{axis} / °", value)
                              for name, axis, value in zip(REFERENCE_FIELDS, ("Roll", "Pitch", "Yaw"), rpy)])
        self._note(layout, "参考朝向也按固定坐标轴 X → Y → Z 顺序旋转。")
        self._section(layout, "固定靶标在基座中的位姿（可选）", "缺失时仍可进行相对监控；提供后用于绝对位置测量。平移单位为毫米，旋转使用固定轴 XYZ 的 RPY 角。")
        target = self.original.get("target_pose_base")
        translation = np.asarray(target)[:3, 3] if target is not None else [None] * 3
        target_rpy = rotation_to_rpy_degrees(np.asarray(target)[:3, :3]) if target is not None else [None] * 3
        self._fields(layout, [
            *[(name, f"基座 {axis} / mm", value) for name, axis, value
              in zip(TARGET_FIELDS[:3], "XYZ", translation)],
            *[(name, f"{axis} / °", value) for name, axis, value
              in zip(TARGET_FIELDS[3:], ("Roll", "Pitch", "Yaw"), target_rpy)],
        ])
        self.board_type.currentIndexChanged.connect(self._show_board_fields)
        self._show_board_fields()
        layout.addStretch()

    def _show_board_fields(self):
        charuco = self.board_type.currentData() == "charuco"
        self.charuco_panel.setVisible(charuco)
        self.checkerboard_panel.setVisible(not charuco)

    def _number(self, name):
        field = self.fields[name]
        try:
            value = float(self.original_values[name]) if self._unchanged((name,)) else float(field.text())
        except (TypeError, ValueError):
            raise ValueError(f"{field.accessibleName()}：请填写数值") from None
        if not isfinite(value):
            raise ValueError(f"{field.accessibleName()}：请填写有限数值")
        return value

    def _optional_group(self, names):
        if all(not self.fields[name].text().strip() for name in names):
            return None
        return [self._number(name) for name in names]

    def _unchanged(self, names):
        return all(self.fields[name].text().strip() == self.original_text[name] for name in names)

    def _pose(self, names, key):
        values = self._optional_group(names)
        if values is None:
            return None
        if self._unchanged(names):
            return self.original[key]
        original = self.original.get(key)
        pose = np.array(original, dtype=float) if original is not None else np.eye(4)
        if original is None or not self._unchanged(names[3:]):
            pose[:3, :3] = rotation_from_rpy_degrees(values[3:])
        pose[:3, 3] = values[:3]
        return pose.tolist()

    def _document(self):
        camera_fields = ("fx", "fy", "cx", "cy")
        intrinsics = self._optional_group(camera_fields)
        camera = None
        if intrinsics is not None:
            camera = np.array(self.original["camera_matrix"], dtype=float) if self.original.get("camera_matrix") is not None else np.eye(3)
            camera[0, 0], camera[1, 1], camera[0, 2], camera[1, 2] = intrinsics
            camera = camera.tolist()
        hand_eye = self._pose(HAND_EYE_FIELDS, "hand_eye")
        reference = self._optional_group(REFERENCE_FIELDS)
        if reference is not None:
            reference = (self.original["reference_rotation"] if self._unchanged(REFERENCE_FIELDS)
                         else rotation_from_rpy_degrees(reference).tolist())
        distortion = [self._number(name) for name, _ in DISTORTION_NAMES[:self.distortion_count.currentData()]]
        limit = self._number("reprojection_limit") if self.fields["reprojection_limit"].text().strip() else None
        document = {
            "camera_matrix": camera, "dist_coeffs": distortion, "hand_eye": hand_eye,
            "reference_rotation": reference,
            "square_size_mm": self._number("square_size"), "max_reprojection_error_px": limit,
            "length_unit": "mm", "transform_convention": "E_T_C",
        }
        board_type = self.board_type.currentData()
        if board_type == "charuco":
            document["charuco"] = {
                **deepcopy(self.original.get("charuco", {})),
                "squares_xy": [self._number("charuco_columns"), self._number("charuco_rows")],
                "marker_length_mm": self._number("marker_length"),
                "dictionary": self.charuco_dictionary.currentText(),
            }
        else:
            document["board_grid"] = [self._number("board_columns"), self._number("board_rows")]
        if board_type == "charuco" or "board_type" in self.original:
            document["board_type"] = board_type
        target_pose = self._pose(TARGET_FIELDS, "target_pose_base")
        if target_pose is not None or "target_pose_base" in self.original:
            document["target_pose_base"] = target_pose
        return document

    def _save(self):
        try:
            self.service.save_parameters(self._document())
        except (ValueError, OSError) as error:
            self.error_label.setText(str(error))
            return
        self.accept()
