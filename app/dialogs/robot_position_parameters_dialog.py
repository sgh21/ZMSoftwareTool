"""相机、手眼及棋盘参数的数值表单；持久化仍由定位服务负责。"""

from copy import deepcopy
from math import isfinite

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QTabWidget, QVBoxLayout, QWidget,
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

    def _tab(self, title):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)
        self.tabs.addTab(page, title)
        return layout

    @staticmethod
    def _note(layout, text):
        label = QLabel(text)
        label.setProperty("robotNote", True)
        label.setWordWrap(True)
        layout.addWidget(label)

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
        layout = self._tab("相机参数")
        self._note(layout, "相机内参（像素）：fx、fy 为焦距，cx、cy 为主点位置。未标定时可将这四项全部留空。")
        camera = self.original.get("camera_matrix")
        self._fields(layout, [
            (name, title, camera[row][column] if camera is not None else None)
            for name, title, row, column in (
                ("fx", "水平焦距 fx / px", 0, 0), ("fy", "垂直焦距 fy / px", 1, 1),
                ("cx", "主点横坐标 cx / px", 0, 2), ("cy", "主点纵坐标 cy / px", 1, 2),
            )
        ])
        self._note(layout, "镜头畸变：按相机标定结果填写，常用为 5 项。")
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
        self._note(layout, "相机相对于被评估末端（TCP）的安装位姿。未标定时可将六项全部留空。")
        hand_eye = self.original.get("hand_eye")
        translation = [None] * 3
        rpy = [None] * 3
        if hand_eye is not None:
            transform = np.asarray(hand_eye)
            translation = transform[:3, 3]
            rpy = rotation_to_rpy_degrees(transform[:3, :3])
        self._note(layout, "平移：相机原点在 TCP 坐标系中的位置，单位毫米。")
        self._fields(layout, [(name, f"{axis} 方向 / mm", value)
                              for name, axis, value in zip(HAND_EYE_FIELDS[:3], "XYZ", translation)])
        self._note(layout, "旋转：Roll 绕 X 轴，Pitch 绕 Y 轴，Yaw 绕 Z 轴，单位为度。")
        self._fields(layout, [(name, f"{name.title()} / °", value)
                              for name, value in zip(HAND_EYE_FIELDS[3:], rpy)])
        self._note(layout, "按固定坐标轴 X → Y → Z 顺序旋转。")
        layout.addStretch()

    def _board_tab(self):
        layout = self._tab("标定板参数")
        self._note(layout, "棋盘格：填写黑白方格内部交点的列数和行数，不是方格数量。")
        columns, rows = self.original["board_grid"]
        self._fields(layout, [
            ("board_columns", "内角点列数", columns), ("board_rows", "内角点行数", rows),
            ("square_size", "单个方格边长 / mm", self.original["square_size_mm"]),
            ("reprojection_limit", "重投影误差上限 / px", self.original.get("max_reprojection_error_px")),
        ])
        self.fields["reprojection_limit"].setPlaceholderText("留空表示不设上限")
        self._note(layout, "参考朝向（可选）：标定板相对于相机的 RPY 角，用于保持棋盘角点方向一致；未指定时三项全部留空。")
        rotation = self.original.get("reference_rotation")
        rpy = rotation_to_rpy_degrees(rotation) if rotation is not None else [None] * 3
        self._fields(layout, [(name, f"{axis} / °", value)
                              for name, axis, value in zip(REFERENCE_FIELDS, ("Roll", "Pitch", "Yaw"), rpy)])
        self._note(layout, "参考朝向也按固定坐标轴 X → Y → Z 顺序旋转。")
        layout.addStretch()

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

    def _document(self):
        camera_fields = ("fx", "fy", "cx", "cy")
        intrinsics = self._optional_group(camera_fields)
        camera = None
        if intrinsics is not None:
            camera = np.array(self.original["camera_matrix"], dtype=float) if self.original.get("camera_matrix") is not None else np.eye(3)
            camera[0, 0], camera[1, 1], camera[0, 2], camera[1, 2] = intrinsics
            camera = camera.tolist()
        pose = self._optional_group(HAND_EYE_FIELDS)
        hand_eye = None
        if pose is not None:
            if self._unchanged(HAND_EYE_FIELDS):
                hand_eye = self.original["hand_eye"]
            else:
                hand_eye = np.eye(4)
                hand_eye[:3, :3] = rotation_from_rpy_degrees(pose[3:])
                hand_eye[:3, 3] = pose[:3]
                hand_eye = hand_eye.tolist()
        reference = self._optional_group(REFERENCE_FIELDS)
        if reference is not None:
            reference = (self.original["reference_rotation"] if self._unchanged(REFERENCE_FIELDS)
                         else rotation_from_rpy_degrees(reference).tolist())
        distortion = [self._number(name) for name, _ in DISTORTION_NAMES[:self.distortion_count.currentData()]]
        limit = self._number("reprojection_limit") if self.fields["reprojection_limit"].text().strip() else None
        return {
            "camera_matrix": camera, "dist_coeffs": distortion, "hand_eye": hand_eye,
            "reference_rotation": reference,
            "board_grid": [self._number("board_columns"), self._number("board_rows")],
            "square_size_mm": self._number("square_size"), "max_reprojection_error_px": limit,
            "length_unit": "mm", "transform_convention": "E_T_C",
        }

    def _save(self):
        try:
            self.service.save_parameters(self._document())
        except (ValueError, OSError) as error:
            self.error_label.setText(str(error))
            return
        self.accept()
