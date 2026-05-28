from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import yaml
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QLabel, QSizePolicy
from yourdfpy import URDF


DEFAULT_JOINT_DEGREES = (0.0, -58.0, 82.0, -112.0, -90.0, 0.0)
SCENE_BACKGROUND_RGB = np.array([248, 251, 255], dtype=np.uint8)
SCENE_GRID_RGB = np.array([224, 233, 244], dtype=np.uint8)
SCENE_BORDER_RGB = np.array([199, 213, 232], dtype=np.uint8)

DEFAULT_RENDER_COLORS = {
    "long_link": (0.94, 0.95, 0.92),
    "joint": (0.43, 0.46, 0.44),
    "connector": (0.43, 0.46, 0.44),
    "cap": (0.62, 0.84, 0.96),
    "silver": (0.88, 0.89, 0.86),
}
# TODO: 旧方案待删除 — 验证通过后移除 CAP_GEOMETRY_NAMES
CAP_GEOMETRY_NAMES = {
    "shoulder.dae_2",
    "wrist1.dae_2",
    "wrist2.dae_2",
    "wrist3.dae",
}


class RobotSimulationWidget(QLabel):
    """URDF renderer backed by yourdfpy kinematics and PyVista off-screen rendering."""

    def __init__(self, project_root: str | Path | None = None) -> None:
        super().__init__()
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self._render_colors = self._load_render_colors()
        self._mesh_color_map = self._load_mesh_color_map()
        self.setObjectName("robot_simulation_widget")
        self.setMinimumSize(520, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("Waiting for UR10 model")
        self._urdf_path: Path | None = None
        self._urdf_model: URDF | None = None
        self._joint_names: list[str] = []
        self._joint_degrees = list(DEFAULT_JOINT_DEGREES)
        self._camera_target = np.array([0.0, 0.0, 0.55], dtype=float)
        self._camera_distance = 2.7
        self._camera_yaw_deg = 75.0
        self._camera_pitch_deg = 22.0
        self._visual_shape_colors: list[tuple[float, float, float]] = []
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.timeout.connect(self.render_scene)

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)

    @property
    def joint_degrees(self) -> list[float]:
        return list(self._joint_degrees)

    @property
    def visual_shape_colors(self) -> list[tuple[float, float, float]]:
        return list(self._visual_shape_colors)

    def load_robot(self, urdf_path: str | Path) -> list[str]:
        path = Path(urdf_path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        if path.suffix.lower() != ".urdf":
            raise ValueError(f"PyVista renderer currently expects .urdf, got {path.suffix}")

        self._prepare_collada_imports()
        self._urdf_model = URDF.load(str(path))
        self._urdf_path = path
        self._joint_names = list(self._urdf_model.actuated_joint_names[:6])
        self.set_joint_angles(DEFAULT_JOINT_DEGREES)
        return self.joint_names

    def set_joint_angles(self, joint_degrees: list[float] | tuple[float, ...]) -> None:
        values = list(joint_degrees)[: len(self._joint_names)]
        self._joint_degrees = values + [0.0] * max(0, 6 - len(values))
        if self._urdf_model is None:
            return

        joint_count = self._urdf_model.num_dofs
        configuration = np.zeros(joint_count, dtype=float)
        for index, angle_deg in enumerate(values[:joint_count]):
            configuration[index] = math.radians(float(angle_deg))
        if joint_count:
            self._urdf_model.update_cfg(configuration)
        self._fit_camera_to_robot()
        self.render_scene()

    def reset_home_pose(self) -> None:
        self.set_joint_angles(DEFAULT_JOINT_DEGREES)

    def reset_camera_to_fit(self) -> None:
        self._fit_camera_to_robot()
        self.render_scene()

    def render_scene(self) -> None:
        if self._urdf_model is None or not self._urdf_model.scene.geometry:
            self.update()
            return

        width = max(640, self.width())
        height = max(400, self.height())
        image_array = self._render_pyvista_scene(width, height)
        self._style_background_pixels(image_array)

        image = QImage(
            image_array.data,
            width,
            height,
            3 * width,
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image)
        self.setPixmap(
            pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._render_timer.start(100)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        if self._urdf_model is not None and self.pixmap() is not None:
            super().paintEvent(event)
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(6, 6, -6, -6)
        painter.fillRect(rect, QColor("#f8fbff"))
        painter.setPen(QPen(QColor("#e0e9f4"), 1))
        for row in range(0, rect.height(), 42):
            painter.drawLine(rect.left(), rect.top() + row, rect.right(), rect.top() + row)
        for col in range(0, rect.width(), 42):
            painter.drawLine(rect.left() + col, rect.top(), rect.left() + col, rect.bottom())
        painter.setPen(QPen(QColor("#8b9bb2"), 1))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Waiting for UR10 model")
        painter.drawRect(rect)

    def _prepare_collada_imports(self) -> None:
        import six  # noqa: F401

        for importer in sys.meta_path:
            if type(importer).__name__ == "_SixMetaPathImporter" and not hasattr(importer, "_path"):
                importer._path = None
        import dateutil.parser  # noqa: F401

    def _render_pyvista_scene(self, width: int, height: int) -> np.ndarray:
        plotter = pv.Plotter(off_screen=True, window_size=(width, height))
        plotter.set_background("#f8fbff")
        plotter.enable_anti_aliasing("ssaa")
        plotter.remove_all_lights()
        plotter.add_light(pv.Light(position=(2.5, -3.0, 4.0), focal_point=(0.0, 0.0, 0.45), intensity=0.95))
        plotter.add_light(pv.Light(position=(-2.5, 2.0, 2.2), focal_point=(0.0, 0.0, 0.45), intensity=0.35))

        self._visual_shape_colors = []
        for mesh, transform, color in self._iter_visual_meshes():
            polydata = self._to_polydata(mesh, transform)
            if polydata.n_points == 0 or polydata.n_cells == 0:
                continue
            self._visual_shape_colors.append(color)
            plotter.add_mesh(
                polydata,
                color=color,
                smooth_shading=True,
                split_sharp_edges=True,
                pbr=False,
                ambient=0.52,
                diffuse=0.76,
                specular=0.58 if color != self._render_colors["connector"] else 0.24,
                specular_power=42,
            )

        camera_position = self._camera_eye_position()
        plotter.camera_position = (
            tuple(camera_position),
            tuple(self._camera_target),
            (0.0, 0.0, 1.0),
        )
        plotter.camera.zoom(0.76)
        plotter.show(auto_close=False, interactive=False)
        image = np.asarray(plotter.screenshot(return_img=True), dtype=np.uint8).copy()
        plotter.close()
        return image[:, :, :3]

    def _iter_visual_meshes(self):
        assert self._urdf_model is not None
        scene = self._urdf_model.scene
        for node_name in scene.graph.nodes_geometry:
            transform, geometry_name = scene.graph.get(node_name)
            mesh = scene.geometry.get(geometry_name)
            if mesh is None:
                continue
            yield mesh, np.asarray(transform, dtype=float), self._resolve_mesh_color(mesh, geometry_name)

    def _to_polydata(self, mesh, transform: np.ndarray) -> pv.PolyData:
        vertices = np.asarray(mesh.vertices, dtype=float)
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if vertices.size == 0 or faces.size == 0:
            return pv.PolyData()
        face_data = np.column_stack((np.full(len(faces), 3, dtype=np.int64), faces)).ravel()
        polydata = pv.PolyData(vertices, face_data)
        polydata.transform(transform, inplace=True)
        return polydata

    def _load_render_colors(self) -> dict[str, tuple[float, float, float]]:
        colors = dict(DEFAULT_RENDER_COLORS)
        theme_path = self.project_root / "config" / "theme.yaml"
        if not theme_path.exists():
            return colors
        try:
            data = yaml.safe_load(theme_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return colors
        configured = data.get("robot_render_colors", {}) if isinstance(data, dict) else {}
        if not isinstance(configured, dict):
            return colors
        for key in colors:
            parsed = self._parse_rgb(configured.get(key))
            if parsed is not None:
                colors[key] = parsed
        return colors

    def _load_mesh_color_map(self) -> dict[str, str]:
        """从 theme.yaml 加载 mesh_color_map 配置（新方案：配置驱动配色）。"""
        theme_path = self.project_root / "config" / "theme.yaml"
        if not theme_path.exists():
            return {}
        try:
            data = yaml.safe_load(theme_path.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
        if not isinstance(data, dict):
            return {}
        mapping = data.get("mesh_color_map", {})
        if not isinstance(mapping, dict):
            return {}
        return {str(k): str(v) for k, v in mapping.items()}

    def _parse_rgb(self, value: object) -> tuple[float, float, float] | None:
        if isinstance(value, str) and value.startswith("#") and len(value) == 7:
            try:
                return tuple(int(value[index : index + 2], 16) / 255.0 for index in (1, 3, 5))
            except ValueError:
                return None
        if isinstance(value, (list, tuple)) and len(value) == 3:
            try:
                values = [float(item) for item in value]
            except (TypeError, ValueError):
                return None
            if all(0.0 <= item <= 1.0 for item in values):
                return tuple(values)
            if all(0.0 <= item <= 255.0 for item in values):
                return tuple(item / 255.0 for item in values)
        return None

    def _resolve_mesh_color(self, mesh, geometry_name: str = "") -> tuple[float, float, float]:
        # 新方案：配置驱动的显式查表，优先使用
        category = self._mesh_color_map.get(geometry_name)
        if category and category in self._render_colors:
            return self._render_colors[category]

        # TODO: 旧方案待删除 — start
        # 以下启发式逻辑作为 fallback，待可视化验证通过后删除
        material = getattr(getattr(mesh, "visual", None), "material", None)
        raw_color = getattr(material, "main_color", None)
        if raw_color is None:
            return self._render_colors["silver"]

        rgb = np.asarray(raw_color[:3], dtype=float) / 255.0
        extents = np.ptp(np.asarray(mesh.vertices, dtype=float), axis=0)
        sorted_extents = np.sort(np.maximum(extents, 1e-6))
        elongated = sorted_extents[-1] / sorted_extents[-2] > 1.8
        blueish = rgb[2] > rgb[0] * 1.18 and rgb[1] > rgb[0] * 1.10
        brightness = float(rgb.mean())

        if geometry_name in CAP_GEOMETRY_NAMES:
            return self._render_colors["cap"]
        if elongated and brightness < 0.28:
            return self._render_colors["connector"]
        if elongated:
            return self._render_colors["long_link"]
        if blueish:
            return self._render_colors["cap"]
        if brightness < 0.58:
            return self._render_colors["joint"]
        if brightness > 0.58:
            return self._render_colors["long_link"]
        return self._render_colors["joint"]
        # TODO: 旧方案待删除 — end

    def _fit_camera_to_robot(self) -> None:
        if self._urdf_model is None or not self._urdf_model.scene.geometry:
            return
        points = []
        for mesh, transform, _ in self._iter_visual_meshes():
            vertices = np.asarray(mesh.vertices, dtype=float)
            if vertices.size == 0:
                continue
            homogeneous = np.column_stack((vertices, np.ones(len(vertices))))
            points.append((transform @ homogeneous.T).T[:, :3])
        if not points:
            return
        all_points = np.vstack(points)
        lower = all_points.min(axis=0)
        upper = all_points.max(axis=0)
        center = (lower + upper) / 2.0
        extent = np.maximum(upper - lower, 0.05)
        self._camera_target = center + np.array([0.0, 0.0, 0.05])
        self._camera_distance = float(max(2.35, np.linalg.norm(extent) * 1.85))

    def _camera_eye_position(self) -> np.ndarray:
        yaw = math.radians(self._camera_yaw_deg)
        pitch = math.radians(self._camera_pitch_deg)
        horizontal_distance = self._camera_distance * math.cos(pitch)
        return np.array(
            [
                self._camera_target[0] + horizontal_distance * math.cos(yaw),
                self._camera_target[1] + horizontal_distance * math.sin(yaw),
                self._camera_target[2] + self._camera_distance * math.sin(pitch),
            ],
            dtype=float,
        )

    def _style_background_pixels(self, image_array: np.ndarray) -> None:
        background_mask = np.all(image_array[:, :, :3] >= 245, axis=2)
        image_array[background_mask, :3] = SCENE_BACKGROUND_RGB

        height, width = background_mask.shape
        for row in range(0, height, 42):
            row_slice = background_mask[row : row + 1, :]
            image_array[row : row + 1, :, :3][row_slice] = SCENE_GRID_RGB
        for col in range(0, width, 42):
            col_slice = background_mask[:, col : col + 1]
            image_array[:, col : col + 1, :3][col_slice] = SCENE_GRID_RGB

        image_array[0:1, :, :3] = SCENE_BORDER_RGB
        image_array[-1:, :, :3] = SCENE_BORDER_RGB
        image_array[:, 0:1, :3] = SCENE_BORDER_RGB
        image_array[:, -1:, :3] = SCENE_BORDER_RGB
