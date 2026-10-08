"""Nominal UR10 observation geometry, without Blender or file-writing side effects.

Public transforms use millimetres and map child coordinates into parent coordinates.
B is the UR controller base, E is tool0, and C uses OpenCV camera axes. Joint
angles are radians. Local Cartesian paths check kinematics and image bounds;
they are simulation plans, not collision-checked robot motion commands.
"""

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def rigid(rotation=None, translation=(0, 0, 0)):
    result = np.eye(4)
    if rotation is not None:
        result[:3, :3] = rotation
    result[:3, 3] = translation
    return result


def inverse(matrix):
    matrix = np.asarray(matrix, dtype=float)
    return rigid(matrix[:3, :3].T, -matrix[:3, :3].T @ matrix[:3, 3])


def skew(vector):
    x, y, z = vector
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


class AcquisitionGeometry:
    """Reuse the official URDF origins and axes exported for the Blender scene."""

    def __init__(self, scene_config, robot_description):
        self.config = scene_config
        self.robot_description = robot_description
        self.joints = []
        for joint in robot_description["joints"]:
            origin = joint["origin"]
            matrix = rigid(
                Rotation.from_euler("xyz", origin["rpy"]).as_matrix(),
                np.asarray(origin["xyz"]) * 1000,
            )
            self.joints.append((joint, matrix))
        # UR's base_link -> base is a half-turn around Z; origins coincide.
        self.B_T_base_link_mm = rigid(np.diag([-1.0, -1.0, 1.0]))
        self.W_T_B_mm = rigid(
            np.diag([-1.0, -1.0, 1.0]),
            np.asarray(scene_config["robot_base_xyz_m"]) * 1000,
        )
        camera = scene_config["camera"]
        mount_rotation = Rotation.from_euler(
            "z", camera["mount_z_rotation_deg"], degrees=True
        ).as_matrix()
        # Mount -> Blender camera is Rx(pi); Blender -> OpenCV is also Rx(pi).
        self.H_E_T_C_mm = rigid(
            mount_rotation,
            mount_rotation @ (np.asarray(camera["mount_to_optical_center_xyz_m"]) * 1000),
        )
        self.H_E_T_C_mm[np.abs(self.H_E_T_C_mm) < 1e-12] = 0
        self.resolution_px = np.asarray(camera["resolution_px"], dtype=int)
        focal_px = camera["focal_length_mm"] * 1000 / camera["pixel_pitch_um"]
        width_px, height_px = self.resolution_px
        # Retain the scene's declared ideal intrinsics: an assumed centred principal point.
        self.K_px = np.array([
            [focal_px, 0, width_px / 2],
            [0, focal_px, height_px / 2],
            [0, 0, 1],
        ])
        board = scene_config["charuco"]
        self.pattern_size_mm = (
            np.asarray(board["squares_xy"]) * board["square_length_m"] * 1000
        )
        self.plate_size_mm = np.asarray(scene_config["board_plate_size_m"][:2]) * 1000
        center_world = np.asarray(scene_config["board_centers_m"][0]) * 1000
        # build_scene.py places the printed plane 0.1 mm above the mount origin.
        center_world = center_world + [0, 0, 0.1]
        world_marker_center = rigid(np.diag([1.0, -1.0, -1.0]), center_world)
        self.B_T_marker_center_mm = inverse(self.W_T_B_mm) @ world_marker_center
        pw, ph = self.pattern_size_mm
        self.B_T_M_mm = self.B_T_marker_center_mm @ rigid(translation=(-pw / 2, -ph / 2, 0))
        self.nominal_q_rad = np.deg2rad(scene_config["robot_joint_angles_deg"])
        self.nominal_B_T_E_mm = self.fk(self.nominal_q_rad)

    @classmethod
    def from_root(cls, root=None):
        root = Path(__file__).resolve().parent if root is None else Path(root)
        config = json.loads((root / "scene_config.json").read_text(encoding="utf-8"))
        robot = json.loads((root / "assets/ur10/robot_description.json").read_text(encoding="utf-8"))
        return cls(config, robot)

    def _chain(self, q_rad):
        frames = {"base_link": self.B_T_base_link_mm}
        axes, origins = [], []
        joint_index = 0
        for joint, origin in self.joints:
            current = frames[joint["parent"]] @ origin
            if joint["type"] == "revolute":
                axis = np.asarray(joint["axis"])
                origins.append(current[:3, 3].copy())
                axes.append(current[:3, :3] @ axis)
                current = current @ rigid(
                    Rotation.from_rotvec(axis * q_rad[joint_index]).as_matrix()
                )
                joint_index += 1
            frames[joint["child"]] = current
        return frames["tool0"], np.asarray(origins), np.asarray(axes)

    def fk(self, q_rad):
        """Return B_T_tool0, translation in mm, for six joint angles in radians."""
        return self._chain(np.asarray(q_rad, dtype=float))[0]

    def ik(self, B_T_E_mm, seed_q_rad):
        """Find a nearby joint solution; reject unresolved or distant branches."""
        target = np.asarray(B_T_E_mm, dtype=float)
        seed = np.asarray(seed_q_rad, dtype=float)
        rotation_weight_mm = 400.0

        def residual(q):
            current = self.fk(q)
            return np.r_[
                current[:3, 3] - target[:3, 3],
                rotation_weight_mm * (current[:3, :3] - target[:3, :3]).ravel(),
            ]

        def jacobian(q):
            current, origins, axes = self._chain(q)
            position = np.cross(axes, current[:3, 3] - origins).T
            orientation = np.column_stack([
                (rotation_weight_mm * skew(axis) @ current[:3, :3]).ravel()
                for axis in axes
            ])
            return np.vstack((position, orientation))

        # Official UR10 joint limits are +/-2pi; the local box preserves a branch.
        lower = np.maximum(-2 * np.pi, seed - np.pi / 2)
        upper = np.minimum(2 * np.pi, seed + np.pi / 2)
        fit = least_squares(
            residual, seed, jac=jacobian, bounds=(lower, upper), max_nfev=50,
            ftol=1e-11, xtol=1e-11, gtol=1e-10,
        )
        actual = self.fk(fit.x)
        position_error = np.linalg.norm(actual[:3, 3] - target[:3, 3])
        rotation_error = Rotation.from_matrix(
            actual[:3, :3] @ target[:3, :3].T
        ).magnitude()
        if position_error > 1e-4 or rotation_error > 1e-6:
            raise ValueError(
                f"Local IK failed: position {position_error:.6g} mm, rotation {rotation_error:.6g} rad"
            )
        return fit.x

    def visibility(self, B_T_E_mm, margin_px=24, position_error_bound_mm=0):
        """Project the entire white plate; optionally bound any endpoint translation.

        A camera-coordinate cube encloses the stated translation-error ball.
        Checking all its projected vertices gives a conservative image margin.
        The board and tool orientation stay fixed during this translation check.
        """
        C_T_center = inverse(np.asarray(B_T_E_mm) @ self.H_E_T_C_mm) @ self.B_T_marker_center_mm
        width, height = self.plate_size_mm
        corners = np.array([
            [-width / 2, -height / 2, 0], [width / 2, -height / 2, 0],
            [width / 2, height / 2, 0], [-width / 2, height / 2, 0],
        ])
        camera_points = corners @ C_T_center[:3, :3].T + C_T_center[:3, 3]
        shifts = np.array(list(itertools.product((-1, 1), repeat=3))) * position_error_bound_mm
        bounded = (camera_points[:, None, :] + shifts).reshape(-1, 3)

        def project(points):
            pixels = points @ self.K_px.T
            return pixels[:, :2] / pixels[:, 2, None]

        corner_pixels = project(camera_points)
        bound_pixels = project(bounded)
        image_width, image_height = self.resolution_px
        # Match the scene's world_to_camera_view projection convention.
        minimum_margin = float(min(
            np.min(bound_pixels[:, 0]),
            np.min(bound_pixels[:, 1]),
            np.min(image_width - bound_pixels[:, 0]),
            np.min(image_height - bound_pixels[:, 1]),
        ))
        return {
            "plate_corners_px": corner_pixels.tolist(),
            "minimum_margin_px": minimum_margin,
            "required_margin_px": margin_px,
            "translation_error_bound_mm": position_error_bound_mm,
            "complete_plate_visible": bool(np.min(bounded[:, 2]) > 0 and minimum_margin >= margin_px),
        }

    def approach(
        self, B_T_E_mm, q_endpoint_rad, direction_B, distance_mm=20, step_mm=2,
        endpoint_offset_mm=0,
    ):
        """Return a constant-orientation straight approach, start -> endpoint.

        B_T_E_mm and q_endpoint_rad describe the ideal endpoint. The start stays
        ideal - distance*d; the actual end is ideal + endpoint_offset*d. The
        offset is signed millimetres along the base-frame unit direction d.
        IK is solved backward from the actual endpoint on the same local branch.
        """
        if distance_mm <= 0 or not 0 < step_mm <= 2:
            raise ValueError("Approach distance must be positive and path step must be in (0, 2] mm")
        target = np.asarray(B_T_E_mm, dtype=float)
        direction = np.asarray(direction_B, dtype=float)
        direction = direction / np.linalg.norm(direction)
        travel_distance = distance_mm + endpoint_offset_mm
        if travel_distance <= 0:
            raise ValueError("The actual endpoint must lie beyond the fixed approach start")
        actual = target.copy()
        actual[:3, 3] += endpoint_offset_mm * direction
        q_actual = (
            self.ik(actual, q_endpoint_rad) if endpoint_offset_mm != 0
            else np.asarray(q_endpoint_rad, dtype=float)
        )
        n_steps = int(np.ceil(travel_distance / step_mm))
        path_poses, path_q = [actual], [q_actual]
        for distance in np.linspace(0, travel_distance, n_steps + 1)[1:]:
            pose = actual.copy()
            pose[:3, 3] -= distance * direction
            q = self.ik(pose, path_q[-1])
            if np.max(np.abs(q - path_q[-1])) > np.deg2rad(5):
                raise ValueError("Approach changes a joint by more than 5 deg in one 2 mm interval")
            path_poses.append(pose)
            path_q.append(q)
        path_poses.reverse()
        path_q.reverse()
        return {
            "direction_B": direction.tolist(),
            "distance_mm": distance_mm,
            "endpoint_offset_mm": endpoint_offset_mm,
            "travel_distance_mm": travel_distance,
            "maximum_sample_spacing_mm": travel_distance / n_steps,
            "ideal_B_T_E_mm": target.tolist(),
            "actual_B_T_E_mm": actual.tolist(),
            "q_actual_rad": q_actual.tolist(),
            "start_B_T_E_mm": path_poses[0].tolist(),
            "approach_q_rad": path_q[0].tolist(),
            "path_B_T_E_mm": [pose.tolist() for pose in path_poses],
            "path_q_rad": [q.tolist() for q in path_q],
        }
