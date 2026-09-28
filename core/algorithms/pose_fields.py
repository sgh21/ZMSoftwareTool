"""参数表单的 RPY 与旋转矩阵转换，固定轴 XYZ：R = Rz(yaw) Ry(pitch) Rx(roll)。"""

import numpy as np

from core.algorithms.position_monitoring import validate_transform


def rotation_from_rpy_degrees(rpy) -> np.ndarray:
    """将以度为单位的 [roll, pitch, yaw] 转为 3×3 旋转矩阵。"""
    message = "RPY 必须包含三个有限数值，单位为度"
    try:
        angles = np.asarray(rpy, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(message) from error
    if angles.shape != (3,) or not np.isfinite(angles).all():
        raise ValueError(message)
    roll, pitch, yaw = np.deg2rad(angles)
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def rotation_to_rpy_degrees(rotation) -> np.ndarray:
    """返回等价 RPY（度），pitch 在 ±90° 内；恰好 ±90° 时选取 roll=0。"""
    message = "旋转必须是有限数值的 3×3 旋转矩阵"
    try:
        matrix = np.asarray(rotation, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(message) from error
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError(message)
    transform = np.eye(4)
    transform[:3, :3] = matrix
    validate_transform(transform, "旋转")
    cos_pitch = np.hypot(matrix[0, 0], matrix[1, 0])
    pitch = np.arctan2(-matrix[2, 0], cos_pitch)
    if cos_pitch > 1e-10:
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-matrix[0, 1], matrix[1, 1])
    return np.rad2deg([roll, pitch, yaw])
