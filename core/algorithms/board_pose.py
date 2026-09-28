"""Checkerboard PnP and eye-in-hand calibration, with translations in mm.

The corner detection, centred board and IPPE/LM pipeline follow the original
master implementation in core/vision/pnp.py. No robot model or UI is required.
"""

from __future__ import annotations

import cv2
import numpy as np


def checkerboard_points(board_grid, square_size_mm):
    """Return centred object points; board_grid is (inner columns, inner rows)."""
    cols, rows = board_grid
    if cols < 2 or rows < 2 or square_size_mm <= 0:
        raise ValueError("棋盘内角点行列数至少为 2，格长必须大于 0。")
    points = np.zeros((cols * rows, 3), dtype=np.float64)
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    points[:, :2] -= [(cols - 1) / 2, (rows - 1) / 2]
    return points * float(square_size_mm)


def detect_board_corners(image, board_grid):
    """Return subpixel corners, preserving the detector's original order."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_FILTER_QUADS
    )
    found, corners = cv2.findChessboardCorners(gray, tuple(board_grid), flags)
    if not found:
        found, corners = cv2.findChessboardCornersSB(gray, tuple(board_grid))
    if not found:
        raise ValueError("未检测到完整棋盘格，请核对内角点行列数与图像。")
    corners = cv2.cornerSubPix(
        gray, np.asarray(corners, dtype=np.float32), (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3),
    )
    return corners.reshape(-1, 2).astype(np.float64)


def _rotation_angle(rotation):
    cosine = np.clip((np.trace(rotation) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(cosine)))


def solve_board_pose(
    image_points, camera_matrix, dist_coeffs, board_grid, square_size_mm,
    reference_rotation=None,
):
    """Solve C_T_M from known corners, with the board origin at its centre.

    A symmetric checkerboard has no physical first-corner identity. By default,
    the first corner is the endpoint with smaller image x+y. For a later view
    of the same target, reference_rotation chooses the nearer of the two 180°
    corner orders. It does not resolve arbitrary board rotations, and does not
    choose a poorer planar-PnP solution merely to match an expected orientation.
    """
    objects = checkerboard_points(board_grid, square_size_mm)
    points = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    if len(points) != len(objects):
        raise ValueError("角点数量与棋盘内角点行列数不匹配。")
    if points[0].sum() > points[-1].sum():
        points = points[::-1]
    points = np.ascontiguousarray(points)
    matrix = np.asarray(camera_matrix, dtype=np.float64).reshape(3, 3)
    distortion = np.asarray(dist_coeffs, dtype=np.float64).reshape(-1, 1)
    if not np.isfinite(matrix).all() or matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("相机矩阵必须有效，焦距必须为正。")

    found, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        objects, points, matrix, distortion, flags=cv2.SOLVEPNP_IPPE,
    )
    if not found:
        raise ValueError("棋盘 PnP 解算失败。")
    candidates = []
    for rvec, tvec in zip(rvecs, tvecs):
        rotation = cv2.Rodrigues(rvec)[0]
        if np.any((objects @ rotation.T + tvec.reshape(3))[:, 2] <= 0):
            continue
        projection = cv2.projectPoints(objects, rvec, tvec, matrix, distortion)[0]
        rms = np.sqrt(np.mean(np.sum((projection.reshape(-1, 2) - points) ** 2, axis=1)))
        candidates.append((float(rms), rvec, tvec))
    if not candidates:
        raise ValueError("PnP 未得到位于相机前方的有效棋盘位姿。")
    candidates.sort(key=lambda entry: entry[0])
    _, rvec, tvec = candidates[0]
    rvec, tvec = cv2.solvePnPRefineLM(objects, points, matrix, distortion, rvec, tvec)
    rotation = cv2.Rodrigues(rvec)[0]

    order = "image_top_left"
    if reference_rotation is not None:
        reference = np.asarray(reference_rotation, dtype=np.float64).reshape(3, 3)
        flipped = rotation @ np.diag([-1.0, -1.0, 1.0])
        if _rotation_angle(reference.T @ flipped) < _rotation_angle(reference.T @ rotation):
            rotation = flipped
            points = np.ascontiguousarray(points[::-1])
            rvec = cv2.Rodrigues(rotation)[0]
            order = "reference_rotation_reversed"
        else:
            order = "reference_rotation_original"

    projection = cv2.projectPoints(objects, rvec, tvec, matrix, distortion)[0].reshape(-1, 2)
    errors = np.linalg.norm(projection - points, axis=1)
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = tvec.reshape(3)
    return {
        "vision_pose": pose,
        "reprojection_error_px": float(np.sqrt(np.mean(errors ** 2))),
        "reprojection_mean_px": float(errors.mean()),
        "corner_count": len(points),
        "corner_order": order,
        "candidate_reprojection_errors_px": [entry[0] for entry in candidates],
        "selection": "positive_depth_minimum_reprojection_then_LM",
        "translation_unit": "mm",
        "image_points": points,
        "projected_points": projection,
    }


def estimate_board_pose(
    image, camera_matrix, dist_coeffs, board_grid, square_size_mm,
    reference_rotation=None,
):
    """Estimate C_T_M from an image, returning mm and pixel reprojection RMS."""
    corners = detect_board_corners(image, board_grid)
    return solve_board_pose(
        corners, camera_matrix, dist_coeffs, board_grid, square_size_mm,
        reference_rotation=reference_rotation,
    )


def calibrate_hand_eye(base_tool_poses, camera_board_poses, method="PARK"):
    """Estimate H=E_T_C from X=B_T_E and V=C_T_M satisfying X H V=A.

    All translations must already use mm. The returned fixed-target residuals
    describe consistency of this calibration data, not robot positioning AP.
    No samples are silently removed and no residual acceptance limit is assumed.
    """
    robot = np.asarray(base_tool_poses, dtype=np.float64)
    board = np.asarray(camera_board_poses, dtype=np.float64)
    if robot.ndim != 3 or robot.shape[1:] != (4, 4) or robot.shape != board.shape:
        raise ValueError("手眼数据须为一一配对的 N×4×4 位姿数组。")
    if len(robot) < 3:
        raise ValueError("手眼标定至少需要 3 组位姿，并绕至少两个非平行轴转动。")
    if not np.isfinite(robot).all() or not np.isfinite(board).all():
        raise ValueError("手眼位姿中存在非有限数值。")
    rotations = [cv2.Rodrigues(robot[0, :3, :3].T @ pose[:3, :3])[0].ravel() for pose in robot[1:]]
    if np.linalg.matrix_rank(rotations) < 2:
        raise ValueError("手眼采样缺少绕两个非平行轴的转动。")
    methods = {"PARK": cv2.CALIB_HAND_EYE_PARK, "TSAI": cv2.CALIB_HAND_EYE_TSAI}
    if method not in methods:
        raise ValueError("手眼方法仅支持 PARK 或 TSAI。")
    rotation, translation = cv2.calibrateHandEye(
        list(robot[:, :3, :3]), list(robot[:, :3, 3]),
        list(board[:, :3, :3]), list(board[:, :3, 3]), method=methods[method],
    )
    if not np.isfinite(rotation).all() or not np.isfinite(translation).all():
        raise ValueError("手眼标定未收敛，请检查位姿配对与采样运动。")
    hand_eye = np.eye(4)
    hand_eye[:3, :3] = rotation
    hand_eye[:3, 3] = translation.reshape(3)
    targets = robot @ hand_eye @ board
    target = np.eye(4)
    target[:3, 3] = targets[:, :3, 3].mean(axis=0)
    left, _, right = np.linalg.svd(targets[:, :3, :3].mean(axis=0))
    target[:3, :3] = left @ np.diag([1, 1, np.linalg.det(left @ right)]) @ right
    translation_residuals = np.linalg.norm(targets[:, :3, 3] - target[:3, 3], axis=1)
    rotation_residuals = np.array([
        _rotation_angle(target[:3, :3].T @ pose[:3, :3]) for pose in targets
    ])
    return {
        "hand_eye": hand_eye,
        "target_pose_base": target,
        "translation_residuals_mm": translation_residuals,
        "rotation_residuals_deg": rotation_residuals,
        "translation_rms_mm": float(np.sqrt(np.mean(translation_residuals ** 2))),
        "rotation_rms_deg": float(np.sqrt(np.mean(rotation_residuals ** 2))),
        "sample_count": len(robot),
        "method": method,
        "translation_unit": "mm",
        "transform_convention": "E_T_C",
    }
