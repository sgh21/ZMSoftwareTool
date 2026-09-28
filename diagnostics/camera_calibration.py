"""从 ChArUco 原图标定相机与眼在手上的 E_T_C，仅写指定输出目录。

图像给出 C_T_M；record.robot_pose（优先）或 actual 提供配对的 B_T_E。
actual 在这里明确用作仿真机器人标定读数，不作为图像测量或 PnP 初值。
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np

from core.algorithms.board_pose import (
    _solve_charuco_pose,
    calibrate_hand_eye,
    make_charuco_board,
)
from core.algorithms.position_monitoring import validate_transform
from core.services.position_image_input import image_identity, simulation_parameters
from core.services.position_monitoring_service import write_document


def _detect_image(path, board):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"无法读取标定图像：{path}")
    corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(image)
    if corners is None or ids is None or len(ids) < 4:
        raise ValueError(f"{path.name} 未检测到至少 4 个 ChArUco 角点")
    ids = ids.reshape(-1)
    objects = board.getChessboardCorners()[ids]
    if np.linalg.matrix_rank(objects[:, :2] - objects[0, :2]) < 2:
        raise ValueError(f"{path.name} 的 ChArUco 角点共线，不能用于标定")
    return corners.reshape(-1, 2), ids, (image.shape[1], image.shape[0])


def _angle_degrees(rotation):
    sine = np.linalg.norm([
        rotation[2, 1] - rotation[1, 2],
        rotation[0, 2] - rotation[2, 0],
        rotation[1, 0] - rotation[0, 1],
    ]) / 2
    return float(np.degrees(np.arctan2(sine, (np.trace(rotation) - 1) / 2)))


def _pose_difference(measured, reference):
    measured, reference = np.asarray(measured), np.asarray(reference)
    delta = measured[:3, 3] - reference[:3, 3]
    return {
        "translation_xyz_mm": delta.tolist(),
        "translation_distance_mm": float(np.linalg.norm(delta)),
        "rotation_deg": _angle_degrees(reference[:3, :3].T @ measured[:3, :3]),
    }


def _fixed_target_quality(robot, vision, hand_eye):
    targets = np.asarray(robot) @ hand_eye @ np.asarray(vision)
    target = np.eye(4)
    target[:3, 3] = targets[:, :3, 3].mean(axis=0)
    left, _, right = np.linalg.svd(targets[:, :3, :3].mean(axis=0))
    target[:3, :3] = left @ np.diag([1, 1, np.linalg.det(left @ right)]) @ right
    translations = np.linalg.norm(targets[:, :3, 3] - target[:3, 3], axis=1)
    rotations = np.array([
        _angle_degrees(target[:3, :3].T @ pose[:3, :3]) for pose in targets
    ])
    return {
        "target_pose_base": target.tolist(),
        "translation_residuals_mm": translations.tolist(),
        "rotation_residuals_deg": rotations.tolist(),
        "translation_rms_mm": float(np.sqrt(np.mean(translations ** 2))),
        "rotation_rms_deg": float(np.sqrt(np.mean(rotations ** 2))),
    }


def calibrate_dataset(
    dataset, output, *, batch_id="B000", intrinsic_mode="calibrate",
    hand_eye_mode="calibrate", method="PARK", distortion_mode="estimate", progress=None,
):
    """标定并导出完整参数、独立手眼及 JSON 报告，不修改主应用状态。

    calibrate 模式不使用真值内参或手眼初始化。reference 显式读取已知值。
    distortion_mode='zero' 显式固定理想针孔零畸变；默认估计五项畸变。
    输入原生 parameters.json 与批次 record.json，所有位姿平移须为 mm。
    """
    dataset, output = Path(dataset).resolve(), Path(output).resolve()
    if output.is_relative_to(dataset):
        raise ValueError("标定输出目录不能位于原始数据目录内")
    if intrinsic_mode not in ("calibrate", "reference") or hand_eye_mode not in ("calibrate", "reference"):
        raise ValueError("内参与手眼模式仅支持 calibrate 或 reference")
    if distortion_mode not in ("estimate", "zero"):
        raise ValueError("畸变模式仅支持 estimate 或 zero")
    if method not in ("PARK", "TSAI"):
        raise ValueError("手眼方法仅支持 PARK 或 TSAI")
    paths = {name: output / f"{name}.json" for name in ("report", "parameters", "hand_eye")}
    if any(path.exists() for path in paths.values()):
        raise ValueError("输出目录已有标定结果，请选择新目录以保留旧参数")

    def notify(value, message):
        if progress:
            progress(value, message)

    notify(0, "读取标定板与配对机器人位姿")
    parameter_path = dataset / "parameters.json"
    document = json.loads(parameter_path.read_text(encoding="utf-8-sig"))
    if document.get("length_unit") != "mm":
        raise ValueError("标定数据 length_unit 必须为 mm")
    record_path = dataset / batch_id / "record.json"
    record = json.loads(record_path.read_text(encoding="utf-8-sig"))
    frames = record["frames"]
    if record["batch"] != batch_id or not frames:
        raise ValueError("标定批次不匹配或没有配对样本")
    if len(frames) < 3 and "calibrate" in (intrinsic_mode, hand_eye_mode):
        raise ValueError("标定至少需要 3 张不同视角的配对图像")
    specification = document["charuco"]
    board = make_charuco_board(specification, specification["square_length_mm"])
    corners, corner_ids, robot, sources, image_paths = [], [], [], [], []
    image_size = None
    for index, frame in enumerate(frames):
        image_path = (record_path.parent / frame["image"]).resolve()
        if image_identity(image_path)[0] != batch_id:
            raise ValueError(f"图片批次与记录不匹配：{image_path.name}")
        if image_path in image_paths:
            raise ValueError(f"标定图片重复：{image_path.name}")
        pose_field = "robot_pose" if "robot_pose" in frame else "actual"
        if pose_field not in frame:
            raise ValueError(f"{image_path.name} 缺少 robot_pose 或 actual 机器人位姿")
        robot.append(validate_transform(frame[pose_field], f"{image_path.name} B_T_E"))
        sources.append(pose_field)
        points, ids, size = _detect_image(image_path, board)
        if image_size is not None and size != image_size:
            raise ValueError("所有标定图像必须具有相同分辨率")
        image_size = size
        corners.append(points)
        corner_ids.append(ids)
        image_paths.append(image_path)
        notify(round(45 * (index + 1) / len(frames)), f"检测角点 {index + 1}/{len(frames)}")

    notify(48, "求解相机内参")
    flags = 0
    calibration_rms = None
    if intrinsic_mode == "calibrate":
        if distortion_mode == "zero":
            flags = cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
        objects = [board.getChessboardCorners()[ids].astype(np.float32) for ids in corner_ids]
        calibration_rms, matrix, distortion, _, _ = cv2.calibrateCamera(
            objects, [points.astype(np.float32) for points in corners], image_size,
            None, None, flags=flags,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-10),
        )
        intrinsic_source = "ChArUco 图像角点联合标定；未使用真值内参初始化"
    else:
        matrix = np.asarray(document["camera"]["camera_matrix"], dtype=float)
        distortion = np.asarray(document["camera"]["dist_coeffs"], dtype=float)
        intrinsic_source = "parameters.json/camera 已记录参考内参与畸变"
    if not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
        raise ValueError("相机标定结果包含非有限值")

    poses, vision = [], []
    for index, (points, ids, image_path) in enumerate(zip(corners, corner_ids, image_paths)):
        solved = _solve_charuco_pose(points, ids, board, matrix, distortion)
        vision.append(solved["vision_pose"])
        poses.append({
            "image_path": str(image_path), "robot_pose_source": sources[index],
            "B_T_E": robot[index].tolist(), "C_T_M": solved["vision_pose"].tolist(),
            "corner_count": solved["corner_count"], "corner_ids": ids.tolist(),
            "reprojection_rms_px": solved["reprojection_error_px"],
        })
        notify(55 + round(30 * (index + 1) / len(frames)), f"PnP {index + 1}/{len(frames)}")

    notify(88, "求解手眼并检查固定靶标一致性")
    if hand_eye_mode == "calibrate":
        hand_eye = calibrate_hand_eye(robot, vision, method)["hand_eye"]
        hand_eye_source = "图像 PnP C_T_M 与 record 机器人 B_T_E 配对标定"
    else:
        hand_eye = validate_transform(document["system_transforms"]["E_T_C_hand_eye"], "参考手眼")
        hand_eye_source = "parameters.json/system_transforms/E_T_C_hand_eye 已记录参考真值"
    validate_transform(hand_eye, "手眼标定输出")
    quality = _fixed_target_quality(robot, vision, hand_eye)
    reprojection_rms = float(np.sqrt(np.average(
        [pose["reprojection_rms_px"] ** 2 for pose in poses],
        weights=[pose["corner_count"] for pose in poses],
    )))

    # 估计全部完成后才构造真值对照；逐图 C_T_M 从不参与标定或选解。
    reference = simulation_parameters(document)
    reference_poses = {
        (dataset / item["image_filename"]).resolve(): item["C_T_M"]
        for item in document.get("extrinsics_by_capture", {}).values()
    }
    pose_comparisons = [
        {"image_path": pose["image_path"], **_pose_difference(pose["C_T_M"], reference_poses[path])}
        for path, pose in zip(image_paths, poses) if path in reference_poses
    ]
    warnings = [
        "标定一致性残差不是机器人定位精度或实机测量能力认证；未设置自动合格阈值。",
        "导出的固定靶标 B_T_M 来自原参数记录；手眼拟合的靶标位姿仅列入报告。",
    ]
    if "actual" in sources:
        warnings.append("record.actual 明确作为本次标定的仿真机器人读数，未当作图像测量结果。")
    if intrinsic_mode == "calibrate" and distortion_mode == "zero":
        warnings.append("本次显式采用理想针孔零畸变假设，不适用于未知畸变的实机相机。")
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset), "parameter_source": str(parameter_path),
        "record_source": str(record_path), "batch_id": batch_id, "sample_count": len(frames),
        "intrinsic_mode": intrinsic_mode, "hand_eye_mode": hand_eye_mode,
        "length_unit": "mm", "end_frame": "tool0",
        "pose_conventions": {"robot": "B_T_E", "vision": "C_T_M", "hand_eye": "E_T_C",
                             "board_origin": "charuco_top_left"},
        "camera_calibration": {
            "rms_px": float(calibration_rms) if calibration_rms is not None else reprojection_rms,
            "pnp_rms_px": reprojection_rms, "camera_matrix": matrix.tolist(),
            "dist_coeffs": distortion.reshape(-1).tolist(), "image_size_px": list(image_size),
            "distortion_mode": distortion_mode if intrinsic_mode == "calibrate" else "reference",
            "opencv_flags": flags, "source": intrinsic_source,
        },
        "hand_eye_calibration": {
            "method": method if hand_eye_mode == "calibrate" else "reference",
            "hand_eye": hand_eye.tolist(), "source": hand_eye_source, **quality,
        },
        "poses": poses,
        "reference_comparison": {
            "purpose": "事后对照，不参与标定初始化或 PnP 选解",
            "camera_matrix_difference": (matrix - np.asarray(reference["camera_matrix"])).tolist(),
            "hand_eye": _pose_difference(hand_eye, reference["hand_eye"]),
            "target_pose_base": _pose_difference(quality["target_pose_base"], reference["target_pose_base"]),
            "vision_poses": pose_comparisons,
        },
        "warnings": warnings,
    }
    parameters = {
        **reference, "camera_matrix": matrix.tolist(), "dist_coeffs": distortion.reshape(-1).tolist(),
        "image_size_px": list(image_size), "hand_eye": hand_eye.tolist(),
        "source_note": f"{batch_id}；内参：{intrinsic_source}；手眼：{hand_eye_source}；固定靶标使用记录参数",
    }
    hand_eye_document = {
        "hand_eye": hand_eye.tolist(), "length_unit": "mm", "transform_convention": "E_T_C",
        "end_frame": "tool0", "source_note": hand_eye_source,
    }
    write_document(paths["parameters"], parameters)
    write_document(paths["hand_eye"], hand_eye_document)
    write_document(paths["report"], report)
    notify(100, "标定结果已保存；主应用参数未修改")
    return {"report": report, **{f"{name}_path": str(path) for name, path in paths.items()}}


def main():
    parser = argparse.ArgumentParser(description="独立 ChArUco 相机内参与手眼标定")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-id", default="B000")
    parser.add_argument("--intrinsic-mode", choices=("calibrate", "reference"), default="calibrate")
    parser.add_argument("--hand-eye-mode", choices=("calibrate", "reference"), default="calibrate")
    parser.add_argument("--distortion-mode", choices=("estimate", "zero"), default="estimate")
    parser.add_argument("--method", choices=("PARK", "TSAI"), default="PARK")
    result = calibrate_dataset(**vars(parser.parse_args()), progress=lambda value, message: print(f"{value}% {message}"))
    print(result["report_path"])


if __name__ == "__main__":
    main()
