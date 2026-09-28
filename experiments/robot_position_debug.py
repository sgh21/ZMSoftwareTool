"""Read the legacy robot_error data without changing images or cached poses.

The three folders are calibration captures, not a documented repeatability
trial. Only calib_00 is used as an explicitly unverified cross-period demo.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from core.services.position_monitoring_service import PositionMonitoringService, write_document


def build_legacy_batch(
    folder, *, program_id="legacy_debug", direction_id="unspecified", label=None,
    include_controller_rotations=False, filenames=None,
):
    """Read robot_poses.npz (m) into an image manifest with transforms in mm.

    Filenames identify observations only. Neither matching filenames across
    folders nor recorded controller poses prove an identical commanded target.
    Each image is kept as one separate point; no repeat samples are invented.
    """
    folder = Path(folder).resolve()
    source = folder / "robot_poses.npz"
    with np.load(source, allow_pickle=False) as data:
        names = [str(name) for name in data["filenames"]]
        poses = np.asarray(data["T_base_tool"], dtype=float).copy()
    if poses.shape != (len(names), 4, 4):
        raise ValueError("robot_poses.npz 的文件名与位姿数量不匹配。")
    selected = set(names if filenames is None else filenames)
    missing = selected - set(names)
    if missing:
        raise ValueError(f"robot_poses.npz 中找不到这些图像记录：{sorted(missing)}")
    if not selected:
        raise ValueError("至少选择一张图像。")
    poses[:, :3, 3] *= 1000
    samples = []
    rotations = {}
    for name, pose in zip(names, poses):
        if name not in selected:
            continue
        image_path = folder / "calibration_images" / name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        point = Path(name).stem
        samples.append({
            "point_id": point,
            "direction_id": direction_id,
            "sample_id": name,
            "image_path": str(image_path),
            "robot_pose": pose.tolist(),
        })
        if include_controller_rotations:
            rotations.setdefault(point, pose[:3, :3].tolist())
    warnings = [
        "旧数据缺少指令点、接近方向及重复到达编号；每张图像暂列为独立点，每组仅 1 次观测，不能计算 RP。",
        "跨目录的同名图像不保证同一指令位姿；不同文件名也不能默认合并为同一检测点。",
        "robot_pose 来自原 robot_poses.npz 的控制器记录，平移已由 m 换算为 mm；未使用 laser_enhanced 数据或旧 PnP 缓存。",
    ]
    if include_controller_rotations:
        warnings.append("Q 使用记录的控制器末端朝向近似，未经独立外部测量确认；不代表已获得绝对定位误差。")
    return {
        "schema_version": 1,
        "batch_id": uuid4().hex,
        "label": label or folder.name,
        "program_id": program_id,
        "target_id": "legacy_checkerboard",
        "length_unit": "mm",
        "orientation_source": "controller_approximation" if rotations else "unspecified",
        "comparison_status": "debug_unverified",
        "source": str(source),
        "samples": samples,
        "base_rotations": rotations,
        "initial_errors": {},
        "warnings": warnings,
    }


def make_debug_manifests(data_root, output_dir):
    """Write three input manifests and legacy camera parameters; do not run PnP."""
    data_root = Path(data_root).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.is_relative_to(data_root):
        raise ValueError("调试产物目录必须位于原始 robot_error 数据目录之外。")
    handeye = build_legacy_batch(data_root / "calib_data20", label="旧数据手眼标定（21 组）")
    handeye["comparison_status"] = "calibration_only"
    handeye["warnings"].append("本批次仅用于手眼标定，不可作为定位监控基准或重复性样本。")
    baseline = build_legacy_batch(
        data_root / "camera_pos_normal50", label="旧数据基准演示（仅 calib_00）",
        direction_id="unverified", include_controller_rotations=True,
        filenames=["calib_00.jpg"],
    )
    current = build_legacy_batch(
        data_root / "camera_pos_x05y-05_50", label="旧数据复测演示（仅 calib_00）",
        direction_id="unverified", filenames=["calib_00.jpg"],
    )
    warning = (
        "仅 calib_00 的两次控制器记录接近；其余同名记录位姿明显不同，已排除。"
        "calib_00 的采集程序、靶标固定关系与相机安装是否一致仍未经确认，"
        "本结果仅供流程调试，不可当作实测机器人精度退化。"
    )
    baseline["warnings"].append(warning)
    current["warnings"].append(warning)
    parameters = {
        "schema_version": 1,
        "version": "legacy-debug-import",
        "length_unit": "mm",
        "camera_matrix": [
            [3675.2707, 0, 1231.6813],
            [0, 3674.6618, 1073.9725],
            [0, 0, 1],
        ],
        "dist_coeffs": [-0.11570, 0.32350, 0.00100, -0.00060, -2.9623],
        "board_grid": [39, 34],
        "square_size_mm": 0.5,
        "hand_eye": None,
        "source": "旧 master/core/model_degradation_monitoring.py 的相机与棋盘参数",
        "warnings": [
            "相机与棋盘参数沿用旧代码，未在本次重新标定；用于给定旧数据的调试。",
            "未填入既有手眼结果；请加载标定批次并执行手眼标定，或另行导入有效参数。",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, filename, content in (
        ("handeye_manifest", "handeye_manifest.json", handeye),
        ("baseline_manifest", "baseline_manifest.json", baseline),
        ("current_manifest", "current_manifest.json", current),
        ("parameters", "legacy_default_parameters.json", parameters),
    ):
        path = output_dir / filename
        path.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")
        paths[key] = str(path)
    return paths


def run_debug_pipeline(paths, output_dir, method="PARK"):
    """Run the existing service in isolated storage; never use main-app state."""
    output_dir = Path(output_dir).resolve()
    service = PositionMonitoringService(root=output_dir / "session")
    service.load_parameters(paths["parameters"])
    service.load_observations(paths["handeye_manifest"])
    calibration = service.calibrate_hand_eye(method=method)
    service.save_parameters({
        "hand_eye": calibration["hand_eye"],
        "calibration_source": calibration["path"],
    })
    service.load_observations(paths["baseline_manifest"])
    service.create_baseline("旧数据 CLI 调试基准 · 采样关系待确认")
    service.load_observations(paths["current_manifest"])
    result = service.evaluate()
    report = {
        "debug_only": True,
        "warning": "仅用于未经采样对应确认的流程调试，不是实测机器人精度退化；单次到达不能计算 RP。",
        "method": method,
        "manifests": paths,
        "session_root": str(output_dir / "session"),
        "calibration": calibration,
        "result": result,
        "report_path": str(output_dir / "debug_result.json"),
    }
    write_document(report["report_path"], report)
    return report


def main():
    workspace = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="生成机器人定位监控旧数据调试输入，不修改原始数据。")
    parser.add_argument("--data-root", type=Path, default=workspace / "data" / "robot_error")
    parser.add_argument("--output-dir", type=Path, default=workspace / "data" / "processed" / "robot_position_debug")
    parser.add_argument("--run", action="store_true", help="在输出目录的隔离会话中执行手眼标定与演示评估")
    parser.add_argument("--method", choices=("PARK", "TSAI"), default="PARK", help="手眼标定方法")
    arguments = parser.parse_args()
    paths = make_debug_manifests(arguments.data_root, arguments.output_dir)
    if not arguments.run:
        print(json.dumps(paths, ensure_ascii=False, indent=2))
        print("已生成输入文件；尚未执行标定或评估。仅 calib_00 用于未经采样确认的演示，不计算 RP。")
        return
    print("仅用于未确认采样对应的流程调试，不是实测机器人精度退化。正在隔离会话中执行。")
    report = run_debug_pipeline(paths, arguments.output_dir, arguments.method)
    print(json.dumps({
        "warning": report["warning"],
        "handeye_translation_rms_mm": report["calibration"]["translation_rms_mm"],
        "handeye_rotation_rms_deg": report["calibration"]["rotation_rms_deg"],
        "status": report["result"]["status"],
        "summary": report["result"]["summary"],
        "report_path": report["report_path"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
