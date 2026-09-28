"""从仿真原图独立重测，并在测量完成后与仿真真值对照。"""

from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from core.algorithms.position_monitoring import evaluate_multidirectional, validate_transform
from core.services.position_monitoring_service import (
    METRIC_LABELS,
    MULTIDIRECTIONAL_LABELS,
    PositionMonitoringService,
    metric_values,
    write_document,
)


def _error_summary(samples):
    errors = np.asarray([sample["position_error_xyz_mm"] for sample in samples])
    distances = np.asarray([sample["position_error_distance_mm"] for sample in samples])
    angles = np.asarray([sample["rotation_error_deg"] for sample in samples])
    absolute = np.abs(errors)
    return {
        "sample_count": len(samples),
        "mean_signed_xyz_mm": errors.mean(axis=0).tolist(),
        "mean_abs_xyz_mm": absolute.mean(axis=0).tolist(),
        "p95_abs_xyz_mm": np.percentile(absolute, 95, axis=0).tolist(),
        "max_abs_xyz_mm": absolute.max(axis=0).tolist(),
        "mean_distance_mm": float(distances.mean()),
        "p95_distance_mm": float(np.percentile(distances, 95)),
        "max_distance_mm": float(distances.max()),
        "mean_rotation_deg": float(angles.mean()),
        "p95_rotation_deg": float(np.percentile(angles, 95)),
        "max_rotation_deg": float(angles.max()),
    }


def _verify_batch(batch, hand_eye, target):
    """真值只在此核验阶段读取；重新构造的理想视觉数据只供 truth_result。"""
    rows, truth_samples = [], []
    inverse_hand_eye = np.linalg.inv(hand_eye)
    for sample in batch["samples"]:
        truth_value = sample.get("simulation_truth", {}).get("end_pose")
        if truth_value is None:
            raise ValueError(f"{sample['sample_id']} 缺少仿真末端真值 actual，无法核验")
        truth = validate_transform(truth_value, f"{sample['sample_id']} 仿真真值")
        measured = validate_transform(sample["end_pose"], f"{sample['sample_id']} 图像测量末端位姿")
        error = measured[:3, 3] - truth[:3, 3]
        relative_rotation = truth[:3, :3].T @ measured[:3, :3]
        sine = np.linalg.norm([
            relative_rotation[2, 1] - relative_rotation[1, 2],
            relative_rotation[0, 2] - relative_rotation[2, 0],
            relative_rotation[1, 0] - relative_rotation[0, 1],
        ]) / 2
        cosine = (np.trace(relative_rotation) - 1) / 2
        rows.append({
            "point_id": sample["point_id"], "direction_id": sample["direction_id"],
            "sample_id": sample["sample_id"], "image_path": sample["image_path"],
            "measured_position_xyz_mm": measured[:3, 3].tolist(),
            "truth_position_xyz_mm": truth[:3, 3].tolist(),
            "position_error_xyz_mm": error.tolist(),
            "position_error_distance_mm": float(np.linalg.norm(error)),
            "rotation_error_deg": float(np.degrees(np.arctan2(sine, cosine))),
            "reprojection_error_px": sample.get("reprojection_error_px"),
        })
        truth_samples.append({
            "point_id": sample["point_id"], "direction_id": sample["direction_id"],
            "ideal_pose": sample["ideal_pose"],
            "vision_pose": (inverse_hand_eye @ np.linalg.inv(truth) @ target).tolist(),
        })
    return {
        "batch_id": batch["batch_id"], "observation_path": batch["saved_path"],
        "samples": rows, "summary": _error_summary(rows),
    }, truth_samples


def _metric_comparison(measured, truth, measured_group=None, truth_group=None):
    rows = []
    for mode in METRIC_LABELS:
        measured_values = metric_values(measured, mode, measured_group)
        truth_values = metric_values(truth, mode, truth_group)
        rows.append({
            "metric": mode, "label": MULTIDIRECTIONAL_LABELS[mode],
            "axes": ["X", "Y", "Z", "space"], "unit": "mm",
            "measured": measured_values, "truth": truth_values,
            "difference": [
                observed - expected if observed is not None and expected is not None else None
                for observed, expected in zip(measured_values, truth_values)
            ],
        })
    return rows


def run_simulation_check(dataset, output_root, baseline_id="B001", current_id="B002", progress=None):
    """在 output_root 内运行隔离测量，原 dataset 只读，返回报告及报告路径。

    两期均经 PositionMonitoringService 的图像解算入口。record.actual 仅在
    测量与指标保存之后用于核验，既不初始化 PnP，也不代替失败的图像测量。
    """
    dataset, output_root = Path(dataset).resolve(), Path(output_root).resolve()
    if output_root.is_relative_to(dataset):
        raise ValueError("核验输出目录不能位于原始仿真数据目录内，请另选输出目录")
    if not dataset.is_dir():
        raise FileNotFoundError(f"仿真数据目录不存在：{dataset}")
    parameter_path = dataset / "parameters.json"
    baseline_path = dataset / baseline_id / "record.json"
    current_path = dataset / current_id / "record.json"
    for name, path in (
        ("相机与手眼参数", parameter_path),
        (f"基准批次 {baseline_id}", baseline_path),
        (f"复测批次 {current_id}", current_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"缺少{name}：{path}")

    def notify(value, message):
        if progress:
            progress(value, message)

    notify(0, "加载仿真测量参数")
    service = PositionMonitoringService(root=output_root)
    parameters = service.load_parameters(parameter_path)
    hand_eye = validate_transform(parameters["hand_eye"], "手眼参数")
    target = validate_transform(parameters["target_pose_base"], "靶标基座外参")
    service.baseline = None
    baseline_batch = service.load_observations(
        baseline_path, progress=lambda value, text: notify(round(value * 0.45), f"{baseline_id}：{text}")
    )
    service.create_baseline(baseline_id)
    current_batch = service.load_observations(
        current_path, progress=lambda value, text: notify(45 + round(value * 0.45), f"{current_id}：{text}")
    )
    notify(90, "计算图像测量指标")
    measured_result = service.evaluate()
    notify(93, "对照两期仿真末端真值")
    baseline_check, baseline_truth = _verify_batch(baseline_batch, hand_eye, target)
    current_check, current_truth = _verify_batch(current_batch, hand_eye, target)
    truth_result = evaluate_multidirectional(
        current_truth, hand_eye, baseline_samples=baseline_truth, target_pose_base=target
    )
    truth_groups = {group["point_id"]: group for group in truth_result["groups"]}
    point_comparisons = [{
        "point_id": group["point_id"],
        "metrics": _metric_comparison(
            measured_result, truth_result, group, truth_groups[group["point_id"]]
        ),
    } for group in measured_result["groups"]]
    report = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset), "parameter_file": str(parameter_path),
        "baseline_batch_id": baseline_id, "current_batch_id": current_id,
        "measurement_source": "images_pnp", "truth_source": "record.json/actual (verification only)",
        "measured_result": measured_result, "truth_result": truth_result,
        "batches": [baseline_check, current_check],
        "measurement_summary": _error_summary(baseline_check["samples"] + current_check["samples"]),
        "metric_comparison": _metric_comparison(measured_result, truth_result),
        "point_metric_comparison": point_comparisons,
    }
    report_path = output_root / "verification.json"
    write_document(report_path, report)
    notify(100, "仿真核验完成，已保存测量结果与真值对照")
    return {"report": report, "report_path": str(report_path)}
