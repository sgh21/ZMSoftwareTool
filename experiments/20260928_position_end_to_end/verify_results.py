"""从末端位置直接重算指标；独立于生产坐标变换和统计实现。"""

import json
from pathlib import Path
import re

import numpy as np


def identity(image):
    match = re.search(r"(P\d{3})_(D\d{3})", str(image))
    if match is None:
        raise ValueError(f"图像缺少 P/D 编号：{image}")
    return match.groups()


def scatter(positions):
    values = np.asarray(positions, dtype=float)
    radii = np.linalg.norm(values - values.mean(axis=0), axis=1)
    return np.r_[3 * values.std(axis=0, ddof=1), radii.mean() + 3 * radii.std(ddof=1)]


def metrics(initial, current, targets):
    """输入为 (P,D) -> XYZ；空间列来自各方向模长，不由三轴汇总合成。"""
    if initial.keys() != current.keys() or current.keys() != targets.keys():
        raise ValueError("基线、复测和目标的 P/D 不一致")
    groups = {}
    for point in sorted({point for point, _ in targets}):
        keys = sorted(key for key in targets if key[0] == point)
        first = np.asarray([initial[key] for key in keys])
        last = np.asarray([current[key] for key in keys])
        commanded = np.asarray([targets[key] for key in keys])
        first_error, last_error = first - commanded, last - commanded
        first_scatter, last_scatter = scatter(first), scatter(last)
        ap_change = np.r_[
            (np.abs(last_error) - np.abs(first_error)).mean(axis=0),
            (np.linalg.norm(last_error, axis=1) - np.linalg.norm(first_error, axis=1)).mean(),
        ]
        groups[point] = {
            "count": len(keys), "baseline_scatter": first_scatter.tolist(),
            "repeatability": last_scatter.tolist(),
            "repeatability_change": (last_scatter - first_scatter).tolist(),
            "absolute_change": ap_change.tolist(),
            "baseline_ap": float(np.linalg.norm(first_error, axis=1).mean()),
            "current_ap": float(np.linalg.norm(last_error, axis=1).mean()),
        }
    names = ("baseline_scatter", "repeatability", "repeatability_change", "absolute_change")
    summary = {name: np.mean([group[name] for group in groups.values()], axis=0).tolist()
               for name in names}
    return {"groups": groups, "summary": summary}


def production_values(result, group=None):
    source = result["summary"] if group is None else group
    axes = source["axis_3sigma_base"] if group is None else source["current"]["axis_3sigma_base"]
    return {
        "repeatability": [*axes, source["rp_current"]],
        "repeatability_change": [*source["axis_3sigma_change_base"], source["rp_change"]],
        "absolute_change": [*source["absolute_axis_change"], source["absolute_ap_change"]],
    }


def error_statistics(vectors):
    values = np.asarray(vectors)
    distances = np.linalg.norm(values, axis=1)
    return {
        "count": len(values), "mean_signed_xyz_mm": values.mean(axis=0).tolist(),
        "mean_abs_xyz_mm": np.abs(values).mean(axis=0).tolist(),
        "mean_distance_mm": float(distances.mean()),
        "p95_distance_mm": float(np.percentile(distances, 95)),
        "max_distance_mm": float(distances.max()),
    }


def verify_dataset(dataset, baseline_observations, current_observations, evaluation):
    """只核验已完成的测量，不解算图片，不将仿真真值传给生产服务。"""
    dataset = Path(dataset)
    truth, measured, targets, models, pose_errors = {}, {}, None, {}, {}
    plan = json.loads((dataset / "fixed_observation_plan.json").read_text(encoding="utf-8-sig"))
    directions = {
        (point["point_id"], f"D{int(direction['direction_id'][1:]):03d}"):
        np.asarray(direction["direction_B"])
        for point in plan["points"] for direction in point["directions"]
    }
    for period, observations in (("baseline", baseline_observations), ("current", current_observations)):
        batch = observations["batch_id"]
        record = json.loads((dataset / batch / "record.json").read_text(encoding="utf-8-sig"))
        frame_map = {identity(frame["image"]): frame for frame in record["frames"]}
        truth[period] = {key: np.asarray(frame["actual"])[:3, 3] for key, frame in frame_map.items()}
        this_targets = {key: np.asarray(frame["ideal"])[:3, 3] for key, frame in frame_map.items()}
        if targets is not None and (targets.keys() != this_targets.keys() or
                                   any(not np.array_equal(targets[key], this_targets[key]) for key in targets)):
            raise ValueError("两期目标位置不一致")
        targets = this_targets
        measured[period] = {
            (sample["point_id"], sample["direction_id"]): np.asarray(sample["end_pose"])[:3, 3]
            for sample in observations["samples"]
        }
        if measured[period].keys() != truth[period].keys():
            raise ValueError(f"{batch} 图像测量与真值的 P/D 不一致")
        errors = [measured[period][key] - truth[period][key] for key in sorted(frame_map)]
        offsets = [truth[period][key] - targets[key] for key in sorted(frame_map)]
        expected = [2 * record["inertia"] * directions[key] for key in sorted(frame_map)]
        difference = np.asarray(offsets) - expected
        models[period] = {
            "batch": batch, "inertia": record["inertia"], "count": len(frame_map),
            "expected_offset_mm": 2 * record["inertia"],
            "offset_min_mm": float(np.linalg.norm(offsets, axis=1).min()),
            "offset_max_mm": float(np.linalg.norm(offsets, axis=1).max()),
            "maximum_model_difference_mm": float(np.linalg.norm(difference, axis=1).max()),
        }
        pose_errors[period] = {
            "summary": error_statistics(errors),
            "samples": [{"point_id": key[0], "direction_id": key[1],
                         "error_xyz_mm": vector.tolist(), "error_distance_mm": float(np.linalg.norm(vector))}
                        for key, vector in zip(sorted(frame_map), errors)],
        }
    exact = metrics(truth["baseline"], truth["current"], targets)
    observed = metrics(measured["baseline"], measured["current"], targets)
    actual_values = production_values(evaluation)
    differences = {
        name: (np.asarray(actual_values[name]) - observed["summary"][name]).tolist()
        for name in actual_values
    }
    group_differences = {}
    for group in evaluation["groups"]:
        point = group["point_id"]
        values = production_values(evaluation, group)
        group_differences[point] = {
            name: (np.asarray(values[name]) - observed["groups"][point][name]).tolist()
            for name in values
        }
    all_differences = [value for values in group_differences.values() for value in values.values()]
    all_differences.extend(differences.values())
    maximum = float(np.abs(all_differences).max())
    ratio = models["current"]["inertia"] / models["baseline"]["inertia"]
    ratios = np.asarray(exact["summary"]["repeatability"]) / exact["summary"]["baseline_scatter"]
    comparison = {
        name: {"measured": actual_values[name], "truth": exact["summary"][name],
               "difference": (np.asarray(actual_values[name]) - exact["summary"][name]).tolist()}
        for name in actual_values
    }
    return {
        "dataset": str(dataset.resolve()), "method": "Independent statistics from Cartesian positions; no production evaluator imports",
        "axes": ["X", "Y", "Z", "space"], "length_unit": "mm",
        "simulation_model": models, "truth_metrics": exact,
        "measured_independent_metrics": observed,
        "statistics_check": {"maximum_difference_mm": maximum, "passed": maximum < 1e-8,
                             "summary_difference_mm": differences, "group_difference_mm": group_differences},
        "metric_comparison": comparison, "position_measurement_errors": pose_errors,
        "inertia_scaling": {"expected_ratio": ratio, "truth_scatter_ratio": ratios.tolist()},
    }
