"""末端位置监控；长度单位统一为 mm，核心计算不依赖 UI 或机器人模型。

vision_pose 为 ^C T_M，hand_eye 为 ^E T_C。每点初始数据的第一条
观测固定为参考，base_rotations 中的 Q 为该参考末端到基座的旋转。
mean_base 是沿基座轴表达的参考点位置差，不是基座中的绝对坐标。
空间 RP 使用 GB/T 12642 的 l_mean + 3*s_l；轴向 3σ 为补充口径。
"""

from collections import defaultdict

import numpy as np


def _rotation(value, name: str) -> np.ndarray:
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError(f"{name} 必须是有限数值的 3×3 旋转矩阵")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=0):
        raise ValueError(f"{name} 的旋转矩阵不正交")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0):
        raise ValueError(f"{name} 的旋转矩阵行列式必须为 +1")
    return rotation


def validate_transform(value, name: str = "变换") -> np.ndarray:
    """校验输入的刚体变换并返回 NumPy 数组；不推断或换算长度单位。"""
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} 必须是有限数值的 4×4 齐次变换")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8, rtol=0):
        raise ValueError(f"{name} 的底行必须为 [0, 0, 0, 1]")
    _rotation(matrix[:3, :3], name)
    return matrix


def _group_samples(samples, period: str):
    groups = defaultdict(list)
    references = {}
    for sample in samples:
        point = str(sample["point_id"])
        direction = str(sample["direction_id"])
        pose = validate_transform(sample["vision_pose"], f"{period}/{point}/{direction}")
        references.setdefault(point, pose)
        groups[point, direction].append(pose)
    if not groups:
        raise ValueError(f"{period}没有观测数据")
    return groups, references


def _statistics(positions: np.ndarray, rotation: np.ndarray | None) -> dict:
    mean = positions.mean(axis=0)
    count = len(positions)
    result = {
        "count": count,
        "mean_local": mean.tolist(),
        "mean_base": (rotation @ mean).tolist() if rotation is not None else None,
        "rp": None,
        "axis_3sigma_local": None,
        "axis_3sigma_base": None,
    }
    if count < 2:
        return result
    radii = np.linalg.norm(positions - mean, axis=1)
    result["rp"] = float(radii.mean() + 3 * radii.std(ddof=1))
    result["axis_3sigma_local"] = (3 * positions.std(axis=0, ddof=1)).tolist()
    if rotation is not None:
        base_positions = positions @ rotation.T
        result["axis_3sigma_base"] = (3 * base_positions.std(axis=0, ddof=1)).tolist()
    return result


def _difference(current, baseline):
    if current is None or baseline is None:
        return None
    return (np.asarray(current) - np.asarray(baseline)).tolist()


def _mean_by_point(groups: list[dict], values: list):
    """先对方向等权，再对测点等权；缺值时不静默调整汇总权重。"""
    if any(value is None for value in values):
        return None
    by_point = defaultdict(list)
    for group, value in zip(groups, values):
        by_point[group["point_id"]].append(value)
    point_means = [np.mean(items, axis=0) for items in by_point.values()]
    return np.mean(point_means, axis=0).tolist()


def _vap(means: list) -> float | None:
    if len(means) < 2:
        return None
    return float(max(
        np.linalg.norm(np.asarray(first) - second)
        for index, first in enumerate(means)
        for second in means[index + 1:]
    ))


def evaluate_current_repeatability(current_samples, hand_eye, *, base_rotations=None) -> dict:
    """计算单期重复定位结果；Q 对应该期每个测点的第一条参考观测。"""
    hand_eye = validate_transform(hand_eye, "手眼变换")
    inverse_hand_eye = np.linalg.inv(hand_eye)
    current, references = _group_samples(current_samples, "当前时期")
    rotations = {
        str(point): _rotation(value, f"测点 {point} 的 Q")
        for point, value in (base_rotations or {}).items()
    }
    groups = []
    warnings = []
    for point in references:
        if point not in rotations:
            warnings.append(f"测点 {point} 缺少当前参考末端朝向 Q；基座 XYZ 结果不可用")
    for point, direction in sorted(current):
        reference_chain = hand_eye @ references[point]
        positions = np.asarray([
            (reference_chain @ np.linalg.inv(pose) @ inverse_hand_eye)[:3, 3]
            for pose in current[point, direction]
        ])
        stats = _statistics(positions, rotations.get(point))
        if stats["count"] < 30:
            warnings.append(f"{point}/{direction} 当前仅 {stats['count']} 次到达，少于 30 次；非完整国标样本")
        if stats["count"] < 2:
            warnings.append(f"{point}/{direction} 当前少于 2 次到达，不能计算重复性")
        groups.append({
            "point_id": point, "direction_id": direction, "baseline": None, "current": stats,
            **dict.fromkeys((
                "drift_local", "drift_base", "drift_distance", "rp_change",
                "axis_3sigma_change_local", "axis_3sigma_change_base", "initial_error_base",
                "current_error_base", "initial_absolute_ap", "absolute_ap",
                "absolute_ap_change", "absolute_axis_change",
            )),
        })
    points = [{
        "point_id": point, "baseline_vap": None, "vap_change": None,
        "current_vap": _vap([
            group["current"]["mean_local"] for group in groups if group["point_id"] == point
        ]),
    } for point in sorted(references)]
    summary = dict.fromkeys((
        "drift_base", "drift_distance", "mean_abs_drift_base", "rp_baseline", "rp_change",
        "axis_3sigma_change_base", "absolute_ap", "absolute_ap_change",
        "absolute_axis", "absolute_axis_change",
    ))
    summary["rp_current"] = _mean_by_point(groups, [group["current"]["rp"] for group in groups])
    summary["axis_3sigma_base"] = _mean_by_point(groups, [
        group["current"]["axis_3sigma_base"] for group in groups
    ])
    return {"groups": groups, "points": points, "summary": summary, "warnings": warnings}


def evaluate_position_monitoring(
    baseline_samples,
    current_samples,
    hand_eye,
    *,
    base_rotations=None,
    initial_errors=None,
) -> dict:
    """比较两期同点、同方向重复到达数据，返回可直接 JSON 保存的结果。

    samples: [{point_id, direction_id, vision_pose: 4×4}, ...]，允许额外元数据。
    base_rotations: {point_id: Q_i(3×3)}；缺少时保留局部结果，基座分量为 None。
    initial_errors: {point_id: {direction_id: 初始误差向量(基座系, mm)}}。
    每一条 sample 必须代表一次重新到达；停留连拍应由上游先合为一次观测。

    两期必须包含完全相同的测点/方向组合。各组样本数可以不同，但每组至少
    两条才能计算 RP；少于 30 条会标记样本不足。达到 30 条也不代表完整执行
    标准试验，轨迹、负载、热状态和测量系统等条件仍需由采集流程保证。
    """
    hand_eye = validate_transform(hand_eye, "手眼变换")
    inverse_hand_eye = np.linalg.inv(hand_eye)
    baseline, references = _group_samples(baseline_samples, "初始时期")
    current, _ = _group_samples(current_samples, "复测时期")
    if baseline.keys() != current.keys():
        missing = sorted(baseline.keys() - current.keys())
        extra = sorted(current.keys() - baseline.keys())
        raise ValueError(f"两期测点/接近方向不匹配；复测缺少 {missing}，新增 {extra}")

    rotations = {
        str(point): _rotation(value, f"测点 {point} 的 Q")
        for point, value in (base_rotations or {}).items()
    }
    errors = {str(point): value for point, value in (initial_errors or {}).items()}
    groups = []
    warnings = []
    for point in references:
        if point not in rotations:
            warnings.append(f"测点 {point} 缺少参考末端朝向 Q；基座 XYZ 结果不可用")

    for point, direction in sorted(baseline):
        rotation = rotations.get(point)
        reference_chain = hand_eye @ references[point]
        periods = []
        for label, samples in (("初始", baseline), ("复测", current)):
            positions = np.asarray([
                (reference_chain @ np.linalg.inv(pose) @ inverse_hand_eye)[:3, 3]
                for pose in samples[point, direction]
            ])
            stats = _statistics(positions, rotation)
            periods.append(stats)
            if stats["count"] < 30:
                warnings.append(
                    f"{point}/{direction} {label}仅 {stats['count']} 次到达，少于 30 次；"
                    "非完整国标样本"
                )
            if stats["count"] < 2:
                warnings.append(f"{point}/{direction} {label}少于 2 次到达，不能计算重复性")
        initial, measured = periods
        drift_local = _difference(measured["mean_local"], initial["mean_local"])
        drift_base = _difference(measured["mean_base"], initial["mean_base"])
        group = {
            "point_id": point,
            "direction_id": direction,
            "baseline": initial,
            "current": measured,
            "drift_local": drift_local,
            "drift_base": drift_base,
            "drift_distance": float(np.linalg.norm(drift_local)),
            "rp_change": _difference(measured["rp"], initial["rp"]),
            "axis_3sigma_change_local": _difference(
                measured["axis_3sigma_local"], initial["axis_3sigma_local"]
            ),
            "axis_3sigma_change_base": _difference(
                measured["axis_3sigma_base"], initial["axis_3sigma_base"]
            ),
            "initial_error_base": None,
            "current_error_base": None,
            "initial_absolute_ap": None,
            "absolute_ap": None,
            "absolute_ap_change": None,
            "absolute_axis_change": None,
        }
        point_errors = {str(key): value for key, value in errors.get(point, {}).items()}
        if direction in point_errors:
            initial_error = np.asarray(point_errors[direction], dtype=float)
            if initial_error.shape != (3,) or not np.isfinite(initial_error).all():
                raise ValueError(f"{point}/{direction} 初始误差必须是有限数值的三维向量")
            group["initial_error_base"] = initial_error.tolist()
            group["initial_absolute_ap"] = float(np.linalg.norm(initial_error))
            if drift_base is not None:
                current_error = initial_error + drift_base
                absolute_ap = float(np.linalg.norm(current_error))
                group.update({
                    "current_error_base": current_error.tolist(),
                    "absolute_ap": absolute_ap,
                    "absolute_ap_change": absolute_ap - group["initial_absolute_ap"],
                    "absolute_axis_change": (
                        np.abs(current_error) - np.abs(initial_error)
                    ).tolist(),
                })
        groups.append(group)

    points = []
    for point in sorted(references):
        point_groups = [group for group in groups if group["point_id"] == point]
        initial_vap = _vap([group["baseline"]["mean_local"] for group in point_groups])
        current_vap = _vap([group["current"]["mean_local"] for group in point_groups])
        points.append({
            "point_id": point,
            "baseline_vap": initial_vap,
            "current_vap": current_vap,
            "vap_change": _difference(current_vap, initial_vap),
        })

    summary = {
        name: _mean_by_point(groups, [group[name] for group in groups])
        for name in (
            "drift_base", "drift_distance", "rp_change", "axis_3sigma_change_base",
            "absolute_ap", "absolute_ap_change", "absolute_axis_change",
        )
    }
    summary["mean_abs_drift_base"] = _mean_by_point(groups, [
        np.abs(group["drift_base"]).tolist() if group["drift_base"] is not None else None
        for group in groups
    ])
    summary["absolute_axis"] = _mean_by_point(groups, [
        np.abs(group["current_error_base"]).tolist()
        if group["current_error_base"] is not None else None
        for group in groups
    ])
    summary["rp_baseline"] = _mean_by_point(groups, [group["baseline"]["rp"] for group in groups])
    summary["rp_current"] = _mean_by_point(groups, [group["current"]["rp"] for group in groups])
    summary["axis_3sigma_base"] = _mean_by_point(groups, [
        group["current"]["axis_3sigma_base"] for group in groups
    ])
    return {"groups": groups, "points": points, "summary": summary, "warnings": warnings}


def _multidirectional_samples(samples, period):
    """每点每方向只接受一条观测，并核对同点的固定指令位姿。"""
    groups = defaultdict(dict)
    ideals = {}
    for sample in samples:
        point, direction = str(sample["point_id"]), str(sample["direction_id"])
        if direction in groups[point]:
            raise ValueError(f"{period}/{point}/{direction} 重复；多方向模式每方向只允许一条观测")
        pose = validate_transform(sample["vision_pose"], f"{period}/{point}/{direction}")
        ideal = sample.get("ideal_pose")
        if ideal is not None:
            ideal = validate_transform(ideal, f"{period}/{point}/{direction} 理想位姿")
            if point in ideals and not np.allclose(ideals[point], ideal, atol=1e-6, rtol=0):
                raise ValueError(f"{period}/{point} 各方向的理想位姿不一致")
            ideals[point] = ideal
        groups[point][direction] = {"vision_pose": pose, "ideal_pose": ideal}
    if not groups:
        raise ValueError(f"{period}没有观测数据")
    return groups, ideals


def evaluate_multidirectional(
    current_samples,
    hand_eye,
    *,
    baseline_samples=None,
    base_rotations=None,
    target_pose_base=None,
) -> dict:
    """评价同点不同接近方向各一次的工程散布，不作为国标同方向 RP。

    散布沿用 mean(radius) + 3*std(radius, ddof=1)，轴向为 3σ。
    有基准时，两期必须包含相同的点位/方向，且使用同一个点位参考。
    Q 对应基准（单期时为当前）每点第一条观测的末端朝向。
    target_pose_base 是固定靶标的 ^B T_M；提供它时由视觉恢复末端位置，
    与各条 sample.ideal_pose (^B T_E) 比较。绝不读取仿真 truth/actual。
    mean_base 仍表示相对参考的基座分量；mean_position_base 为可用的绝对位置。
    """
    hand_eye = validate_transform(hand_eye, "手眼变换")
    inverse_hand_eye = np.linalg.inv(hand_eye)
    target = (
        validate_transform(target_pose_base, "靶标基座外参 A")
        if target_pose_base is not None else None
    )
    current, current_ideals = _multidirectional_samples(current_samples, "当前时期")
    baseline = None
    if baseline_samples is not None:
        baseline, baseline_ideals = _multidirectional_samples(baseline_samples, "初始时期")
        initial_keys = {(point, direction) for point, items in baseline.items() for direction in items}
        current_keys = {(point, direction) for point, items in current.items() for direction in items}
        if initial_keys != current_keys:
            raise ValueError(
                "两期测点/接近方向不匹配；"
                f"复测缺少 {sorted(initial_keys - current_keys)}，"
                f"新增 {sorted(current_keys - initial_keys)}"
            )
        for point in baseline_ideals.keys() & current_ideals.keys():
            if not np.allclose(baseline_ideals[point], current_ideals[point], atol=1e-6, rtol=0):
                raise ValueError(f"测点 {point} 两期理想位姿不一致")
    references = baseline if baseline is not None else current
    rotations = {
        str(point): _rotation(value, f"测点 {point} 的 Q")
        for point, value in (base_rotations or {}).items()
    }
    warnings = ["多方向工程散布：每方向一次，不是国标同方向重复定位试验"]
    if target is None:
        warnings.append("缺少靶标基座外参 A；绝对定位误差及其退化不可用")
    groups, points = [], []
    for point in sorted(current):
        reference = next(iter(references[point].values()))["vision_pose"]
        reference_chain = hand_eye @ reference
        rotation = rotations.get(point)
        if target is not None:
            reference_pose = target @ np.linalg.inv(reference) @ inverse_hand_eye
            rotation = reference_pose[:3, :3]
        elif rotation is None:
            warnings.append(f"测点 {point} 缺少参考末端朝向 Q；基座 XYZ 结果不可用")

        statistics, measurements = {}, {}
        for period, samples in (("baseline", baseline), ("current", current)):
            if samples is None:
                statistics[period] = None
                measurements[period] = {}
                continue
            measured = {}
            for direction, sample in samples[point].items():
                camera_chain = np.linalg.inv(sample["vision_pose"]) @ inverse_hand_eye
                local = (reference_chain @ camera_chain)[:3, 3]
                position = (target @ camera_chain)[:3, 3] if target is not None else None
                ideal = sample["ideal_pose"]
                error = position - ideal[:3, 3] if position is not None and ideal is not None else None
                measured[direction] = {
                    "local": local,
                    "base": rotation @ local if rotation is not None else None,
                    "position": position,
                    "error": error,
                }
            stats = _statistics(np.asarray([item["local"] for item in measured.values()]), rotation)
            stats["mean_position_base"] = (
                np.mean([item["position"] for item in measured.values()], axis=0).tolist()
                if target is not None else None
            )
            statistics[period], measurements[period] = stats, measured
            if stats["count"] < 2:
                label = "初始" if period == "baseline" else "当前"
                warnings.append(f"{point} {label}少于 2 个接近方向，不能计算多方向散布")
            if target is not None and any(item["error"] is None for item in measured.values()):
                warnings.append(f"{point}/{period} 缺少理想位姿；对应绝对定位误差不可用")

        details = []
        for direction in sorted(current[point]):
            measured = measurements["current"][direction]
            initial = measurements["baseline"].get(direction)
            error = measured["error"]
            initial_error = initial["error"] if initial is not None else None
            drift_local = _difference(measured["local"], initial["local"] if initial else None)
            drift_base = _difference(measured["base"], initial["base"] if initial else None)
            absolute_ap = float(np.linalg.norm(error)) if error is not None else None
            initial_ap = float(np.linalg.norm(initial_error)) if initial_error is not None else None
            details.append({
                "point_id": point, "direction_id": direction,
                "baseline_position_base": initial["position"].tolist()
                if initial is not None and initial["position"] is not None else None,
                "current_position_base": measured["position"].tolist()
                if measured["position"] is not None else None,
                "initial_error_base": initial_error.tolist() if initial_error is not None else None,
                "current_error_base": error.tolist() if error is not None else None,
                "initial_absolute_ap": initial_ap, "absolute_ap": absolute_ap,
                "absolute_ap_change": _difference(absolute_ap, initial_ap),
                "absolute_axis": np.abs(error).tolist() if error is not None else None,
                "absolute_axis_change": _difference(
                    np.abs(error) if error is not None else None,
                    np.abs(initial_error) if initial_error is not None else None,
                ),
                "drift_local": drift_local, "drift_base": drift_base,
                "mean_abs_drift_base": np.abs(drift_base).tolist() if drift_base is not None else None,
                "drift_distance": float(np.linalg.norm(drift_local)) if drift_local is not None else None,
            })
        initial, measured = statistics["baseline"], statistics["current"]
        group = {
            "point_id": point, "direction_id": "多方向", "directions": details,
            "baseline": initial, "current": measured,
            "rp_current": measured["rp"],
            "rp_baseline": initial["rp"] if initial is not None else None,
            "rp_change": _difference(measured["rp"], initial["rp"] if initial else None),
            "axis_3sigma_change_local": _difference(
                measured["axis_3sigma_local"], initial["axis_3sigma_local"] if initial else None
            ),
            "axis_3sigma_change_base": _difference(
                measured["axis_3sigma_base"], initial["axis_3sigma_base"] if initial else None
            ),
        }
        for name in (
            "drift_local", "drift_base", "drift_distance", "mean_abs_drift_base",
            "initial_error_base", "current_error_base", "initial_absolute_ap",
            "absolute_ap", "absolute_ap_change", "absolute_axis", "absolute_axis_change",
        ):
            group[name] = _mean_by_point(details, [item[name] for item in details])
        groups.append(group)
        initial_vap = _vap([item["local"] for item in measurements["baseline"].values()])
        current_vap = _vap([item["local"] for item in measurements["current"].values()])
        points.append({
            "point_id": point, "baseline_vap": initial_vap, "current_vap": current_vap,
            "vap_change": _difference(current_vap, initial_vap),
        })

    summary = {
        name: _mean_by_point(groups, [group[name] for group in groups])
        for name in (
            "drift_base", "drift_distance", "mean_abs_drift_base", "rp_baseline", "rp_current",
            "rp_change", "axis_3sigma_change_base", "absolute_ap", "absolute_ap_change",
            "absolute_axis", "absolute_axis_change",
        )
    }
    summary["axis_3sigma_base"] = _mean_by_point(groups, [
        group["current"]["axis_3sigma_base"] for group in groups
    ])
    return {
        "sampling_protocol": "multidirectional", "is_standard_repeatability": False,
        "groups": groups, "points": points, "summary": summary, "warnings": warnings,
    }
