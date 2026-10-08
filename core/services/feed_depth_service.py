"""窝深理论值与误差报警设置；按当前设置评价数据，不改原始测量。"""

from math import isclose, isfinite
from pathlib import Path
from statistics import fmean

from core.services.position_persistence import read_json, write_document


DEFAULT_SETTINGS = {
    "depth_mode": "per_hole",
    "uniform_depth": 1.5,
    "error_lower_mm": None,
    "error_upper_mm": None,
}


def validate_feed_depth_settings(settings):
    values = {key: settings.get(key, default) for key, default in DEFAULT_SETTINGS.items()}
    if values["depth_mode"] not in ("per_hole", "uniform"):
        raise ValueError("请选择逐孔理论值或统一理论窝深。")
    if values["depth_mode"] == "uniform" or values["uniform_depth"] is not None:
        try:
            depth = float(values["uniform_depth"])
        except (TypeError, ValueError):
            raise ValueError("理论窝深必须为大于 0 的有限数值，单位 mm。") from None
        if not isfinite(depth) or depth <= 0:
            raise ValueError("理论窝深必须为大于 0 的有限数值，单位 mm。")
        values["uniform_depth"] = depth
    lower, upper = values["error_lower_mm"], values["error_upper_mm"]
    if lower is None and upper is None:
        return values
    try:
        lower, upper = float(lower), float(upper)
    except (TypeError, ValueError):
        raise ValueError("请同时填写误差下限和上限，或同时留空。") from None
    if not isfinite(lower) or not isfinite(upper):
        raise ValueError("误差下限和上限必须为有限数值，单位 mm。")
    if lower > upper:
        raise ValueError("误差下限不能大于上限。")
    values.update(error_lower_mm=lower, error_upper_mm=upper)
    return values


def load_feed_depth_settings(path):
    path = Path(path)
    return validate_feed_depth_settings(read_json(path)) if path.exists() else dict(DEFAULT_SETTINGS)


def save_feed_depth_settings(path, settings):
    values = validate_feed_depth_settings(settings)
    write_document(path, values)
    return values


def _over_limit(bias, settings):
    lower, upper = settings["error_lower_mm"], settings["error_upper_mm"]
    if lower is None:
        return False
    return ((bias < lower and not isclose(bias, lower, rel_tol=0.0, abs_tol=1e-12))
            or (bias > upper and not isclose(bias, upper, rel_tol=0.0, abs_tol=1e-12)))


def evaluate_feed_depth(data, settings):
    """评价已有模拟协议：每批同孔排、同理论值；保留来源及实测统计。"""
    values = validate_feed_depth_settings(settings)
    rows, history = [], []
    batches = {record["batch_id"]: [] for record in data["history"]}
    for source in data["rows"]:
        theory = values["uniform_depth"] if values["depth_mode"] == "uniform" else source["theoretical_depth_mm"]
        bias = source["actual_depth_mm"] - theory
        row = {**source, "theoretical_depth_mm": theory, "bias_mm": bias,
               "alarm": _over_limit(bias, values)}
        rows.append(row)
        batches[row["batch_id"]].append(row)
    for source in data["history"]:
        batch = batches[source["batch_id"]]
        bias = fmean(row["bias_mm"] for row in batch)
        history.append({
            **source,
            "theoretical_depth_mm": batch[0]["theoretical_depth_mm"],
            "bias_mm": bias,
            "alarm": _over_limit(bias, values),
            "alarm_count": sum(row["alarm"] for row in batch),
        })
    return {"rows": rows, "history": history}
