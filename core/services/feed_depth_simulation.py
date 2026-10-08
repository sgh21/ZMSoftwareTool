"""生成明确标记的窝深演示数据，不写入实测存储或给出精度判定。"""

import csv
from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
import random
from statistics import fmean, pstdev, pvariance


def _write_csv(path, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            # 保留测量分辨率与孔号；统计文件保留计算精度。
            values = dict(row)
            if "actual_depth_mm" in values:
                values["actual_depth_mm"] = f"{row['actual_depth_mm']:.3f}"
                values["theoretical_depth_mm"] = f"{row['theoretical_depth_mm']:.3f}"
            writer.writerow(values)


def generate_feed_depth_simulation(
    output_dir, progress=None, *, start_date="2026-07-11", days=90,
    holes_per_day=30, seed=20261008,
):
    """按日生成一批，并导出逐孔总表、逐批文件、统计表及假设说明。

    退化阶段以第 1 天为起点：前 15 天近稳定，第 16 天起渐变，
    第 61 天起叠加加速项。模型参数为演示假设，不是实机标定值。
    progress(value, message) 最后到 99，界面刷新后由调用方完成 100%。
    """
    if not isinstance(days, int) or days < 1:
        raise ValueError("模拟天数必须为正整数")
    if not isinstance(holes_per_day, int) or holes_per_day < 1:
        raise ValueError("每天孔数必须为正整数")
    first_date = date.fromisoformat(start_date)
    last_date = first_date + timedelta(days=days - 1)
    destination = Path(output_dir).resolve()
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError("模拟输出目录已有文件，请选择一个新目录")

    rng = random.Random(seed)
    position_offsets = [rng.gauss(0.0, 0.0015) for _ in range(holes_per_day)]
    position_mean = fmean(position_offsets)
    position_offsets = [value - position_mean for value in position_offsets]
    rows, history, batches = [], [], []
    day_offset = 0.0
    local_timezone = timezone(timedelta(hours=8))

    if progress:
        progress(0, "生成模拟窝深：前期稳定，随后偏浅和孔间波动逐渐增大")
    for day_index in range(days):
        measured_date = first_date + timedelta(days=day_index)
        batch_id = f"SIM-FEED-{measured_date:%Y%m%d}"
        start_time = datetime.combine(measured_date, time(9), tzinfo=local_timezone)
        gradual = (max(day_index - 14, 0) / 75) ** 1.25
        accelerated = (max(day_index - 59, 0) / 30) ** 2
        base_bias = -0.003 - 0.028 * gradual - 0.044 * accelerated
        hole_sigma = 0.006 + 0.008 * gradual + 0.010 * accelerated
        day_offset = 0.72 * day_offset + rng.gauss(0.0, 0.0012)
        batch_rows = []
        for hole_index, position_offset in enumerate(position_offsets):
            actual_depth = round(
                1.500 + base_bias + day_offset + position_offset + rng.gauss(0.0, hole_sigma), 3,
            )
            batch_rows.append({
                "hole_id": f"{hole_index + 1:03d}",
                "actual_depth_mm": actual_depth,
                "theoretical_depth_mm": 1.500,
                "row_id": "R01",
                "batch_id": batch_id,
                "measured_at": (start_time + timedelta(seconds=20 * hole_index)).isoformat(),
                "condition_id": "SIM-CONDITION-01",
                "data_source": "simulation",
                "is_simulated": True,
                "simulation_seed": seed,
            })
        depths = [row["actual_depth_mm"] for row in batch_rows]
        mean_depth = fmean(depths)
        history.append({
            "batch_id": batch_id,
            "measured_at": batch_rows[-1]["measured_at"],
            "day_index": day_index,
            "count": len(depths),
            "theoretical_depth_mm": 1.500,
            "mean_depth_mm": mean_depth,
            "bias_mm": fmean(depth - 1.500 for depth in depths),
            "variance_mm2": pvariance(depths),
            "stddev_mm": pstdev(depths),
            "row_id": "R01",
            "condition_id": "SIM-CONDITION-01",
            "data_source": "simulation",
            "is_simulated": True,
            "simulation_seed": seed,
        })
        rows.extend(batch_rows)
        batches.append(batch_rows)
        if progress:
            progress(round(65 * (day_index + 1) / days), f"已生成第 {day_index + 1}/{days} 天模拟孔记录")

    destination.mkdir(parents=True, exist_ok=True)
    batch_dir = destination / "batches"
    batch_dir.mkdir()
    csv_path = destination / "hole_depths.csv"
    summary_path = destination / "batch_summary.csv"
    metadata_path = destination / "metadata.json"
    _write_csv(csv_path, rows)
    _write_csv(summary_path, history)
    for batch_index, batch_rows in enumerate(batches):
        _write_csv(batch_dir / f"{batch_rows[0]['batch_id']}.csv", batch_rows)
        if progress:
            progress(65 + round(30 * (batch_index + 1) / days), f"已保存第 {batch_index + 1}/{days} 个模拟批次")

    metadata = {
        "schema_version": 1,
        "data_source": "simulation",
        "is_simulated": True,
        "simulation_seed": seed,
        "start_date": first_date.isoformat(),
        "end_date": last_date.isoformat(),
        "days": days,
        "holes_per_day": holes_per_day,
        "theoretical_depth_mm": 1.500,
        "resolution_mm": 0.001,
        "timezone": "+08:00",
        "measurement_start_time": "09:00:00",
        "measurement_interval_seconds": 20,
        "condition_id": "SIM-CONDITION-01",
        "row_id": "R01",
        "model": {
            "day_index": "t = 0, 1, ..., days-1",
            "gradual": "g = (max(t-14, 0)/75)^1.25",
            "accelerated": "a = (max(t-59, 0)/30)^2",
            "base_bias_mm": "-0.003 - 0.028*g - 0.044*a",
            "independent_hole_sigma_mm": "0.006 + 0.008*g + 0.010*a",
            "day_offset_mm": "d[t] = 0.72*d[t-1] + N(0, 0.0012^2), d[-1] = 0",
            "fixed_hole_position_offset_mm": "N(0, 0.0015^2), centered across hole positions",
            "actual_depth_mm": "round(1.500 + base_bias + day_offset + position_offset + N(0, sigma^2), 3)",
            "phases": ["第1–15天近稳定", "第16–60天渐变", "第61天起叠加加速项"],
        },
        "statistics": {
            "basis": "All statistics use the rounded actual_depth_mm values exported to CSV.",
            "variance": "Population variance, denominator N; stddev = sqrt(variance).",
            "bias": "mean(actual_depth_mm - theoretical_depth_mm)",
            "single_hole": "Population variance is zero for one hole; insufficient for a dispersion trend.",
        },
        "assumptions": [
            "仅为页面演示而设定的渐进偏浅和散布增大，不代表设备的真实测量或参数。",
            "每日一批，同孔排、同理论窝深、同工况；每天均采样，不假设周末停机。",
            "保持孔位差异、相关日间偏移及独立孔噪声，日间统计值不强制单调。",
            "前90天模型为演示范围；更长时段继续外推公式，不表示实机寿命预测。",
            "未模拟维修、换刀或突发故障；无合格阈值、报警结论或进给轴故障归因。",
            "一天一个孔时，方差为零不代表已验证重复性。",
        ],
        "files": {
            "all_holes": csv_path.name,
            "batch_summary": summary_path.name,
            "per_batch": "batches/SIM-FEED-YYYYMMDD.csv",
            "import_note": "每批CSV符合一文件一批约定；hole_depths.csv是跨批总表，供检查与分析。",
        },
    }
    with metadata_path.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if progress:
        progress(99, "模拟数据与 CSV 已保存，等待刷新页面")
    return {
        "rows": rows,
        "history": history,
        "output_dir": str(destination),
        "csv_path": str(csv_path),
        "summary_path": str(summary_path),
        "metadata_path": str(metadata_path),
    }
