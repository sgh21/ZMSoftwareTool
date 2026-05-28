"""UTF-8 reports and figures for the Bayesian Calibration Pipeline ablation."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.calibration.bayesian_calibration_pipeline.configs.pipeline_config import METHOD_ORDER


def write_outputs(report: dict[str, Any]) -> None:
    """Generate figures, clean internal objects, and write JSON/HTML/notes."""
    output_dir = Path(str(report["artifacts"]["output_dir"]))
    figures_dir = Path(str(report["artifacts"]["figures_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    public_report = _with_figures_and_cleaned(report, figures_dir)
    (output_dir / "report.json").write_text(
        json.dumps(public_report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (output_dir / "report.html").write_text(_format_html(public_report, output_dir), encoding="utf-8")
    (output_dir / "experiment_notes.html").write_text(
        _format_notes(public_report, output_dir), encoding="utf-8"
    )


def _with_figures_and_cleaned(report: dict[str, Any], figures_dir: Path) -> dict[str, Any]:
    cleaned = dict(report)
    scenarios = []
    for scenario in report["scenarios"]:
        scenario_copy = dict(scenario)
        artifacts = scenario_copy.pop("_plot_artifacts")
        datasets = scenario_copy.pop("_datasets")
        scenario_copy["figures"] = _make_figures(
            scenario_copy,
            datasets["train"],
            datasets["holdout"],
            datasets["validation"],
            artifacts,
            figures_dir,
        )
        scenarios.append(scenario_copy)
    cleaned["scenarios"] = scenarios
    return cleaned


def _make_figures(
    scenario: dict[str, Any],
    train: dict[str, np.ndarray],
    holdout: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    artifacts: Any,
    figures_dir: Path,
) -> dict[str, str]:
    scenario_id = str(scenario["id"])
    label = str(scenario["label"])
    return {
        "workspace": str(_plot_workspace(scenario_id, label, train, holdout, validation, figures_dir)),
        "rmse_bars": str(_plot_rmse_bars(scenario_id, label, scenario["methods"], figures_dir)),
        "ab_balance_curves": str(_plot_ab_curves(scenario_id, label, scenario["curves"], figures_dir)),
        "global_weights": str(_plot_global_weights(scenario_id, label, artifacts, figures_dir)),
        "dynamic_weights": str(_plot_dynamic_weights(scenario_id, label, artifacts, figures_dir)),
        "subspace_scatter": str(_plot_subspace_scatter(scenario_id, label, train, artifacts, figures_dir)),
        "subspace_heatmap": str(_plot_subspace_heatmap(scenario_id, label, artifacts, figures_dir)),
    }


def _plot_workspace(
    scenario_id: str,
    label: str,
    train: dict[str, np.ndarray],
    holdout: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    figures_dir: Path,
) -> Path:
    path = figures_dir / f"{scenario_id}_workspace.png"
    fig = plt.figure(figsize=(7.4, 5.4), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    for dataset, name, color in (
        (train, "A train", "#2563eb"),
        (holdout, "B selection", "#dc2626"),
        (validation, "C validation", "#059669"),
    ):
        points = dataset["measured_positions"]
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=20, alpha=0.78, label=name, color=color)
    ax.set_title(f"{label}: A/B/C workspace")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_rmse_bars(
    scenario_id: str,
    label: str,
    methods: dict[str, dict[str, Any]],
    figures_dir: Path,
) -> Path:
    path = figures_dir / f"{scenario_id}_rmse_bars.png"
    labels = [method for method in METHOD_ORDER if method in methods]
    x = np.arange(len(labels), dtype=float)
    width = 0.24
    fig, ax = plt.subplots(figsize=(10.4, 4.8), constrained_layout=True)
    ax.bar(x - width, [methods[m]["train_A_rmse_mm"] for m in labels], width, label="A train", color="#2563eb")
    ax.bar(x, [methods[m]["selection_B_rmse_mm"] for m in labels], width, label="B selection", color="#dc2626")
    ax.bar(x + width, [methods[m]["validation_C_rmse_mm"] for m in labels], width, label="C validation", color="#059669")
    ax.set_title(f"{label}: A/B/C RMSE")
    ax.set_ylabel("RMSE (mm)")
    ax.set_xticks(x, labels, rotation=25, ha="right", fontsize=8)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_ab_curves(
    scenario_id: str,
    label: str,
    curves: dict[str, Any],
    figures_dir: Path,
) -> Path:
    path = figures_dir / f"{scenario_id}_ab_balance_curves.png"
    fig, ax = plt.subplots(figsize=(8.8, 5.0), constrained_layout=True)
    colors = {"M6": "#2563eb", "W3": "#7c3aed", "S1": "#ea580c", "D1": "#0f766e"}
    for method in ("M6", "W3", "S1", "D1"):
        rows = curves.get(method, [])
        if not rows:
            continue
        ordered = sorted(rows, key=lambda row: row["lambda"])
        best = min(ordered, key=lambda row: (row["ab_balance_score_m"], row["lambda"]))
        ax.plot(
            [row["lambda"] for row in ordered],
            [row["ab_balance_score_m"] * 1000.0 for row in ordered],
            marker="o",
            linewidth=1.3,
            label=method,
            color=colors[method],
        )
        ax.axvline(best["lambda"], color=colors[method], alpha=0.25)
    ax.set_xscale("log")
    ax.set_title(f"{label}: A/B balance lambda search")
    ax.set_xlabel("lambda")
    ax.set_ylabel("A+B+|A-B| score (mm)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_global_weights(scenario_id: str, label: str, artifacts: Any, figures_dir: Path) -> Path:
    path = figures_dir / f"{scenario_id}_global_weights.png"
    metrics = artifacts.global_metrics
    weights = np.asarray(artifacts.global_weights, dtype=float)
    x = np.arange(len(weights))
    fig, ax = plt.subplots(figsize=(10.6, 4.4), constrained_layout=True)
    colors = np.where(metrics.unidentifiable_mask, "#dc2626", "#2563eb")
    ax.bar(x, weights, color=colors)
    ax.set_yscale("log")
    ax.set_title(f"{label}: W3 global identifiability weights")
    ax.set_xlabel("geometry33 parameter index")
    ax.set_ylabel("weight coefficient")
    ax.set_xticks(x, [str(i) for i in x], fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_dynamic_weights(scenario_id: str, label: str, artifacts: Any, figures_dir: Path) -> Path:
    path = figures_dir / f"{scenario_id}_dynamic_weights.png"
    metrics = artifacts.global_metrics
    dynamic = artifacts.dynamic_fit
    x = np.arange(len(metrics.parameter_names))
    fig, ax = plt.subplots(figsize=(10.6, 4.4), constrained_layout=True)
    ax.bar(x - 0.18, metrics.base_weights, width=0.36, label="global base", color="#64748b")
    ax.bar(x + 0.18, dynamic.final_weights, width=0.36, label="D1 final", color="#0f766e")
    ax.set_yscale("log")
    ax.set_title(f"{label}: D1 dynamic weights")
    ax.set_xlabel("geometry33 parameter index")
    ax.set_ylabel("weight coefficient")
    ax.set_xticks(x, [str(i) for i in x], fontsize=7)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_subspace_scatter(
    scenario_id: str,
    label: str,
    train: dict[str, np.ndarray],
    artifacts: Any,
    figures_dir: Path,
) -> Path:
    path = figures_dir / f"{scenario_id}_subspace_scatter.png"
    partition = artifacts.subspace_partition
    points = train["measured_positions"]
    fig = plt.figure(figsize=(7.4, 5.4), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=partition.labels, cmap="tab10", s=24, alpha=0.82)
    ax.set_title(f"{label}: S1 identifiability subspaces")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    fig.colorbar(scatter, ax=ax, shrink=0.72, label="subspace")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _plot_subspace_heatmap(scenario_id: str, label: str, artifacts: Any, figures_dir: Path) -> Path:
    path = figures_dir / f"{scenario_id}_subspace_weight_heatmap.png"
    partition = artifacts.subspace_partition
    matrix = np.vstack(partition.subspace_weights)
    fig, ax = plt.subplots(figsize=(10.5, 3.2 + 0.32 * partition.K), constrained_layout=True)
    image = ax.imshow(np.log10(np.maximum(matrix, 1.0e-30)), aspect="auto", cmap="viridis")
    ax.set_title(f"{label}: S1 subspace-local weights")
    ax.set_xlabel("geometry33 parameter index")
    ax.set_ylabel("subspace")
    ax.set_xticks(np.arange(matrix.shape[1]), [str(i) for i in range(matrix.shape[1])], fontsize=7)
    ax.set_yticks(np.arange(partition.K), [str(i) for i in range(partition.K)])
    fig.colorbar(image, ax=ax, label="log10(weight)")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _format_html(report: dict[str, Any], output_dir: Path) -> str:
    settings_rows = "\n".join(
        f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
        for key, value in report["settings"].items()
    )
    sections = "\n".join(_scenario_html(scenario, output_dir) for scenario in report["scenarios"])
    method_items = "".join(
        f"<li><b>{html.escape(method)}</b>: {html.escape(report['method_labels'][method])}</li>"
        for method in report["method_order"]
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>Bayesian Calibration Pipeline Ablation</title>
<style>
body {{ font-family: Arial, 'Microsoft YaHei', sans-serif; margin: 24px; color: #111827; line-height: 1.5; }}
table {{ border-collapse: collapse; width: 100%; margin: 12px 0 20px; font-size: 12px; }}
th, td {{ border: 1px solid #d1d5db; padding: 6px 8px; text-align: right; }}
th:first-child, td:first-child, td:nth-child(2) {{ text-align: left; }}
th {{ background: #f3f4f6; }}
.note {{ background: #eef2ff; border-left: 4px solid #4f46e5; padding: 10px 12px; margin: 12px 0; }}
.warn {{ background: #fff7ed; border-left: 4px solid #ea580c; padding: 10px 12px; margin: 12px 0; }}
.grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }}
img {{ max-width: 100%; border: 1px solid #e5e7eb; margin: 8px 0 18px; }}
</style>
</head>
<body>
<h1>Bayesian Calibration Pipeline Ablation</h1>
<div class="note">This report is the legacy geometry-anchor ablation layer. B is the selection space and C is the held-out validation space.</div>
<h2>Settings</h2>
<table><tbody>{settings_rows}</tbody></table>
<h2>Methods</h2>
<ul>
{method_items}
</ul>
{sections}
</body>
</html>"""


def _scenario_html(scenario: dict[str, Any], output_dir: Path) -> str:
    method_rows = "\n".join(_method_row_html(scenario["methods"][method]) for method in METHOD_ORDER)
    selection_rows = "\n".join(
        f"<tr><td>{html.escape(method)}</td><td>{_fmt(value.get('lambda'))}</td><td>{html.escape(str(value.get('source')))}</td><td>{_fmt_mm(value.get('score_m'))}</td><td>{_fmt_mm(value.get('ab_abs_gap_m'))}</td></tr>"
        for method, value in scenario["lambda_selections"].items()
    )
    subspace_rows = "\n".join(
        "<tr>"
        f"<td>{row['subspace']}</td><td>{row['sample_count']}</td><td>{row['rank']}</td>"
        f"<td>{_fmt(row['condition_number'])}</td><td>{_fmt(row['mean_risk'])}</td>"
        f"<td>{_fmt(row['weight_min'])}</td><td>{_fmt(row['weight_max'])}</td><td>{row['unidentifiable_count']}</td>"
        "</tr>"
        for row in scenario["subspace"]["summaries"]
    )
    dynamic_rows = "\n".join(
        "<tr>"
        f"<td>{int(row['iteration'])}</td><td>{_fmt(row['weight_min'])}</td><td>{_fmt(row['weight_max'])}</td>"
        f"<td>{_fmt(row['weight_mean'])}</td><td>{_fmt(row['relative_weight_change'])}</td><td>{_fmt(row['theta_delta_l2'])}</td><td>{int(row['nfev'])}</td>"
        "</tr>"
        for row in scenario["dynamic"]["iterations"]
    )
    parameter_rows = "\n".join(_parameter_row_html(row, scenario["structure"]["subspace_K"]) for row in scenario["identifiability"]["parameter_table"])
    figures = "\n".join(
        f"<div><h3>{html.escape(name)}</h3><img src='{html.escape(_relative_path(path, output_dir))}'></div>"
        for name, path in scenario["figures"].items()
    )
    return f"""
<h2>{html.escape(scenario['label'])}</h2>
<div class="note">candidate=33 geometry parameters; SVD rank={scenario['structure']['rank']}; active={scenario['structure']['active_count']}; S1 K={scenario['structure']['subspace_K']}; order={scenario['structure']['subspace_order']}.</div>
{_scenario_conclusion(scenario)}
<h3>A/B/C 绮惧害</h3>
<table>
<thead><tr><th>鏂规硶</th><th>位</th><th>active</th><th>A RMSE mm</th><th>B RMSE mm</th><th>C RMSE mm</th><th>A/B gap mm</th><th>C/A gap mm</th><th>B max mm</th><th>C max mm</th><th>||胃/s||</th><th>weighted ||胃/s||</th><th>B gain vs M6</th><th>C gain vs M6</th></tr></thead>
<tbody>{method_rows}</tbody>
</table>
<h3>位 閫夋嫨</h3>
<table><thead><tr><th>鏂规硶</th><th>位</th><th>鏉ユ簮</th><th>AB score mm</th><th>|A-B| gap mm</th></tr></thead><tbody>{selection_rows}</tbody></table>
<h3>S1 瀛愮┖闂磋瘖鏂?/h3>
<table><thead><tr><th>瀛愮┖闂?/th><th>鏍锋湰鏁?/th><th>rank</th><th>condition</th><th>mean risk</th><th>w min</th><th>w max</th><th>寮卞彲杈ㄨ瘑鏁?/th></tr></thead><tbody>{subspace_rows}</tbody></table>
<h3>D1 鍔ㄦ€佹潈閲嶅寰幆</h3>
<table><thead><tr><th>iter</th><th>w min</th><th>w max</th><th>w mean</th><th>relative weight change</th><th>theta delta</th><th>nfev</th></tr></thead><tbody>{dynamic_rows}</tbody></table>
<h3>鍥捐〃</h3>
<div class="grid">{figures}</div>
<h3>鍙傛暟鍙鲸璇嗘€т笌鏉冮噸</h3>
<table>
<thead><tr><th>idx</th><th>鍙傛暟</th><th>active</th><th>scale</th><th>global rho</th><th>global kappa</th><th>global risk</th><th>global w</th><th>W3 w</th><th>D1 w</th>{''.join(f'<th>S1 s{k} w</th>' for k in range(scenario['structure']['subspace_K']))}</tr></thead>
<tbody>{parameter_rows}</tbody>
</table>
"""


def _scenario_conclusion(scenario: dict[str, Any]) -> str:
    methods = scenario["methods"]
    best_b = min(methods.values(), key=lambda row: row["selection_B_rmse_mm"])
    best_c = min(methods.values(), key=lambda row: row["validation_C_rmse_mm"])
    return (
        "<div class='warn'>"
        f"B-space best: <b>{html.escape(best_b['method'])}</b> "
        f"({best_b['selection_B_rmse_mm']:.4f} mm). "
        f"C-space best: <b>{html.escape(best_c['method'])}</b> "
        f"({best_c['validation_C_rmse_mm']:.4f} mm). "
        "B is the selection space and C is the held-out validation space; interpret them separately."
        "</div>"
    )


def _method_row_html(row: dict[str, Any]) -> str:
    return (
        "<tr>"
        f"<td>{html.escape(row['method'])}</td><td>{_fmt(row['lambda'])}</td><td>{row['active_count']}</td>"
        f"<td>{_fmt(row['train_A_rmse_mm'])}</td><td>{_fmt(row['selection_B_rmse_mm'])}</td><td>{_fmt(row['validation_C_rmse_mm'])}</td>"
        f"<td>{_fmt(row['A_B_gap_mm'])}</td><td>{_fmt(row['C_A_gap_mm'])}</td>"
        f"<td>{_fmt(row['selection_B_max_mm'])}</td><td>{_fmt(row['validation_C_max_mm'])}</td>"
        f"<td>{_fmt(row['normalized_parameter_l2'])}</td><td>{_fmt(row['weighted_parameter_l2'])}</td>"
        f"<td>{_fmt(row['B_gain_vs_M6_mm'])}</td><td>{_fmt(row['C_gain_vs_M6_mm'])}</td>"
        "</tr>"
    )


def _parameter_row_html(row: dict[str, Any], K: int) -> str:
    subspace_cells = "".join(f"<td>{_fmt(row[f'S1_s{k}_weight'])}</td>" for k in range(K))
    return (
        "<tr>"
        f"<td>{row['index']}</td><td>{html.escape(row['parameter'])}</td><td>{row['svd_active']}</td>"
        f"<td>{_fmt(row['scale'])}</td><td>{_fmt(row['global_rho'])}</td><td>{_fmt(row['global_kappa'])}</td>"
        f"<td>{_fmt(row['global_risk'])}</td><td>{_fmt(row['global_base_weight'])}</td>"
        f"<td>{_fmt(row['W3_weight'])}</td><td>{_fmt(row['D1_final_weight'])}</td>{subspace_cells}"
        "</tr>"
    )


def _format_notes(report: dict[str, Any], output_dir: Path) -> str:
    full_report_path = html.escape(str((output_dir / "report.html").resolve()))
    parts = [
        "<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'><title>Bayesian Calibration Pipeline Notes</title>",
        "<style>body{font-family:Arial,'Microsoft YaHei',sans-serif;margin:24px;line-height:1.6}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border:1px solid #ddd;padding:6px;text-align:right}td:first-child,th:first-child{text-align:left}</style>",
        "</head><body><h1>Bayesian calibration ablation notes</h1>",
    ]
    for scenario in report["scenarios"]:
        methods = scenario["methods"]
        best_b = min(methods.values(), key=lambda row: row["selection_B_rmse_mm"])
        best_c = min(methods.values(), key=lambda row: row["validation_C_rmse_mm"])
        parts.append(f"<h2>{html.escape(scenario['label'])}</h2>")
        parts.append(
            f"<p>B-space best: <b>{html.escape(best_b['method'])}</b>, "
            f"B RMSE={best_b['selection_B_rmse_mm']:.4f} mm. "
            f"C-space best: <b>{html.escape(best_c['method'])}</b>, "
            f"C RMSE={best_c['validation_C_rmse_mm']:.4f} mm.</p>"
        )
        parts.append("<table><thead><tr><th>鏂规硶</th><th>A</th><th>B</th><th>C</th><th>B gain vs M6</th><th>C gain vs M6</th></tr></thead><tbody>")
        for method in METHOD_ORDER:
            row = methods[method]
            parts.append(
                f"<tr><td>{html.escape(method)}</td><td>{row['train_A_rmse_mm']:.4f}</td><td>{row['selection_B_rmse_mm']:.4f}</td><td>{row['validation_C_rmse_mm']:.4f}</td><td>{row['B_gain_vs_M6_mm']:.4f}</td><td>{row['C_gain_vs_M6_mm']:.4f}</td></tr>"
            )
        parts.append("</tbody></table>")
    parts.append(f"<p>Full report: {full_report_path}</p>")
    parts.append("</body></html>")
    return "\n".join(parts)


def _relative_path(path: str, base: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(base.resolve())).replace("\\", "/")
    except ValueError:
        return str(Path(path).resolve()).replace("\\", "/")


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    numeric = float(value)
    if not np.isfinite(numeric):
        return str(numeric)
    if abs(numeric) >= 1000.0 or (0.0 < abs(numeric) < 1.0e-3):
        return f"{numeric:.3e}"
    return f"{numeric:.4f}"


def _fmt_mm(value_m: Any) -> str:
    if value_m is None:
        return ""
    return _fmt(float(value_m) * 1000.0)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return str(value)


