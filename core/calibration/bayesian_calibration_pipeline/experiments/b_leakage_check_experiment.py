"""B-leakage validation for the Bayesian calibration mainline."""

from __future__ import annotations

import argparse
import csv
import json
from html import escape
from pathlib import Path
from typing import Any

from core.calibration.bayesian_calibration_pipeline.configs.pipeline_config import Geometry33PipelineConfig
from core.calibration.bayesian_calibration_pipeline.core.bayesian_mainline import (
    json_ready,
    run_bayesian_calibration_analysis,
)
from core.calibration.bayesian_calibration_pipeline.core.data_io import load_dataset
from core.calibration.bayesian_calibration_pipeline.core.non_geometric import NonGeometricConfig
from core.calibration.bayesian_calibration_pipeline.core.statistical_residual import StatisticalResidualConfig


ROUTES = [
    {
        "route": "AB_Bayes_AB_Finetune_reference",
        "label": "AB Bayesian fit + AB joint fine-tune",
        "bayesian_train_scope": "AB",
        "fine_tune_train_scope": "AB",
        "bayesian_selection_scope": "internal",
        "b_role": "B_train participates in Bayesian coefficient fitting and Stage 4 joint fine-tune.",
    },
    {
        "route": "AB_Bayes_A_Finetune",
        "label": "AB Bayesian fit + A-only joint fine-tune",
        "bayesian_train_scope": "AB",
        "fine_tune_train_scope": "A",
        "bayesian_selection_scope": "internal",
        "b_role": "B_train participates in Bayesian coefficient fitting, but not Stage 4 joint fine-tune.",
    },
    {
        "route": "A_Bayes_BSelect_A_Finetune",
        "label": "A Bayesian fit with B selection + A-only joint fine-tune",
        "bayesian_train_scope": "A",
        "fine_tune_train_scope": "A",
        "bayesian_selection_scope": "B_validation",
        "b_role": "B_train selects basis groups and priors only; coefficients and Stage 4 use A_train only.",
    },
]


STAGES_TO_REPORT = {
    "geometry_anchor",
    "bayesian_basis",
    "bayesian_basis_fine_tuned_candidate",
    "fine_tuned_geometry_only",
}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    route_reports: dict[str, dict[str, Any]] = {}
    metric_rows: list[dict[str, Any]] = []
    selected_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for route in ROUTES:
        print(f"Running route: {route['route']}")
        report = run_one_route(args, output_dir, route)
        route_name = str(route["route"])
        route_reports[route_name] = report
        metric_rows.extend(stage_metric_rows(route, report))
        selected_rows.extend(selected_group_rows(route, report))
        diagnostic_rows.append(fine_tune_diagnostic_row(route, report))

    summary = build_summary(route_reports, metric_rows, diagnostic_rows)
    output = {
        "settings": {
            "train": str(args.train),
            "selection": str(args.selection),
            "output_dir": str(output_dir),
            "seed": int(args.seed),
            "c_fraction": float(args.c_fraction),
            "noise_std_mm": float(args.noise_std_mm),
            "fixed_fine_tune_ratio": 100000.0,
            "routes": ROUTES,
        },
        "summary": summary,
        "metrics": metric_rows,
        "bayesian_selected_groups": selected_rows,
        "fine_tune_diagnostics": diagnostic_rows,
        "route_reports": route_reports,
    }
    output = json_ready(output)

    (output_dir / "report.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_csv(output_dir / "leakage_comparison_metrics.csv", metric_rows)
    write_csv(output_dir / "bayesian_selected_groups_by_route.csv", selected_rows)
    write_csv(output_dir / "fine_tune_diagnostics_by_route.csv", diagnostic_rows)
    html = render_leakage_report(output)
    (output_dir / "report.html").write_text(html, encoding="utf-8")

    append_to_reference_report(Path(args.append_report), output_dir, output)

    print("B-leakage check complete.")
    print(f"  output: {output_dir.resolve()}")
    print(f"  report: {(output_dir / 'report.html').resolve()}")
    for row in diagnostic_rows:
        print(
            "  {route}: fine_tuned C_all={c:.6f} mm, peeled C_all={p:.6f} mm, "
            "theta_l2={t:.6f}".format(
                route=row["route"],
                c=float(row["fine_tuned_C_all_rmse_mm"]),
                p=float(row["peeled_C_all_rmse_mm"]),
                t=float(row["theta_update_scaled_l2"]),
            )
        )


def run_one_route(args: argparse.Namespace, output_dir: Path, route: dict[str, Any]) -> dict[str, Any]:
    geometry_config = Geometry33PipelineConfig(
        real_a=Path(args.train),
        real_b=Path(args.selection),
        output_dir=output_dir / str(route["route"]),
        seed=int(args.seed),
        jacobian_method=str(args.jacobian_method),
        max_nfev=int(args.max_nfev),
        lambda_count=int(args.lambda_count),
        fine_count=int(args.fine_count),
        real_c_fraction=float(args.c_fraction),
        quick=bool(args.quick),
    )
    non_geo_config = NonGeometricConfig(
        nonlinear_max_nfev=int(args.fine_tune_max_nfev),
        spectrum_permutations=int(args.spectrum_permutations),
        seed=int(args.seed),
    )
    statistical_config = StatisticalResidualConfig(
        noise_std_m=float(args.noise_std_mm) * 1.0e-3,
        cv_folds=int(args.stat_cv_folds),
        seed=int(args.seed),
        max_basis_groups=int(args.max_basis_groups),
    )
    report = run_bayesian_calibration_analysis(
        load_dataset(args.train),
        load_dataset(args.selection),
        geometry_config,
        non_geo_config,
        statistical_config,
        fine_tune_lambda_ratios=(100000.0,),
        bayesian_train_scope=str(route["bayesian_train_scope"]),
        fine_tune_train_scope=str(route["fine_tune_train_scope"]),
        bayesian_selection_scope=str(route["bayesian_selection_scope"]),
    )
    return json_ready(report)


def stage_metric_rows(route: dict[str, Any], report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in report.get("bayesian_stage_metrics", []):
        if str(row.get("stage")) not in STAGES_TO_REPORT:
            continue
        rows.append(
            {
                "route": route["route"],
                "route_label": route["label"],
                "bayesian_train_scope": route["bayesian_train_scope"],
                "bayesian_selection_scope": route["bayesian_selection_scope"],
                "fine_tune_train_scope": route["fine_tune_train_scope"],
                "B_train_role": route["b_role"],
                **row,
            }
        )
    return rows


def selected_group_rows(route: dict[str, Any], report: dict[str, Any]) -> list[dict[str, Any]]:
    groups = (
        report.get("bayesian_residual", {})
        .get("models", {})
        .get("bayesian_basis", {})
        .get("selected_groups", [])
    )
    rows = []
    for index, group in enumerate(groups, start=1):
        rows.append(
            {
                "route": route["route"],
                "order": index,
                "bayesian_train_scope": route["bayesian_train_scope"],
                "bayesian_selection_scope": route["bayesian_selection_scope"],
                **group,
            }
        )
    return rows


def fine_tune_diagnostic_row(route: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    summary = report.get("final_summary", {})
    rows = (
        report.get("bayesian_residual", {})
        .get("fine_tune", {})
        .get("search_rows", [])
    )
    row = rows[0] if rows else {}
    selected_groups = summary.get("selected_basis_groups", [])
    return {
        "route": route["route"],
        "route_label": route["label"],
        "bayesian_train_scope": route["bayesian_train_scope"],
        "bayesian_selection_scope": route["bayesian_selection_scope"],
        "fine_tune_train_scope": route["fine_tune_train_scope"],
        "B_train_role": route["b_role"],
        "stage1_anchor_method": summary.get("stage1_anchor_method"),
        "stage1_anchor_C_all_rmse_mm": summary.get("stage1_anchor_C_all_rmse_mm"),
        "bayesian_C_all_rmse_mm": summary.get("bayesian_before_finetune_C_all_rmse_mm"),
        "fine_tuned_C_all_rmse_mm": summary.get("bayesian_finetuned_C_all_rmse_mm"),
        "peeled_C_all_rmse_mm": summary.get("peeled_geometry_C_all_rmse_mm"),
        "fine_tune_lambda_ratio": summary.get("fine_tune_lambda_ratio"),
        "fine_tune_lambda_theta": summary.get("fine_tune_lambda_theta"),
        "fine_tune_accepted_by_C_all": summary.get("fine_tune_accepted_by_C_all"),
        "theta_update_scaled_l2": row.get("theta_update_scaled_l2"),
        "beta_l2_mm": row.get("beta_l2_mm"),
        "initial_beta_l2_mm": row.get("initial_beta_l2_mm"),
        "beta_update_l2_mm": row.get("beta_update_l2_mm"),
        "peeled_C_all_delta_vs_anchor_mm": row.get("peeled_C_all_delta_vs_anchor_mm"),
        "controlled_geometry": row.get("controlled_geometry"),
        "objective_initial": row.get("objective_initial"),
        "objective_final": row.get("objective_final"),
        "selected_basis_groups": ", ".join(str(item) for item in selected_groups),
    }


def build_summary(
    route_reports: dict[str, dict[str, Any]],
    metric_rows: list[dict[str, Any]],
    diagnostic_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = {
        (str(row["route"]), str(row["stage"])): row
        for row in metric_rows
    }
    ref = metrics.get(("AB_Bayes_AB_Finetune_reference", "bayesian_basis_fine_tuned_candidate"), {})
    a_only = metrics.get(("AB_Bayes_A_Finetune", "bayesian_basis_fine_tuned_candidate"), {})
    bselect = metrics.get(("A_Bayes_BSelect_A_Finetune", "bayesian_basis_fine_tuned_candidate"), {})
    ref_c = float_or_nan(ref.get("C_all_rmse_mm"))
    a_only_c = float_or_nan(a_only.get("C_all_rmse_mm"))
    bselect_c = float_or_nan(bselect.get("C_all_rmse_mm"))
    conclusion = []
    if is_finite(ref_c) and is_finite(a_only_c):
        conclusion.append(
            f"Stage 4 changed to A-only shifts C_all by {a_only_c - ref_c:+.6f} mm relative to the AB reference."
        )
    if is_finite(ref_c) and is_finite(bselect_c):
        conclusion.append(
            f"A-only Bayesian coefficients with B selection shifts C_all by {bselect_c - ref_c:+.6f} mm relative to the AB reference."
        )
    if is_finite(ref_c) and is_finite(bselect_c) and bselect_c > ref_c + 0.02:
        judgment = "B participation materially improves the reported B/C metrics; treat the AB reference as a fitted-space result, not leakage-free validation."
    elif is_finite(ref_c) and is_finite(bselect_c):
        judgment = "A-only residual fitting remains close to the AB reference; the improvement is not dominated by B coefficient leakage in this split."
    else:
        judgment = "Insufficient numeric data for a leakage judgment."
    return {
        "reference_fine_tuned_C_all_rmse_mm": ref_c,
        "AB_Bayes_A_Finetune_C_all_rmse_mm": a_only_c,
        "A_Bayes_BSelect_A_Finetune_C_all_rmse_mm": bselect_c,
        "judgment": judgment,
        "notes": conclusion,
        "route_count": len(route_reports),
        "diagnostic_count": len(diagnostic_rows),
    }


def render_leakage_report(report: dict[str, Any]) -> str:
    settings = report["settings"]
    summary = report["summary"]
    metrics = report["metrics"]
    groups = report["bayesian_selected_groups"]
    diagnostics = report["fine_tune_diagnostics"]
    route_table = [
        {
            "route": route["route"],
            "Bayesian coefficient fit": route["bayesian_train_scope"],
            "Bayesian selection": route["bayesian_selection_scope"],
            "Stage 4 fine-tune fit": route["fine_tune_train_scope"],
            "B_train role": route["b_role"],
        }
        for route in settings["routes"]
    ]
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>B 娉勯湶楠岃瘉锛歳atio=100000</title>
  <style>
    body {{ font-family: Arial, "Microsoft YaHei", sans-serif; margin: 28px; color: #1f2933; line-height: 1.55; }}
    h1, h2, h3 {{ color: #102a43; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }}
    th, td {{ border: 1px solid #d9e2ec; padding: 7px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #f0f4f8; }}
    code {{ background: #f0f4f8; padding: 1px 4px; border-radius: 3px; }}
    .note {{ background: #f8fafc; border-left: 4px solid #486581; padding: 10px 14px; margin: 12px 0 20px; }}
  </style>
</head>
<body>
  <h1>B 娉勯湶楠岃瘉锛歳atio=100000</h1>
  <div class="note">
    鏈姤鍛婂彧浣跨敤鐪熷疄鏁版嵁 <code>{escape(str(settings['train']))}</code> 鍒?    <code>{escape(str(settings['selection']))}</code>銆係tage 1 閿氱偣鍥哄畾涓?S1锛?    Stage 4 鍑犱綍姝ｅ垯姣斾緥鍥哄畾涓?<code>ratio=100000</code>锛屼笉鍋氫豢鐪熷疄楠屻€佷笉閲嶆柊鎼滅储 ratio銆?    C_all 娌跨敤褰撳墠瀹氫箟锛?code>C_all = A_C + B_C</code>銆?  </div>

  <h2>瀹為獙璺嚎涓庢暟鎹鑹?/h2>
  {html_table(route_table, None)}

  <h2>鏍稿績鍒ゆ柇</h2>
  <p>{escape(summary['judgment'])}</p>
  <ul>
    {''.join(f"<li>{escape(str(note))}</li>" for note in summary.get('notes', []))}
  </ul>

  <h2>A/B/C 鎸囨爣瀵规瘮</h2>
  <p>鍥涗釜闃舵鍒嗗埆鏄細Stage 1 鍑犱綍閿氱偣銆丅ayesian-only 琛ュ伩銆丅ayesian+鍏?33 鍙傛暟鑱斿悎寰皟銆佸墺绂昏ˉ鍋垮悗鐨勫嚑浣曟湰浣撱€傚娉勯湶鍒ゆ柇涓昏鐪?  <code>Bayesian+鍏ㄥ弬鏁板井璋?/code> 鍜?<code>鍓ョ琛ュ伩</code> 鍦?B_train銆丄_C銆丅_C銆丆_all 涓婃槸鍚﹀洜 B 閫€鍑烘嫙鍚堣€屾槑鏄鹃€€鍖栥€?/p>
  {html_table(metrics, ['route', 'stage', 'label', 'train_rmse_mm', 'A_train_rmse_mm', 'B_train_rmse_mm', 'A_C_rmse_mm', 'B_C_rmse_mm', 'C_all_rmse_mm'])}

  <h2>Stage 4 璇婃柇</h2>
  {html_table(diagnostics, ['route', 'bayesian_train_scope', 'bayesian_selection_scope', 'fine_tune_train_scope', 'fine_tuned_C_all_rmse_mm', 'peeled_C_all_rmse_mm', 'theta_update_scaled_l2', 'beta_l2_mm', 'beta_update_l2_mm', 'peeled_C_all_delta_vs_anchor_mm', 'controlled_geometry', 'selected_basis_groups'])}

  <h2>Bayesian 閫変腑鍑芥暟</h2>
  {html_table(groups, ['route', 'order', 'name', 'label', 'formula', 'columns', 'prior_std_mm'])}

  <h2>缁撹杈圭晫</h2>
  <ul>
    <li>濡傛灉 A-only 鍚?B_train/B_C/C_all 鏄庢樉鍙樺樊锛岃鏄庝笂涓€鏉?AB 涓荤嚎鐨?B/C 琛ㄧ幇鍖呭惈 B 杩涘叆鍚庝袱闃舵鎷熷悎甯︽潵鐨勬敹鐩娿€?/li>
    <li>濡傛灉 A-only 鍚?B/C 浠嶆帴杩?AB reference锛岃鏄庡綋鍓?split 涓嬬殑鏀剁泭涓昏鏉ヨ嚜 A 绌洪棿鍙鐢ㄧ殑娈嬪樊缁撴瀯锛岃€屼笉鏄?B 绯绘暟娉勯湶銆?/li>
    <li>B_validation 璺嚎涓紝B_train 浠嶇敤浜?basis/prior 閫夋嫨锛屽洜姝ゅ畠涓嶆槸涓ユ牸 untouched test锛涘畠鍙獙璇佲€滅郴鏁版嫙鍚堝拰 Stage 4 鍙傛暟鏇存柊涓嶄娇鐢?B鈥濄€?/li>
  </ul>
</body>
</html>"""


def append_to_reference_report(reference_path: Path, output_dir: Path, report: dict[str, Any]) -> None:
    if not reference_path.exists():
        return
    content = reference_path.read_text(encoding="utf-8")
    begin = "<!-- B_LEAKAGE_RATIO100000_BEGIN -->"
    end = "<!-- B_LEAKAGE_RATIO100000_END -->"
    if begin in content and end in content:
        start = content.index(begin)
        stop = content.index(end) + len(end)
        content = content[:start] + content[stop:]
    link = (output_dir / "report.html").resolve().as_uri()
    summary = report["summary"]
    rows = [
        row for row in report["metrics"]
        if row.get("stage") == "bayesian_basis_fine_tuned_candidate"
    ]
    section = f"""
{begin}
<section id="b-leakage-ratio100000">
  <h2>B 娉勯湶楠岃瘉锛歳atio=100000</h2>
  <p>鏂板涓夎矾绾垮姣旓紝妫€鏌?Bayesian 娈嬪樊鎷熷悎涓?Stage 4 鑱斿悎寰皟鐨勬敹鐩婃槸鍚︿緷璧?B_train 杩涘叆鍚庝袱闃舵鎷熷悎銆?  瀹屾暣鎶ュ憡锛?a href="{escape(link)}">{escape(link)}</a></p>
  <p>{escape(summary['judgment'])}</p>
  {html_table(rows, ['route', 'label', 'A_train_rmse_mm', 'B_train_rmse_mm', 'A_C_rmse_mm', 'B_C_rmse_mm', 'C_all_rmse_mm'])}
</section>
{end}
"""
    if "</body>" in content:
        content = content.replace("</body>", section + "\n</body>")
    else:
        content += section
    reference_path.write_text(content, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run B-leakage checks for the Bayesian calibration mainline.")
    parser.add_argument("--train", default="data/calibration/bayesian_calibration_pipeline/real_world200_nongeometric.pkl")
    parser.add_argument("--selection", default="data/calibration/bayesian_calibration_pipeline/real_world_normal50_nongeometric.pkl")
    parser.add_argument(
        "--output-dir",
        default="data/reports/bayesian_calibration_pipeline/real_world200_to_normal50_b_leakage_check_ratio100000",
    )
    parser.add_argument(
        "--append-report",
        default="data/reports/bayesian_calibration_pipeline/real_world200_to_normal50_unscaled_ratio_search/report.html",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--jacobian-method", default="analytic", choices=("analytic", "finite", "auto"))
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--lambda-count", type=int, default=13)
    parser.add_argument("--fine-count", type=int, default=7)
    parser.add_argument("--c-fraction", type=float, default=0.2)
    parser.add_argument("--fine-tune-max-nfev", type=int, default=80)
    parser.add_argument("--spectrum-permutations", type=int, default=100)
    parser.add_argument("--noise-std-mm", type=float, default=0.06)
    parser.add_argument("--stat-cv-folds", type=int, default=4)
    parser.add_argument("--max-basis-groups", type=int, default=4)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_cell(row.get(key)) for key in keys})


def csv_cell(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def html_table(rows: list[dict[str, Any]], keys: list[str] | None) -> str:
    if not rows:
        return "<p><em>鏃犳暟鎹?/em></p>"
    if keys is None:
        keys = []
        for row in rows:
            for key in row:
                if key not in keys:
                    keys.append(key)
    header = "".join(f"<th>{escape(str(key))}</th>" for key in keys)
    body = []
    for row in rows:
        cells = "".join(f"<td>{escape(format_cell(row.get(key)))}</td>" for key in keys)
        body.append(f"<tr>{cells}</tr>")
    return f"<table><tr>{header}</tr>{''.join(body)}</table>"


def format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(format_cell(item) for item in value)
    if isinstance(value, dict):
        return "; ".join(f"{key}={format_cell(val)}" for key, val in value.items())
    return "" if value is None else str(value)


def float_or_nan(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


if __name__ == "__main__":
    main()

