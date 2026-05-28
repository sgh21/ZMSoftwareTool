"""CLI for the real-data Bayesian non-geometric calibration mainline."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from core.calibration.bayesian_calibration_pipeline.configs.pipeline_config import Geometry33PipelineConfig
from core.calibration.bayesian_calibration_pipeline.core.bayesian_mainline import (
    json_ready,
    render_bayesian_html_report,
    run_bayesian_calibration_analysis,
)
from core.calibration.bayesian_calibration_pipeline.core.data_io import load_dataset
from core.calibration.bayesian_calibration_pipeline.core.non_geometric import NonGeometricConfig
from core.calibration.bayesian_calibration_pipeline.core.statistical_residual import StatisticalResidualConfig


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    geometry_config = Geometry33PipelineConfig(
        real_a=Path(args.train),
        real_b=Path(args.selection),
        output_dir=output_dir,
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
        fine_tune_lambda_ratios=parse_ratio_list(args.fine_tune_lambda_ratios),
    )
    report = json_ready(report)

    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "report.html").write_text(
        render_bayesian_html_report(report),
        encoding="utf-8",
    )

    bayes = report["bayesian_residual"]
    basis = bayes["models"]["bayesian_basis"]
    write_csv(
        output_dir / "stage1_geometry_split_metrics.csv",
        report["stage1_geometry"].get("split_metrics", []),
    )
    write_csv(
        output_dir / "bayesian_candidate_library.csv",
        bayes.get("candidate_library", []),
    )
    write_csv(
        output_dir / "bayesian_basis_search.csv",
        basis.get("search_rows", []),
    )
    write_csv(
        output_dir / "bayesian_selected_groups.csv",
        basis.get("selected_groups", []),
    )
    write_csv(
        output_dir / "bayesian_stage_metrics.csv",
        report.get("bayesian_stage_metrics", []),
    )
    write_csv(
        output_dir / "fine_tuned_geometry_only_metrics.csv",
        report.get("fine_tuned_geometry_only", {}).get("comparison_rows", []),
    )
    write_csv(
        output_dir / "fine_tune_lambda_search.csv",
        bayes.get("fine_tune", {}).get("search_rows", []),
    )

    summary = report["final_summary"]
    print("Bayesian calibration mainline complete.")
    print(f"  output: {output_dir.resolve()}")
    print(f"  anchor: {summary['stage1_anchor_method']}")
    print(f"  selected basis groups: {', '.join(summary['selected_basis_groups']) or 'none'}")
    print(
        "  C_all RMSE mm: "
        f"anchor={summary['stage1_anchor_C_all_rmse_mm']:.6f}, "
        f"bayesian={summary['bayesian_before_finetune_C_all_rmse_mm']:.6f}, "
        f"fine_tuned={summary['bayesian_finetuned_C_all_rmse_mm']:.6f}, "
        f"peeled_geometry={summary['peeled_geometry_C_all_rmse_mm']:.6f}"
    )
    print(
        "  fine-tune: "
        f"accepted_by_C_all={summary['fine_tune_accepted_by_C_all']}, "
        f"objective_decreased={summary['fine_tune_objective_decreased']}, "
        f"released_geometry_params={summary['released_geometry_parameter_count']}, "
        f"lambda_ratio={summary['fine_tune_lambda_ratio']:.6g}, "
        f"lambda_theta={summary['fine_tune_lambda_theta']:.6g}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Bayesian basis non-geometric calibration on enhanced real datasets."
    )
    parser.add_argument(
        "--train",
        default="data/calibration/bayesian_calibration_pipeline/real_world200_nongeometric.pkl",
        help="A-space enhanced pkl dataset.",
    )
    parser.add_argument(
        "--selection",
        default="data/calibration/bayesian_calibration_pipeline/real_world_normal50_nongeometric.pkl",
        help="B-space enhanced pkl dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/reports/bayesian_calibration_pipeline/real_world200_to_normal50",
        help="Output directory for report.html, report.json and CSV tables.",
    )
    parser.add_argument("--quick", action="store_true", help="Run a reduced smoke validation.")
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
    parser.add_argument(
        "--fine-tune-lambda-ratios",
        default="1000",
        help="Comma-separated ratio grid for lambda_theta = stage1_anchor_lambda * ratio.",
    )
    return parser.parse_args()


def parse_ratio_list(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(text).split(",") if item.strip())
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("--fine-tune-lambda-ratios must contain positive values")
    return values


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row.keys():
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


if __name__ == "__main__":
    main()

