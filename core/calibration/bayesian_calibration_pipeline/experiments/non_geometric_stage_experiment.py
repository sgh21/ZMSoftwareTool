"""CLI for the world200 -> normal50 non-geometric stage experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from core.calibration.bayesian_calibration_pipeline.configs.pipeline_config import Geometry33PipelineConfig
from core.calibration.bayesian_calibration_pipeline.core.data_io import load_dataset
from core.calibration.bayesian_calibration_pipeline.core.non_geometric import (
    NonGeometricConfig,
    json_ready,
    render_html_report,
    run_full_nongeometric_analysis,
)
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
        alpha=float(args.alpha),
        eta_direct=float(args.eta_direct),
        eta_project=float(args.eta_project),
        harmonics=tuple(int(item) for item in args.harmonics.split(",") if item.strip()),
        nonlinear_max_nfev=int(args.nonlinear_max_nfev),
        spectrum_permutations=int(args.spectrum_permutations),
        seed=int(args.seed),
    )
    statistical_config = StatisticalResidualConfig(
        noise_std_m=float(args.noise_std_mm) * 1.0e-3,
        cv_folds=int(args.stat_cv_folds),
        seed=int(args.seed),
        max_basis_groups=int(args.max_basis_groups),
    )

    train = load_dataset(args.train)
    selection = load_dataset(args.selection)
    report = run_full_nongeometric_analysis(
        train,
        selection,
        geometry_config,
        non_geo_config,
        statistical_config,
    )
    report = json_ready(report)

    report_json = output_dir / "report.json"
    report_html = output_dir / "report.html"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report_html.write_text(render_html_report(report), encoding="utf-8")

    write_csv(output_dir / "stage1_geometry_baselines.csv", report["stage1_geometry"]["methods"])
    write_csv(
        output_dir / "stage2_candidate_blocks.csv",
        report["stage2_candidates"]["blocks"],
    )
    write_csv(
        output_dir / "stage2_forward_selection.csv",
        report["stage2_candidates"]["selection_trace"],
    )
    write_csv(
        output_dir / "stage2_selected_blocks.csv",
        report["stage2_candidates"]["selected_blocks"],
    )
    write_csv(
        output_dir / "stage3_summary.csv",
        [
            {"path": "FWL", **report["stage3"]["fwl"]},
            {"path": "nonlinear_lm", **report["stage3"]["nonlinear_lm"]},
        ],
    )
    write_csv(
        output_dir / "stage1_geometry_split_metrics.csv",
        report["stage1_geometry"].get("split_metrics", []),
    )
    statistical = report.get("statistical_residual", {})
    write_csv(
        output_dir / "stat_model_comparison.csv",
        [
            {
                "method": "geometry_anchor",
                **statistical.get("geometry_anchor_metrics", {}),
                "reason": "Stage 1 anchor",
            }
        ]
        + statistical.get("model_comparison", []),
    )
    write_csv(
        output_dir / "stat_basis_search.csv",
        statistical.get("models", {}).get("bayesian_basis", {}).get("search_rows", []),
    )
    write_csv(
        output_dir / "stat_basis_selected_groups.csv",
        statistical.get("models", {}).get("bayesian_basis", {}).get("selected_groups", []),
    )
    write_csv(
        output_dir / "stat_rff_search.csv",
        statistical.get("models", {}).get("rff_gpr", {}).get("search_rows", []),
    )
    write_csv(
        output_dir / "stat_fine_tune.csv",
        statistical.get("fine_tune", {}).get("rows", []),
    )

    summary = report["final_summary"]
    print("Non-geometric stage experiment complete.")
    print(f"  output: {output_dir.resolve()}")
    print(f"  anchor: {summary['stage1_anchor_method']}")
    print(f"  final path: {summary['final_path']}")
    print(
        "  A identification RMSE mm: "
        f"anchor={summary['stage1_anchor_A_rmse_mm']:.6f}, "
        f"FWL={summary['fwl_A_rmse_mm']:.6f}, "
        f"nonlinear_aug={summary['nonlinear_augmented_A_rmse_mm']:.6f}, "
        f"final={summary['final_A_rmse_mm']:.6f}"
    )
    print(
        "  B reference RMSE mm: "
        f"anchor={summary['stage1_anchor_B_rmse_mm']:.6f}, "
        f"FWL={summary['fwl_B_rmse_mm']:.6f}, "
        f"nonlinear_aug={summary['nonlinear_augmented_B_rmse_mm']:.6f}, "
        f"final={summary['final_B_reference_rmse_mm']:.6f}"
    )
    print(f"  selected blocks: {', '.join(summary['selected_blocks']) or 'none'}")
    print(f"  whiteness A identification passed: {summary['whiteness_identification_passed']}")
    print(
        "  statistical C_all RMSE mm: "
        f"anchor={summary['statistical_geometry_anchor_C_all_rmse_mm']:.6f}, "
        f"final={summary['statistical_final_C_all_rmse_mm']:.6f}, "
        f"method={summary['statistical_final_method']}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run non-geometric Stage 0/1/2/3 experiment on enhanced real pkl data."
    )
    parser.add_argument(
        "--train",
        default="data/calibration/bayesian_calibration_pipeline/real_world200_nongeometric.pkl",
        help="A-space enhanced pkl dataset.",
    )
    parser.add_argument(
        "--selection",
        default="data/calibration/bayesian_calibration_pipeline/real_world_normal50_nongeometric.pkl",
        help="B-selection enhanced pkl dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/reports/bayesian_calibration_pipeline/non_geometric_stage_experiment/world200_to_normal50",
        help="Output directory for report.html, report.json and stage CSV tables.",
    )
    parser.add_argument("--quick", action="store_true", help="Run a small smoke validation.")
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument("--jacobian-method", default="analytic", choices=("analytic", "finite", "auto"))
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--lambda-count", type=int, default=13)
    parser.add_argument("--fine-count", type=int, default=7)
    parser.add_argument("--c-fraction", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--eta-direct", type=float, default=0.30)
    parser.add_argument("--eta-project", type=float, default=0.10)
    parser.add_argument("--harmonics", default="1,2,3,4")
    parser.add_argument("--nonlinear-max-nfev", type=int, default=80)
    parser.add_argument("--spectrum-permutations", type=int, default=100)
    parser.add_argument("--noise-std-mm", type=float, default=0.05)
    parser.add_argument("--stat-cv-folds", type=int, default=4)
    parser.add_argument("--max-basis-groups", type=int, default=4)
    return parser.parse_args()


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


