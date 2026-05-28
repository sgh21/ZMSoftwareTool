"""CLI for the stable geometry33 real-data ablation pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.calibration.bayesian_calibration_pipeline.configs.pipeline_config import Geometry33PipelineConfig
from core.calibration.bayesian_calibration_pipeline.core.pipeline import run_real_ablation
from core.calibration.bayesian_calibration_pipeline.reports.html_report import write_outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the mature geometry33 identifiability-regularized real-data ablation."
    )
    parser.add_argument("--real-a", default="data/calibration/bayesian_calibration_pipeline/real_world200.pkl")
    parser.add_argument("--real-b", default="data/calibration/bayesian_calibration_pipeline/real_world_normal50.pkl")
    parser.add_argument("--output-dir", default="data/reports/bayesian_calibration_pipeline/real_ablation")
    parser.add_argument("--real-c-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260524)
    parser.add_argument(
        "--jacobian-method",
        choices=("auto", "analytic", "finite_difference"),
        default="analytic",
    )
    parser.add_argument("--max-nfev", type=int, default=80)
    parser.add_argument("--cv-folds", type=int, default=3)
    parser.add_argument("--lambda-min-power", type=float, default=-10.0)
    parser.add_argument("--lambda-max-power", type=float, default=2.0)
    parser.add_argument("--lambda-count", type=int, default=13)
    parser.add_argument("--fine-count", type=int, default=7)
    parser.add_argument("--ab-balance-alpha", type=float, default=1.0)
    parser.add_argument("--redundancy-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--redundancy-max-combinations", type=int, default=200_000)
    parser.add_argument("--rho-threshold", type=float, default=0.5)
    parser.add_argument("--kappa-threshold", type=float, default=0.05)
    parser.add_argument("--risk-beta", type=float, default=0.0)
    parser.add_argument("--risk-power", type=float, default=1.0)
    parser.add_argument("--min-weight", type=float, default=1.0)
    parser.add_argument("--max-weight", type=float, default=100.0)
    parser.add_argument("--strong-weight", type=float, default=10_000.0)
    parser.add_argument("--dynamic-risk-quantile", type=float, default=0.25)
    parser.add_argument("--dynamic-outer-iterations", type=int, default=3)
    parser.add_argument("--dynamic-convergence-tol", type=float, default=1.0e-3)
    parser.add_argument("--subspace-min-cluster-size", type=int, default=10)
    parser.add_argument("--subspace-k-candidates", default="2,3,4")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    config = Geometry33PipelineConfig(
        real_a=Path(args.real_a),
        real_b=Path(args.real_b),
        output_dir=Path(args.output_dir),
        real_c_fraction=args.real_c_fraction,
        seed=args.seed,
        jacobian_method=args.jacobian_method,
        max_nfev=args.max_nfev,
        cv_folds=args.cv_folds,
        lambda_min_power=args.lambda_min_power,
        lambda_max_power=args.lambda_max_power,
        lambda_count=args.lambda_count,
        fine_count=args.fine_count,
        ab_balance_alpha=args.ab_balance_alpha,
        redundancy_tolerance=args.redundancy_tolerance,
        redundancy_max_combinations=args.redundancy_max_combinations,
        rho_threshold=args.rho_threshold,
        kappa_threshold=args.kappa_threshold,
        risk_beta=args.risk_beta,
        risk_power=args.risk_power,
        min_weight=args.min_weight,
        max_weight=args.max_weight,
        strong_weight=args.strong_weight,
        dynamic_risk_quantile=args.dynamic_risk_quantile,
        dynamic_outer_iterations=args.dynamic_outer_iterations,
        dynamic_convergence_tol=args.dynamic_convergence_tol,
        subspace_min_cluster_size=args.subspace_min_cluster_size,
        subspace_k_candidates=_parse_k_candidates(args.subspace_k_candidates),
        quick=bool(args.quick),
    )
    report = run_real_ablation(config)
    write_outputs(report)
    print(f"Wrote {Path(report['artifacts']['html']).resolve()}")
    print(f"Wrote {Path(report['artifacts']['json']).resolve()}")
    print(f"Wrote {Path(report['artifacts']['notes']).resolve()}")


def _parse_k_candidates(value: str) -> tuple[int, ...]:
    parsed = tuple(sorted({int(part.strip()) for part in str(value).split(",") if part.strip()}))
    return parsed or (2, 3, 4)


if __name__ == "__main__":
    main()


