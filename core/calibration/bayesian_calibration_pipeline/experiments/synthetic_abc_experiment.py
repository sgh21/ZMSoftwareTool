"""Synthetic A/B/C validation for the Bayesian non-geometric mainline."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from core.calibration.bayesian_calibration_pipeline.configs.nominal_robot import NOMINAL_ROBOT
from core.calibration.bayesian_calibration_pipeline.configs.pipeline_config import Geometry33PipelineConfig
from core.calibration.bayesian_calibration_pipeline.core.bayesian_mainline import (
    json_ready,
    render_bayesian_html_report,
    run_bayesian_calibration_analysis,
)
from core.calibration.bayesian_calibration_pipeline.core.geometric import select_geometric_parameters
from core.calibration.bayesian_calibration_pipeline.core.non_geometric import NonGeometricConfig
from core.calibration.bayesian_calibration_pipeline.core.parameters import (
    build_error_parameters,
    parameter_scales,
    sample_truth_vector,
    vector_to_named_dict,
)
from core.calibration.bayesian_calibration_pipeline.core.robot_model import MultiSourceRobotModel
from core.calibration.bayesian_calibration_pipeline.core.statistical_residual import StatisticalResidualConfig


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    data_dir = output_dir / "datasets"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    model = MultiSourceRobotModel()
    parameters = build_error_parameters()
    geometric_names = {parameter.name for parameter in select_geometric_parameters(parameters)}
    geometric_mask = np.asarray([parameter.name in geometric_names for parameter in parameters], dtype=bool)
    scales = parameter_scales(parameters)

    shared_truth = sample_truth_vector(rng, parameters, sigma_scale=float(args.truth_scale))
    shared_geometry = shared_truth.copy()
    shared_geometry[~geometric_mask] = 0.0

    truth_vectors = {
        name: make_space_truth(
            rng,
            shared_truth,
            shared_geometry,
            geometric_mask,
            scales,
            nongeometric_jitter=float(args.nongeometric_jitter),
        )
        for name in ("A", "B", "C")
    }

    datasets = {
        "A": make_dataset(
            rng,
            model,
            parameters,
            truth_vectors["A"],
            sample_joints(rng, int(args.samples_a), "A"),
            payload=float(args.payload),
            noise_std_m=float(args.noise_std_mm) * 1.0e-3,
            name="synthetic_A_train",
        ),
        "B": make_dataset(
            rng,
            model,
            parameters,
            truth_vectors["B"],
            sample_joints(rng, int(args.samples_b), "B"),
            payload=float(args.payload),
            noise_std_m=float(args.noise_std_mm) * 1.0e-3,
            name="synthetic_B_selection",
        ),
        "C": make_dataset(
            rng,
            model,
            parameters,
            truth_vectors["C"],
            sample_joints(rng, int(args.samples_c), "C"),
            payload=float(args.payload),
            noise_std_m=float(args.noise_std_mm) * 1.0e-3,
            name="synthetic_C_external",
        ),
    }

    for name, dataset in datasets.items():
        with (data_dir / f"synthetic_{name}.pkl").open("wb") as file:
            pickle.dump(dataset, file)

    geometry_config = Geometry33PipelineConfig(
        output_dir=output_dir,
        seed=int(args.seed),
        jacobian_method=str(args.jacobian_method),
        max_nfev=int(args.max_nfev),
        lambda_count=int(args.lambda_count),
        fine_count=int(args.fine_count),
        real_c_fraction=0.2,
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
        datasets["A"],
        datasets["B"],
        geometry_config,
        non_geo_config,
        statistical_config,
        fine_tune_lambda_ratios=parse_ratio_list(args.fine_tune_lambda_ratios),
        external_c=datasets["C"],
    )
    report = json_ready(report)
    report["synthetic_truth"] = synthetic_truth_summary(
        parameters,
        truth_vectors,
        geometric_mask,
        shared_geometry,
        float(args.noise_std_mm),
        float(args.payload),
    )
    report["synthetic_datasets"] = {
        name: {
            "samples": int(len(dataset["joints"])),
            "path": str((data_dir / f"synthetic_{name}.pkl").resolve()),
        }
        for name, dataset in datasets.items()
    }

    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "report.html").write_text(
        render_bayesian_html_report(report),
        encoding="utf-8",
    )
    (output_dir / "truth_summary.json").write_text(
        json.dumps(report["synthetic_truth"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = report["final_summary"]
    print("Synthetic A/B/C Bayesian calibration complete.")
    print(f"  output: {output_dir.resolve()}")
    print(f"  anchor: {summary['stage1_anchor_method']}")
    print(
        "  C_all RMSE mm: "
        f"anchor={summary['stage1_anchor_C_all_rmse_mm']:.6f}, "
        f"bayesian={summary['bayesian_before_finetune_C_all_rmse_mm']:.6f}, "
        f"fine_tuned={summary['bayesian_finetuned_C_all_rmse_mm']:.6f}, "
        f"peeled_geometry={summary['peeled_geometry_C_all_rmse_mm']:.6f}"
    )
    print(
        "  fine-tune: "
        f"ratio={summary['fine_tune_lambda_ratio']:.6g}, "
        f"lambda_theta={summary['fine_tune_lambda_theta']:.6g}"
    )


def make_space_truth(
    rng: np.random.Generator,
    shared_truth: np.ndarray,
    shared_geometry: np.ndarray,
    geometric_mask: np.ndarray,
    scales: np.ndarray,
    *,
    nongeometric_jitter: float,
) -> np.ndarray:
    truth = np.asarray(shared_truth, dtype=float).copy()
    truth[geometric_mask] = shared_geometry[geometric_mask]
    jitter = rng.normal(0.0, scales * float(nongeometric_jitter))
    truth[~geometric_mask] = shared_truth[~geometric_mask] + jitter[~geometric_mask]
    return truth


def sample_joints(rng: np.random.Generator, count: int, space: str) -> np.ndarray:
    limits = np.asarray(NOMINAL_ROBOT["joint_limits"], dtype=float)
    ranges = {
        "A": np.array(
            [[-2.8, -0.4], [-2.35, -1.15], [-2.35, -0.8], [-2.2, 0.8], [-2.1, 1.2], [-np.pi, np.pi]]
        ),
        "B": np.array(
            [[0.35, 2.8], [-1.55, -0.45], [-1.25, 0.05], [-0.6, 2.2], [-1.4, 2.4], [-np.pi, np.pi]]
        ),
        "C": np.array(
            [[-0.8, 1.1], [-2.25, -0.55], [-2.55, -0.15], [0.7, 2.8], [-2.8, 0.4], [-np.pi, np.pi]]
        ),
    }[space]
    low = np.maximum(ranges[:, 0], limits[:, 0])
    high = np.minimum(ranges[:, 1], limits[:, 1])
    return rng.uniform(low, high, size=(int(count), 6))


def make_dataset(
    rng: np.random.Generator,
    model: MultiSourceRobotModel,
    parameters: list[Any],
    truth_vector: np.ndarray,
    joints: np.ndarray,
    *,
    payload: float,
    noise_std_m: float,
    name: str,
) -> dict[str, Any]:
    count = len(joints)
    payloads = np.full(count, float(payload), dtype=float)
    directions = rng.choice(np.array([-1.0, 1.0]), size=(count, 6))
    true_positions = model.batch_positions(joints, truth_vector, parameters, payloads, directions)
    measured = true_positions + rng.normal(0.0, float(noise_std_m), size=true_positions.shape)
    payload_torques = np.asarray(
        [model.joint_load_torque(q, float(payload)) for q in joints],
        dtype=float,
    )
    self_weight_torques = np.zeros_like(payload_torques)
    return {
        "name": name,
        "joints": joints,
        "measured_positions": measured,
        "true_positions": true_positions,
        "payloads": payloads,
        "directions": directions,
        "joint_torques": self_weight_torques + payload_torques,
        "self_weight_joint_torques": self_weight_torques,
        "payload_joint_torques": payload_torques,
        "true_error_vector": np.asarray(truth_vector, dtype=float),
        "true_error_parameters": vector_to_named_dict(truth_vector, parameters),
        "parameter_names": [parameter.name for parameter in parameters],
    }


def synthetic_truth_summary(
    parameters: list[Any],
    truth_vectors: dict[str, np.ndarray],
    geometric_mask: np.ndarray,
    shared_geometry: np.ndarray,
    noise_std_mm: float,
    payload: float,
) -> dict[str, Any]:
    names = [parameter.name for parameter in parameters]
    scales = parameter_scales(parameters)
    geometry_max_delta = max(
        float(np.max(np.abs(truth_vectors[name][geometric_mask] - shared_geometry[geometric_mask])))
        for name in truth_vectors
    )
    nongeo_pairs: list[dict[str, float | str]] = []
    for left, right in (("A", "B"), ("A", "C"), ("B", "C")):
        delta = truth_vectors[left][~geometric_mask] - truth_vectors[right][~geometric_mask]
        scale = scales[~geometric_mask]
        nongeo_pairs.append(
            {
                "pair": f"{left}-{right}",
                "nongeometric_scaled_l2": float(np.linalg.norm(delta / np.maximum(scale, 1.0e-20))),
                "nongeometric_scaled_max": float(np.max(np.abs(delta / np.maximum(scale, 1.0e-20)))),
            }
        )
    return {
        "noise_std_mm": float(noise_std_mm),
        "payload_kg": float(payload),
        "truth_model": "54-dimensional multisource error model",
        "geometric_parameter_rule": "33 geometry candidates share identical truth across A/B/C",
        "nongeometric_parameter_rule": "non-geometry parameters are jittered by workspace",
        "geometric_max_abs_delta_across_spaces": geometry_max_delta,
        "nongeometric_pair_differences": nongeo_pairs,
        "geometric_names": [name for name, is_geo in zip(names, geometric_mask) if is_geo],
        "nongeometric_names": [name for name, is_geo in zip(names, geometric_mask) if not is_geo],
    }


def parse_ratio_list(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(text).split(",") if item.strip())
    if not values or any(value <= 0.0 for value in values):
        raise ValueError("--fine-tune-lambda-ratios must contain positive values")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/reports/bayesian_calibration_pipeline/synthetic_abc")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seed", type=int, default=20260525)
    parser.add_argument("--samples-a", type=int, default=90)
    parser.add_argument("--samples-b", type=int, default=70)
    parser.add_argument("--samples-c", type=int, default=60)
    parser.add_argument("--payload", type=float, default=20.0)
    parser.add_argument("--noise-std-mm", type=float, default=0.06)
    parser.add_argument("--truth-scale", type=float, default=1.0)
    parser.add_argument("--nongeometric-jitter", type=float, default=0.45)
    parser.add_argument("--jacobian-method", default="analytic", choices=("analytic", "finite", "auto"))
    parser.add_argument("--max-nfev", type=int, default=60)
    parser.add_argument("--lambda-count", type=int, default=9)
    parser.add_argument("--fine-count", type=int, default=5)
    parser.add_argument("--fine-tune-max-nfev", type=int, default=60)
    parser.add_argument("--spectrum-permutations", type=int, default=50)
    parser.add_argument("--stat-cv-folds", type=int, default=3)
    parser.add_argument("--max-basis-groups", type=int, default=4)
    parser.add_argument("--fine-tune-lambda-ratios", default="1000")
    return parser.parse_args()


if __name__ == "__main__":
    main()

