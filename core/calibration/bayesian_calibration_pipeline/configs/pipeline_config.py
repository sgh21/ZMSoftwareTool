"""Configuration for the stable geometry33 real-data ablation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


METHOD_ORDER = ("M0", "M1", "M6", "W3", "S1", "D1")

METHOD_LABELS = {
    "M0": "SVD active LM, no regularization",
    "M1": "SVD active + uniform L2, random CV lambda",
    "M6": "SVD active + uniform L2, A/B-balance lambda",
    "W3": "all33 + global identifiability weighted L2, A/B-balance lambda",
    "S1": "all33 + subspace-local identifiability weights, sequential fit",
    "D1": "SVD active + dynamic pose-identifiability weights",
}


@dataclass
class Geometry33PipelineConfig:
    """Decision-complete defaults for the mature Bayesian Calibration Pipeline."""

    real_a: Path = Path("data/calibration/bayesian_calibration_pipeline/real_world200.pkl")
    real_b: Path = Path("data/calibration/bayesian_calibration_pipeline/real_world_normal50.pkl")
    output_dir: Path = Path("data/reports/bayesian_calibration_pipeline/real_ablation")
    real_c_fraction: float = 0.2
    seed: int = 20260524
    jacobian_method: str = "analytic"
    max_nfev: int = 80
    cv_folds: int = 3
    lambda_min_power: float = -10.0
    lambda_max_power: float = 2.0
    lambda_count: int = 13
    fine_count: int = 7
    ab_balance_alpha: float = 1.0
    redundancy_tolerance: float = 1.0e-7
    redundancy_max_combinations: int = 200_000
    rho_threshold: float = 0.5
    kappa_threshold: float = 0.05
    risk_beta: float = 0.0
    risk_power: float = 1.0
    min_weight: float = 1.0
    max_weight: float = 100.0
    strong_weight: float = 10_000.0
    dynamic_risk_quantile: float = 0.25
    dynamic_outer_iterations: int = 3
    dynamic_convergence_tol: float = 1.0e-3
    subspace_min_cluster_size: int = 10
    subspace_k_candidates: tuple[int, ...] = (2, 3, 4)
    quick: bool = False

    def quickened(self) -> "Geometry33PipelineConfig":
        """Return a copy with small settings for smoke validation."""
        if not self.quick:
            return self
        return Geometry33PipelineConfig(
            real_a=self.real_a,
            real_b=self.real_b,
            output_dir=self.output_dir,
            real_c_fraction=self.real_c_fraction,
            seed=self.seed,
            jacobian_method=self.jacobian_method,
            max_nfev=min(self.max_nfev, 20),
            cv_folds=min(self.cv_folds, 2),
            lambda_min_power=self.lambda_min_power,
            lambda_max_power=self.lambda_max_power,
            lambda_count=min(self.lambda_count, 3),
            fine_count=min(self.fine_count, 2),
            ab_balance_alpha=self.ab_balance_alpha,
            redundancy_tolerance=self.redundancy_tolerance,
            redundancy_max_combinations=self.redundancy_max_combinations,
            rho_threshold=self.rho_threshold,
            kappa_threshold=self.kappa_threshold,
            risk_beta=self.risk_beta,
            risk_power=self.risk_power,
            min_weight=self.min_weight,
            max_weight=self.max_weight,
            strong_weight=self.strong_weight,
            dynamic_risk_quantile=self.dynamic_risk_quantile,
            dynamic_outer_iterations=min(self.dynamic_outer_iterations, 2),
            dynamic_convergence_tol=self.dynamic_convergence_tol,
            subspace_min_cluster_size=min(self.subspace_min_cluster_size, 4),
            subspace_k_candidates=self.subspace_k_candidates,
            quick=True,
        )

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-friendly settings dictionary."""
        output = dict(self.__dict__)
        output["real_a"] = str(self.real_a)
        output["real_b"] = str(self.real_b)
        output["output_dir"] = str(self.output_dir)
        output["subspace_k_candidates"] = list(self.subspace_k_candidates)
        return output


