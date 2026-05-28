# Archived Bayesian Calibration Pipeline

This folder contains the archived mainline for real-data calibration with a
Geometry33 anchor, interpretable Bayesian non-geometric residual identification,
and all-33-geometry MAP fine-tuning. It does not import the repository's legacy
`calibration.*` modules; the core algorithms needed by the pipeline are copied
into this folder and use only `core.calibration.bayesian_calibration_pipeline.*` imports.

Install and run it from the repository root so the `core.*` package path and
`data/calibration/bayesian_calibration_pipeline/` datasets resolve correctly.

## Structure

- `core/`: robot model, parameter definitions, analytic Jacobian, SVD
  redundancy analysis, L2 LM, AB-balance lambda selection, identifiability
  weighting, dynamic weighting, subspace sequential weighting, data splitting,
  Bayesian basis residual modeling, and the real-data mainline.
- `configs/`: nominal robot constants and pipeline configuration.
- `experiments/`: command-line experiment and independence verification
  entry points.
- `reports/`: HTML/JSON report generation.

## Main Experiment

```powershell
python -B -m core.calibration.bayesian_calibration_pipeline.experiments.bayesian_real_experiment `
  --train data\calibration\bayesian_calibration_pipeline\real_world200_nongeometric.pkl `
  --selection data\calibration\bayesian_calibration_pipeline\real_world_normal50_nongeometric.pkl `
  --c-fraction 0.2 `
  --seed 20260524 `
  --jacobian-method analytic `
  --max-nfev 80 `
  --lambda-count 13 `
  --fine-count 7 `
  --stat-cv-folds 4 `
  --max-basis-groups 4 `
  --fine-tune-max-nfev 80 `
  --noise-std-mm 0.06 `
  --fine-tune-lambda-ratios 1000 `
  --output-dir data\reports\bayesian_calibration_pipeline\real_world200_to_normal50
```

Stage 1 geometry anchors are compared as:

- `M0`: SVD redundancy-filtered LM, no regularization.
- `M6`: SVD + uniform L2, AB-balance lambda.
- `W3`: all33 + global identifiability weighted L2, AB-balance lambda.
- `S1`: all33 + subspace-local identifiability weights, sequential fit.
- `D1`: SVD + dynamic pose-identifiability weights, AB-balance lambda.

Stage 1 and Stage 4 use unscaled position residuals:
`||r||^2 + lambda * prior`. The tracker noise default, `sigma = 0.06 mm`,
is used as the Bayesian residual noise prior and as a controlled fine-tune
threshold, not as a divisor in the geometry objectives. The final mainline is
fixed as Bayesian basis residual compensation plus all-33 geometry MAP
fine-tuning. The Stage 4 geometry prior is set dynamically from the selected
Stage 1 lambda as `lambda_theta = 1000 * lambda_stage1`; this ratio was fixed
after the unscaled-objective real-data lambda-ratio search with consistent
Bayesian prior scaling.
Analytic non-geometric blocks and RFF-GPR remain available in the older
diagnostic experiment, but they are not used for final mainline selection.

## Independence Verification

```powershell
python -B core\calibration\bayesian_calibration_pipeline\experiments\verify_independence.py --quick-smoke
```

The verifier scans the project for forbidden legacy imports and runs the
pipeline with a runtime import blocker that rejects `calibration.*`.


