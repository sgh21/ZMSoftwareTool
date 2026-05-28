"""Verify that the standalone Geometry33 project does not use legacy modules."""

from __future__ import annotations

import argparse
import importlib.abc
import json
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


FORBIDDEN_PATTERNS = (
    "from calibration.",
    "import calibration.",
    "from config.",
    "import config.",
)


class LegacyImportBlocker(importlib.abc.MetaPathFinder):
    """Reject imports from legacy project packages during runtime verification."""

    def find_spec(self, fullname: str, path=None, target=None):  # type: ignore[override]
        if fullname == "calibration" or fullname.startswith("calibration."):
            raise ImportError(f"Forbidden legacy import: {fullname}")
        if fullname == "config" or fullname.startswith("config."):
            raise ImportError(f"Forbidden legacy import: {fullname}")
        return None


def scan_forbidden_imports(project_root: Path) -> list[dict[str, object]]:
    """Return source locations that still reference legacy project imports."""
    violations: list[dict[str, object]] = []
    for path in sorted(project_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith(FORBIDDEN_PATTERNS):
                violations.append(
                    {
                        "path": str(path.relative_to(REPO_ROOT)),
                        "line": line_no,
                        "text": stripped,
                    }
                )
    return violations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick-smoke", action="store_true", help="Run a quick pipeline smoke test.")
    parser.add_argument(
        "--output-dir",
        default="data/reports/bayesian_calibration_pipeline/standalone/independence_smoke",
        help="Output directory for the optional quick smoke run.",
    )
    args = parser.parse_args()

    violations = scan_forbidden_imports(PACKAGE_ROOT)
    if violations:
        print(json.dumps({"status": "failed", "violations": violations}, indent=2, ensure_ascii=False))
        raise SystemExit(1)

    sys.meta_path.insert(0, LegacyImportBlocker())
    from core.calibration.bayesian_calibration_pipeline import Geometry33PipelineConfig, run_real_ablation
    from core.calibration.bayesian_calibration_pipeline.reports.html_report import write_outputs

    result: dict[str, object] = {
        "status": "passed",
        "forbidden_import_violations": 0,
        "runtime_import_blocker": "enabled",
    }
    if args.quick_smoke:
        config = Geometry33PipelineConfig(quick=True, output_dir=args.output_dir)
        report = run_real_ablation(config)
        write_outputs(report)
        result["quick_smoke_output_dir"] = args.output_dir
        result["method_order"] = report["method_order"]
        result["scenario_method_counts"] = [
            len(scenario["methods"])
            for scenario in report["scenarios"]
        ]
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()


