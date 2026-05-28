from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HealthStatus:
    score: float
    level: str
    message: str


def evaluate_health(rms_error: float, over_tolerance_rate: float, confidence: float) -> HealthStatus:
    score = 100.0
    score -= min(max(rms_error, 0.0) * 1000.0, 35.0)
    score -= min(max(over_tolerance_rate, 0.0) * 100.0, 35.0)
    score -= min(max(1.0 - confidence, 0.0) * 30.0, 30.0)
    score = max(0.0, min(100.0, score))
    level = "good" if score >= 80 else "warning" if score >= 60 else "critical"
    return HealthStatus(score=score, level=level, message=f"health={level}, score={score:.1f}")
