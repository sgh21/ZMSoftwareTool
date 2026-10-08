"""窝深理论设置、误差边界及原始数据保留。"""

from copy import deepcopy
from statistics import fmean, pstdev, pvariance

import pytest

from core.services.feed_depth_service import (
    DEFAULT_SETTINGS, evaluate_feed_depth, load_feed_depth_settings,
    save_feed_depth_settings, validate_feed_depth_settings,
)


@pytest.fixture
def measurements():
    rows, history = [], []
    for batch_id, depths in (("B01", [1.3, 1.4, 1.5, 1.6, 1.7]), ("B02", [1.45, 1.46])):
        rows.extend({"batch_id": batch_id, "hole_id": f"{index + 1:03d}", "row_id": "R01",
                     "actual_depth_mm": depth, "theoretical_depth_mm": 1.5,
                     "data_source": "simulation", "is_simulated": True}
                    for index, depth in enumerate(depths))
        history.append({
            "batch_id": batch_id, "count": len(depths), "theoretical_depth_mm": 1.5,
            "mean_depth_mm": fmean(depths), "bias_mm": fmean(depths) - 1.5,
            "variance_mm2": pvariance(depths), "stddev_mm": pstdev(depths),
            "data_source": "simulation", "is_simulated": True,
        })
    return {"rows": rows, "history": history, "csv_path": "original.csv"}


def test_signed_limits_include_boundaries_and_do_not_hide_individual_alarms(measurements):
    result = evaluate_feed_depth(measurements, {"error_lower_mm": -0.1, "error_upper_mm": 0.1})
    assert [row["alarm"] for row in result["rows"][:5]] == [True, False, False, False, True]
    assert result["history"][0]["alarm"] is False
    assert result["history"][0]["alarm_count"] == 2
    assert result["history"][1]["alarm_count"] == 0
    unset = evaluate_feed_depth(measurements, DEFAULT_SETTINGS)
    assert not any(row["alarm"] for row in unset["rows"])
    assert not any(record["alarm"] or record["alarm_count"] for record in unset["history"])


def test_uniform_theory_updates_bias_without_rewriting_measurements(measurements):
    original = deepcopy(measurements)
    result = evaluate_feed_depth(measurements, {
        "depth_mode": "uniform", "uniform_depth": 1.6,
        "error_lower_mm": -0.1, "error_upper_mm": 0.1,
    })
    assert measurements == original
    for source, row in zip(original["rows"], result["rows"]):
        assert row["theoretical_depth_mm"] == 1.6
        assert row["actual_depth_mm"] == source["actual_depth_mm"]
        assert row["bias_mm"] == pytest.approx(source["actual_depth_mm"] - 1.6)
        assert row["is_simulated"] and row["data_source"] == "simulation"
    for source, record in zip(original["history"], result["history"]):
        assert record["theoretical_depth_mm"] == 1.6
        for key in ("mean_depth_mm", "variance_mm2", "stddev_mm", "count"):
            assert record[key] == source[key]
    assert result["history"][0]["bias_mm"] == pytest.approx(-0.1)
    assert result["history"][0]["alarm"] is False
    assert result["history"][1]["bias_mm"] == pytest.approx(-0.145)
    assert result["history"][1]["alarm"] is True
    restored = evaluate_feed_depth(measurements, DEFAULT_SETTINGS)
    assert all(row["theoretical_depth_mm"] == 1.5 for row in restored["rows"])


@pytest.mark.parametrize("settings", [
    {"depth_mode": "unknown"},
    {"depth_mode": "uniform", "uniform_depth": None},
    {"depth_mode": "uniform", "uniform_depth": 0},
    {"depth_mode": "uniform", "uniform_depth": float("nan")},
    {"depth_mode": "uniform", "uniform_depth": float("inf")},
    {"error_lower_mm": -0.1},
    {"error_upper_mm": 0.1},
    {"error_lower_mm": -0.1, "error_upper_mm": "invalid"},
    {"error_lower_mm": float("-inf"), "error_upper_mm": 0.1},
    {"error_lower_mm": -0.1, "error_upper_mm": float("nan")},
    {"error_lower_mm": 0.1, "error_upper_mm": -0.1},
])
def test_invalid_settings_are_rejected(settings):
    with pytest.raises(ValueError):
        validate_feed_depth_settings(settings)


def test_settings_persist_and_invalid_save_keeps_previous_file(tmp_path):
    path = tmp_path / "settings" / "feed_depth.json"
    defaults = load_feed_depth_settings(path)
    assert defaults == DEFAULT_SETTINGS and not path.exists()
    defaults["uniform_depth"] = 9.0
    assert load_feed_depth_settings(path)["uniform_depth"] == 1.5
    settings = save_feed_depth_settings(path, {
        "depth_mode": "uniform", "uniform_depth": "1.55",
        "error_lower_mm": "-0.02", "error_upper_mm": "0.03",
    })
    assert settings == {"depth_mode": "uniform", "uniform_depth": 1.55,
                        "error_lower_mm": -0.02, "error_upper_mm": 0.03}
    assert load_feed_depth_settings(path) == settings
    content = path.read_bytes()
    with pytest.raises(ValueError):
        save_feed_depth_settings(path, {"error_lower_mm": 0.2, "error_upper_mm": 0.1})
    assert path.read_bytes() == content
    assert validate_feed_depth_settings({"uniform_depth": None})["uniform_depth"] is None
    assert validate_feed_depth_settings({"error_lower_mm": 0, "error_upper_mm": 0})["error_upper_mm"] == 0.0
