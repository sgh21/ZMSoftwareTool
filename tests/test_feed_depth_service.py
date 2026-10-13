"""窝深理论设置、误差边界及原始数据保留。"""

from copy import deepcopy
import json
from statistics import fmean, pstdev, pvariance

import pytest

from core.services.feed_depth_service import (
    DEFAULT_SETTINGS, evaluate_feed_depth, load_feed_depth_data, load_feed_depth_settings,
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


def test_symmetric_tolerance_includes_boundaries_and_does_not_hide_individual_alarms(measurements):
    result = evaluate_feed_depth(measurements, {"error_tolerance_mm": 0.1})
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
        "error_tolerance_mm": 0.1,
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
    {"error_tolerance_mm": -0.1},
    {"error_tolerance_mm": "invalid"},
    {"error_tolerance_mm": float("-inf")},
    {"error_tolerance_mm": float("inf")},
    {"error_tolerance_mm": float("nan")},
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
        "error_tolerance_mm": "0.03",
    })
    assert settings == {"depth_mode": "uniform", "uniform_depth": 1.55,
                        "error_tolerance_mm": 0.03}
    assert load_feed_depth_settings(path) == settings
    content = path.read_bytes()
    with pytest.raises(ValueError):
        save_feed_depth_settings(path, {"error_tolerance_mm": -0.1})
    assert path.read_bytes() == content
    assert validate_feed_depth_settings({"uniform_depth": None})["uniform_depth"] is None
    assert validate_feed_depth_settings({"error_tolerance_mm": 0})["error_tolerance_mm"] == 0.0


@pytest.mark.parametrize(("tolerance", "alarms"), [
    (0.05, [True, False, False, False, False, False, True]),
    (0, [True, True, True, True, False, True, True]),
    (None, [False] * 7),
])
def test_depth_range_around_theory_is_symmetric(tolerance, alarms):
    depths = [1.449, 1.45, 1.49, 1.497, 1.5, 1.55, 1.551]
    data = {
        "rows": [{"batch_id": "B01", "hole_id": str(index), "actual_depth_mm": depth,
                  "theoretical_depth_mm": 1.5} for index, depth in enumerate(depths)],
        "history": [{"batch_id": "B01"}],
    }
    result = evaluate_feed_depth(data, {"error_tolerance_mm": tolerance})
    assert [row["alarm"] for row in result["rows"]] == alarms
    assert result["history"][0]["alarm_count"] == sum(alarms)
    assert result["history"][0]["alarm"] is (tolerance == 0)


@pytest.mark.parametrize(("lower", "upper", "tolerance", "needs_input"), [
    (-0.05, 0.05, 0.05, False),
    (0, 0, 0.0, False),
    (None, None, None, False),
    (-0.02, 0.03, None, True),
    (1.45, 1.55, None, True),
])
def test_legacy_settings_load_without_rewriting_file(tmp_path, lower, upper, tolerance, needs_input):
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps({"depth_mode": "uniform", "uniform_depth": 1.5,
                                "error_lower_mm": lower, "error_upper_mm": upper}), encoding="utf-8")
    original = path.read_bytes()
    settings = load_feed_depth_settings(path)
    assert settings["error_tolerance_mm"] == tolerance
    assert settings.get("legacy_asymmetric_limits", False) is needs_input
    assert settings["depth_mode"] == "uniform" and settings["uniform_depth"] == 1.5
    assert path.read_bytes() == original
    saved = save_feed_depth_settings(path, {**settings, "error_tolerance_mm": 0.05})
    assert saved == {"depth_mode": "uniform", "uniform_depth": 1.5, "error_tolerance_mm": 0.05}
    assert json.loads(path.read_text(encoding="utf-8")) == saved


@pytest.mark.parametrize("suffix", [".csv", ".xlsx"])
def test_import_preserves_identifiers_missing_measurements_and_groups(tmp_path, suffix):
    path = tmp_path / ("batch" + suffix)
    records = [
        ["hole_id", "actual_depth_mm", "theoretical_depth_mm", "row_id"],
        ["001", 1.4, 1.5, "R1"], ["002", 1.6, 1.5, "R1"],
        ["003", None, 1.5, "R1"], ["001", 2.1, 2.0, "R2"],
        ["004", None, 2.0, "R3"],
    ]
    if suffix == ".csv":
        import csv

        with path.open("w", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream).writerows(records)
    else:
        from openpyxl import Workbook

        workbook = Workbook()
        for row in records:
            workbook.active.append(row)
        workbook.save(path)
        workbook.close()
    original = path.read_bytes()
    data = load_feed_depth_data(path, DEFAULT_SETTINGS)
    result = evaluate_feed_depth(data, DEFAULT_SETTINGS)
    assert [row["hole_id"] for row in data["rows"]] == ["001", "002", "003", "001", "004"]
    assert all(not row["is_simulated"] for row in result["rows"])
    assert result["rows"][2]["actual_depth_mm"] is None
    assert result["rows"][2]["bias_mm"] is None
    first, second, empty = result["history"]
    assert first["count"] == 2 and first["mean_depth_mm"] == pytest.approx(1.5)
    assert first["variance_mm2"] == pytest.approx(0.01)
    assert second["count"] == 1 and second["theoretical_depth_mm"] == 2.0
    assert empty["count"] == 0 and empty["mean_depth_mm"] is None
    assert all(row["measured_at"] == "" for row in result["history"])
    assert path.read_bytes() == original


def test_import_without_theory_uses_uniform_setting_but_cannot_invent_per_hole_values(tmp_path):
    path = tmp_path / "uniform.csv"
    path.write_text("hole_id,actual_depth_mm\n001,1.6\n", encoding="utf-8")
    settings = {"depth_mode": "uniform", "uniform_depth": 1.5}
    data = load_feed_depth_data(path, settings)
    assert evaluate_feed_depth(data, settings)["rows"][0]["bias_mm"] == pytest.approx(0.1)
    with pytest.raises(ValueError, match="逐孔理论窝深"):
        evaluate_feed_depth(data, DEFAULT_SETTINGS)
    path.write_text("hole_id,actual_depth_mm,theoretical_depth_mm\n001,1.6,not-used\n", encoding="utf-8")
    data = load_feed_depth_data(path, settings)
    assert evaluate_feed_depth(data, settings)["rows"][0]["bias_mm"] == pytest.approx(0.1)


@pytest.mark.parametrize(("body", "message"), [
    ("hole_id,actual_depth_mm\n001,1.5\n", "缺少字段"),
    ("hole_id,actual_depth_mm,theoretical_depth_mm\n001,nan,1.5\n", "有限数值"),
    ("hole_id,actual_depth_mm,theoretical_depth_mm\n001,,1.5\n", "没有有效实测"),
    ("hole_id,actual_depth_mm,theoretical_depth_mm\n001,1.5,1.5\n001,1.6,1.5\n", "孔号重复"),
    ("hole_id,actual_depth_mm,theoretical_depth_mm,batch_id\n001,1.5,1.5,B1\n002,1.6,1.5,B2\n", "batch_id"),
    ("hole_id,actual_depth_mm,theoretical_depth_mm,measured_at\n001,1.5,1.5,unknown\n", "ISO"),
])
def test_invalid_import_explains_how_to_fix_input(tmp_path, body, message):
    path = tmp_path / "invalid.csv"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_feed_depth_data(path, DEFAULT_SETTINGS)


def test_reimported_simulation_retains_source_and_matches_exported_statistics(tmp_path):
    from core.services.feed_depth_simulation import generate_feed_depth_simulation

    original = generate_feed_depth_simulation(tmp_path / "simulation", days=1)
    path = next((tmp_path / "simulation" / "batches").glob("*.csv"))
    imported = load_feed_depth_data(path, DEFAULT_SETTINGS)
    result = evaluate_feed_depth(imported, DEFAULT_SETTINGS)
    assert all(row["is_simulated"] for row in result["rows"])
    assert result["history"][0]["variance_mm2"] == original["history"][0]["variance_mm2"]
