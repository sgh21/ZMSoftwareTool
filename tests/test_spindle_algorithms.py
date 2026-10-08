"""核对主轴单位转换、采集级正常相容度及冻结网络校准闭环。"""

import json

import h5py
import numpy as np
import pytest
import torch

from core.algorithms.spindle_monitoring import (
    acceleration_to_velocity,
    calibrate_model,
    evaluate_run,
    fit_normal_reference,
    normal_compatibility,
    order_band_energy,
    read_run,
    train_model,
)
from core.algorithms.spindle_network import TCNAutoencoder


def make_run(name="normal", amplitude=0.05):
    time = np.arange(20480) / 2048
    velocity = np.stack([
        amplitude * np.sin(2 * np.pi * 120 * time + phase)
        + 0.01 * np.sin(2 * np.pi * 240 * time)
        for phase in [0.0, 0.7, 1.4]
    ]).astype(np.float32)[None]
    return {
        "run_id": name, "run_name": name, "run_date": "20260929",
        "target_speed_rpm": 7000, "sample_rate_hz": 2048,
        "velocity": velocity, "start_time_s": np.asarray([610.0]),
        "temperature": np.full((1, 2, 100), 25.0, dtype=np.float32),
        "temperature_valid": np.ones((1, 2, 100), dtype=bool),
        "sample_valid": np.ones(1, dtype=bool), "source_segment_index": np.asarray([61]),
        "telemetry": {"time_s": np.asarray([610.0, 611.0]),
                      "actual_speed_rpm": np.asarray([7000.0, 7000.0]),
                      "current_a": np.asarray([0.4, 0.6])},
    }


def test_velocity_conversion_matches_analytic_integral_and_rejects_band_outside():
    time = np.arange(256000) / 25600
    frequency = 120.0
    acceleration = np.stack([
        0.01 * np.cos(2 * np.pi * frequency * time),
        0.02 * np.cos(2 * np.pi * frequency * time),
        np.cos(2 * np.pi * 1500 * time),
    ])
    velocity = acceleration_to_velocity(acceleration)
    expected = 0.01 * 9806.65 / (2 * np.pi * frequency) * np.sin(2 * np.pi * frequency * np.arange(20480) / 2048)
    np.testing.assert_allclose(velocity[0], expected, atol=1e-8)
    np.testing.assert_allclose(velocity[1], 2 * expected, atol=2e-8)
    assert np.max(np.abs(velocity[2])) < 1e-9
    assert velocity.dtype == np.float32


def test_read_run_preserves_channel_mapping_missing_temperature_and_gaps(tmp_path):
    run = make_run()
    path = tmp_path / "run.h5"
    with h5py.File(path, "w") as h5:
        h5.attrs.update({
            "velocity_sample_rate_hz": 2048, "velocity_unit": "mm/s",
            "velocity_frequency_band_hz": [10, 900], "velocity_channels": ["ACC3", "ACC1", "ACC2"],
            "run_name": "normal", "run_date": "20260929", "target_speed_rpm": 7000,
        })
        group = h5.create_group("windows_10s")
        group["velocity"] = np.repeat(run["velocity"][:, [2, 0, 1]], 2, axis=0)
        group["start_time_s"] = [610, 670]
        group["temperature"] = np.repeat(run["temperature"], 2, axis=0)
        validity = np.ones((2, 2, 100), dtype=bool)
        validity[0, 0, 0] = False
        group["temperature_valid"] = validity
        group["sample_valid"] = [False, True]
        group["source_segment_index"] = [61, 67]
    telemetry = tmp_path / "telemetry.csv"
    telemetry.write_text("time_s,actual_speed_rpm,current_a,speed_ok,current_ok\n610,9999,999,false,false\n611,7000,0.5,true,true\n", encoding="utf-8")
    loaded = read_run(path, telemetry)
    assert np.isnan(loaded["telemetry"]["actual_speed_rpm"][0])
    assert np.isnan(loaded["telemetry"]["current_a"][0])
    np.testing.assert_array_equal(loaded["velocity"][0], run["velocity"][0])
    assert loaded["start_time_s"].tolist() == [610, 670]
    assert np.isnan(loaded["temperature"][0, 0, 0])
    assert not loaded["sample_valid"][0]
    result = evaluate_run(loaded)
    assert result["window_count"] == 20  # 温度缺测不抹去有效振动。
    assert result["window_time_s"][10] == 670
    json.dumps(result, allow_nan=False)


def test_dsp_without_model_has_physical_spectrum_and_no_score():
    result = evaluate_run(make_run())
    assert result["score"] is None
    spectrum = result["spectrum"]
    peak = np.argmax(spectrum["amplitude_mm_s"][0])
    assert spectrum["frequency_hz"][peak] == pytest.approx(120)
    assert spectrum["amplitude_mm_s"][0][peak] == pytest.approx(0.05, rel=1e-3)
    assert result["rms_mm_s"][0] == pytest.approx(np.sqrt((0.05 ** 2 + 0.01 ** 2) / 2), rel=1e-5)
    assert sum(result["band_energy"]["values"][0]) == pytest.approx(result["rms_mm_s"][0] ** 2)
    assert result["current_a"] == pytest.approx(0.5)
    json.dumps(result, allow_nan=False)


def test_order_energy_uses_actual_speed_fixed_half_width_and_disjoint_band_power():
    run = make_run()
    run["telemetry"]["actual_speed_rpm"] = np.asarray([np.nan, 6000.0])
    time = np.arange(20480) / 2048
    # 90/110、190/210 Hz 均含在边界；218 Hz 在 2× ±0.1× 之外。
    tones = [(50, .04), (90, .02), (100, .03), (110, .02),
             (190, .01), (200, .02), (210, .01), (218, .05),
             (300, .06), (700, .01), (950, .2)]
    signal = sum(amplitude * np.sin(2 * np.pi * frequency * time) for frequency, amplitude in tones)
    run["velocity"] = np.stack([signal, signal * 2, signal * .5]).astype(np.float32)[None]
    energy = order_band_energy(run)
    assert energy["rotation_hz"] == 100
    assert energy["speed_source"] == "actual"
    assert energy["labels"] == ["异步", *[f"{order}×" for order in range(1, 10)]]
    np.testing.assert_allclose(energy["bands_hz"][:3], [[90, 110], [190, 210], [290, 310]])
    expected = np.asarray([(.04 ** 2 + .05 ** 2) / 2, (.02 ** 2 + .03 ** 2 + .02 ** 2) / 2,
                           (.01 ** 2 + .02 ** 2 + .01 ** 2) / 2, .06 ** 2 / 2,
                           0, 0, 0, .01 ** 2 / 2, 0, 0])
    np.testing.assert_allclose(energy["values"], [expected, expected * 4, expected * .25],
                               rtol=1e-6, atol=1e-12)
    total = sum(amplitude ** 2 / 2 for frequency, amplitude in tones if frequency <= 900)
    assert sum(energy["values"][0]) == pytest.approx(total, rel=1e-6)
    json.dumps(energy, allow_nan=False)


def test_order_energy_falls_back_to_target_speed_when_valid_telemetry_is_missing():
    run = make_run()
    run["telemetry"]["actual_speed_rpm"] = np.asarray([np.nan, 0.0])
    energy = evaluate_run(run)["band_energy"]
    assert energy["rotation_hz"] == pytest.approx(7000 / 60)
    assert energy["speed_source"] == "target"
    assert energy["labels"] == ["异步", *[f"{order}×" for order in range(1, 8)]]
    assert energy["values"][0][1] == pytest.approx(.05 ** 2 / 2, rel=1e-5)
    assert energy["values"][0][2] == pytest.approx(.01 ** 2 / 2, rel=1e-5)


def test_original_tcn_structure_is_preserved():
    model = TCNAutoencoder()
    assert sum(parameter.numel() for parameter in model.parameters()) == 146755


@pytest.fixture
def trained_model(tmp_path, monkeypatch):
    path = tmp_path / "fresh.pt"
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    config = {"epochs": 1, "batch_size": 10, "device": "cpu", "seed": 17}
    # 训练不得读取旧权重；加载自己保存的模型只发生在后续独立推理中。
    with monkeypatch.context() as patch:
        patch.setattr(torch, "load", lambda *args, **kwargs: pytest.fail("训练读取了 checkpoint"))
        summary = train_model([make_run("train")], [
            make_run(f"calibration_{index}", amplitude)
            for index, amplitude in enumerate([.065, .068, .071, .074, .077])
        ], path, config)
    yield path, summary
    torch.set_num_threads(threads)


def test_training_from_scratch_saves_training_only_statistics(trained_model):
    path, summary = trained_model
    checkpoint = torch.load(path, weights_only=True)
    expected_scale = make_run()["velocity"].std(axis=(0, 2))
    np.testing.assert_allclose(checkpoint["scale"].numpy().ravel(), expected_scale, rtol=1e-5)
    assert summary["initialization"] == "random"
    assert summary["train_run_ids"] == ["train"]
    assert summary["validation_run_ids"] == [f"calibration_{index}" for index in range(5)]
    assert checkpoint["network_id"] == summary["network_id"]
    group = summary["normal_reference"]["groups"][0]
    assert group["one_second_window_count"] == 10
    reference = group["vibration"]
    expected_rms = np.sqrt(np.array([.065, .068, .071, .074, .077]) ** 2 + .01 ** 2)
    np.testing.assert_allclose(reference["values"], expected_rms, rtol=1e-6)
    assert reference["mu"] == pytest.approx(np.log(expected_rms).mean(), rel=1e-6)
    assert reference["s"] == pytest.approx(np.log(expected_rms).std(ddof=1), rel=1e-6)
    assert reference["M"] == group["network"]["M"] == 5
    assert reference["run_ids"] == summary["validation_run_ids"]
    assert reference["calibration_level"] == "preliminary"
    assert "初步校准" in reference["message"]
    assert "healthy_reference_mse" not in summary
    json.dumps(summary, allow_nan=False)


def test_inference_uses_acquisition_p95_and_frozen_normal_reference(trained_model):
    path, summary = trained_model
    result = evaluate_run(make_run("unknown", 0.3), path)
    errors = np.asarray(result["window_channel_mse"])[:, :2].mean(axis=1)
    np.testing.assert_allclose(result["window_errors"], errors, rtol=1e-6)
    feature = np.quantile(errors, 0.95)
    assert result["reconstruction_error_p95"] == pytest.approx(feature, rel=1e-6)
    reference = summary["normal_reference"]["groups"][0]["network"]
    expected = normal_compatibility(feature, reference)
    assert result["score"] == pytest.approx(expected["score"])
    assert 0 <= result["score"] < 1
    assert 0 <= result["analysis_score"] < 1
    assert result["reference_features"]["network"] == reference["values"]
    assert len(result["reference_features"]["network"]) == 5  # 每次采集一个 E，不是 50 个窗口。
    assert "unknown" not in result["compatibility"]["network"]["run_ids"]
    assert np.asarray(result["reconstruction"]["reconstructed"]).shape == (3, 2048)
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize(("feature", "score_key", "other_feature", "other_score_key"), [
    ("vibration", "analysis_score", "network", "score"),
    ("network", "score", "vibration", "analysis_score"),
])
def test_metric_alpha_only_changes_its_own_compatibility_not_features_p_values_or_reference(
        trained_model, feature, score_key, other_feature, other_score_key):
    path, summary = trained_model
    run = make_run("unknown", .3)
    original = evaluate_run(run, path)
    changed = evaluate_run(run, path, {f"{feature}_alpha": .2})
    assert 0 < original[score_key] < 1
    assert changed[score_key] == pytest.approx(original[score_key] / 4)
    assert changed[other_score_key] == original[other_score_key]
    assert changed["compatibility"][feature]["alpha"] == .2
    assert changed["compatibility"][other_feature]["alpha"] == .05
    assert changed["alpha"] == changed["compatibility"]["network"]["alpha"]
    for key in ("xy_rms_mm_s", "reconstruction_error_p95", "vibration_p_value", "network_p_value",
                "normalized_mse", "window_errors", "reference_features"):
        assert changed[key] == original[key]
    assert changed["compatibility"][other_feature] == original["compatibility"][other_feature]
    assert torch.load(path, weights_only=True)["normal_reference"] == summary["normal_reference"]


@pytest.mark.parametrize(("config", "vibration_alpha", "network_alpha"), [
    ({}, .05, .05),
    ({"alpha": .1}, .1, .1),
    ({"alpha": .1, "vibration_alpha": .2}, .2, .1),
    ({"alpha": .1, "network_alpha": .3}, .1, .3),
    ({"vibration_alpha": .2, "network_alpha": .3}, .2, .3),
])
def test_metric_alpha_defaults_and_shared_configuration_remain_compatible(config, vibration_alpha, network_alpha):
    result = evaluate_run(make_run(), config=config)
    assert result["alpha"] == network_alpha
    assert result["compatibility"]["vibration"]["alpha"] == vibration_alpha
    assert result["compatibility"]["network"]["alpha"] == network_alpha
    assert result["score"] is result["analysis_score"] is None


def test_training_rejects_same_run_in_training_and_calibration(tmp_path):
    run = make_run()
    with pytest.raises(ValueError, match="隔离"):
        train_model([run], [run], tmp_path / "invalid.pt", {})
    assert not (tmp_path / "invalid.pt").exists()


def test_other_speed_has_dsp_but_is_not_scored():
    run = make_run()
    run["target_speed_rpm"] = 7400
    result = evaluate_run(run, "not_loaded.pt")
    assert result["score"] is None
    assert result["spectrum"]["frequency_hz"]


def test_calibration_signal_cannot_change_learned_weights(trained_model, tmp_path):
    path, _ = trained_model
    changed_path = tmp_path / "different_calibration.pt"
    train_model([make_run("train")], [make_run("other_calibration", 0.8)], changed_path,
                {"epochs": 1, "batch_size": 10, "device": "cpu", "seed": 17})
    original = torch.load(path, weights_only=True)
    changed = torch.load(changed_path, weights_only=True)
    for name, value in original["model_state"].items():
        torch.testing.assert_close(value, changed["model_state"][name], rtol=0, atol=0)
    assert (original["normal_reference"]["groups"][0]["network"]["values"]
            != changed["normal_reference"]["groups"][0]["network"]["values"])


def test_predictive_t_compatibility_center_tails_and_acceptance_mapping():
    from scipy.stats import t

    reference = fit_normal_reference(np.exp(np.linspace(-.4, .4, 21)))
    assert reference["status"] == "valid"
    scale = reference["s"] * np.sqrt(1 + 1 / reference["M"])
    center = np.exp(reference["mu"])
    assert normal_compatibility(center, reference, two_sided=True)["score"] == 1
    assert normal_compatibility(center, reference)["score"] == 1
    assert normal_compatibility(center * np.exp(-10 * scale), reference)["score"] == 1
    for p_value, expected in [(.30, 1), (.05, 1), (.02, .4), (.001, .02)]:
        for two_sided in (False, True):
            u = t.isf(p_value / (2 if two_sided else 1), df=reference["M"] - 1)
            x = np.exp(reference["mu"] + scale * u)
            result = normal_compatibility(x, reference, two_sided=two_sided)
            assert result["u"] == pytest.approx(u)
            assert result["p_value"] == pytest.approx(p_value)
            assert result["score"] == pytest.approx(expected)
            if two_sided:
                low = np.exp(reference["mu"] - scale * u)
                assert normal_compatibility(low, reference, two_sided=True)["score"] == pytest.approx(expected)


@pytest.mark.parametrize("values", [[1, 2], [1, 0, 2], [1, -1, 2], [1, np.nan, 2],
                                    [1, np.inf, 2], [2] * 7])
def test_invalid_normal_reference_never_produces_full_score(values):
    reference = fit_normal_reference(values)
    assert reference["status"] == "invalid"
    result = normal_compatibility(1, reference)
    assert result["score"] is result["p_value"] is None
    assert result["message"]
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("value", [None, 0, -1, np.nan, np.inf])
def test_invalid_current_feature_is_explicitly_invalid(value):
    result = normal_compatibility(value, fit_normal_reference([1, 2, 3]))
    assert result["status"] == "invalid"
    assert result["score"] is result["p_value"] is None
    json.dumps(result, allow_nan=False)


def test_normal_reference_rejects_duplicate_runs_and_clear_log_non_gaussian_data():
    assert fit_normal_reference([1, 2, 3], ["same", "same", "third"])["status"] == "invalid"
    reference = fit_normal_reference(np.exp([0] * 20 + [6] * 10))
    assert reference["status"] == "invalid"
    assert reference["normality_p_value"] < .01
    assert "高斯" in reference["message"]


def test_compatibility_calls_survival_function_for_tiny_tail(monkeypatch):
    from scipy.stats import t

    reference = fit_normal_reference(np.exp(np.linspace(-.4, .4, 30)))
    monkeypatch.setattr(t, "cdf", lambda *args, **kwargs: pytest.fail("必须使用 sf"))
    result = normal_compatibility(1e50, reference)
    assert 0 < result["p_value"] < 1e-16
    assert 0 < result["score"] < 1e-14


def test_evaluation_invalidates_unmatched_context_length_and_version(trained_model):
    from copy import deepcopy

    path, summary = trained_model
    context_changed = make_run("unknown", .071)
    context_changed["calibration_context"] = {"condition": {"seal_pressure_mpa": .2}}
    length_changed = make_run("longer", .071)
    for key in ("velocity", "temperature", "temperature_valid", "sample_valid"):
        length_changed[key] = np.repeat(length_changed[key], 2, axis=0)
    length_changed["start_time_s"] = np.array([610., 620.])
    for run in (context_changed, length_changed):
        result = evaluate_run(run, path)
        assert result["score"] is result["analysis_score"] is None
        assert "匹配" in result["compatibility"]["network"]["message"]
        assert result["reconstruction_error_p95"] > 0
    for field, mismatch in [("network_id", "other-network"), ("preprocessing", {"id": "other-v2"})]:
        reference = deepcopy(summary["normal_reference"])
        reference[field] = mismatch
        result = evaluate_run(make_run("unknown", .071), path, {"normal_reference": reference})
        assert result["score"] is result["analysis_score"] is None
        assert "版本不匹配" in result["compatibility"]["network"]["message"]


def test_calibrate_existing_network_without_changing_checkpoint_or_using_training_runs(trained_model):
    path, _ = trained_model
    original_bytes = path.read_bytes()
    runs = [make_run(f"new_calibration_{index}", amplitude)
            for index, amplitude in enumerate([.064, .068, .072, .076, .080])]
    reference = calibrate_model(runs, path)
    assert path.read_bytes() == original_bytes
    assert reference["groups"][0]["network"]["M"] == 5
    result = evaluate_run(make_run("center", .072), path, {"normal_reference": reference})
    assert result["score"] == result["analysis_score"] == 1
    with pytest.raises(ValueError, match="训练采集重叠"):
        calibrate_model([make_run("train")], path)


def test_legacy_checkpoint_requires_new_reference_and_keeps_raw_features(trained_model, tmp_path):
    path, _ = trained_model
    checkpoint = torch.load(path, weights_only=True)
    checkpoint.pop("normal_reference")
    checkpoint.pop("network_id")
    legacy_path = tmp_path / "legacy.pt"
    torch.save(checkpoint, legacy_path)
    run = make_run("unknown", .071)
    result = evaluate_run(run, legacy_path)
    assert result["score"] is result["analysis_score"] is None
    assert result["xy_rms_mm_s"] > 0
    assert result["reconstruction_error_p95"] > 0
    reference = calibrate_model([make_run(f"c{i}", .065 + i * .003) for i in range(5)], legacy_path)
    assert reference["network_id"].startswith("legacy:")
    result = evaluate_run(run, legacy_path, {"normal_reference": reference})
    assert result["score"] == result["analysis_score"] == 1
