"""核对主轴单位转换、冻结健康基线和随机初始化训练闭环。"""

import json

import h5py
import numpy as np
import pytest
import torch

from core.algorithms.spindle_monitoring import (
    acceleration_to_velocity,
    evaluate_run,
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
        summary = train_model([make_run("train")], [make_run("calibration", 0.07)], path, config)
    yield path, summary
    torch.set_num_threads(threads)


def test_training_from_scratch_saves_training_only_statistics(trained_model):
    path, summary = trained_model
    checkpoint = torch.load(path, weights_only=True)
    expected_scale = make_run()["velocity"].std(axis=(0, 2))
    np.testing.assert_allclose(checkpoint["scale"].numpy().ravel(), expected_scale, rtol=1e-5)
    assert summary["initialization"] == "random"
    assert summary["train_run_ids"] == ["train"]
    assert summary["validation_run_ids"] == ["calibration"]
    assert summary["healthy_reference_mse"] > 0
    expected_rms = (np.sqrt(.05 ** 2 + .01 ** 2) + np.sqrt(.07 ** 2 + .01 ** 2)) / 2
    assert summary["healthy_reference_rms_mm_s"] == pytest.approx(expected_rms, rel=1e-6)
    assert checkpoint["healthy_reference_rms_mm_s"] == summary["healthy_reference_rms_mm_s"]
    assert np.median(summary["reference_errors"]) == pytest.approx(1)
    json.dumps(summary, allow_nan=False)


def test_inference_uses_frozen_calibration_not_unknown_run_median(trained_model):
    path, summary = trained_model
    result = evaluate_run(make_run("unknown", 0.3), path)
    errors = np.asarray(result["window_channel_mse"])[:, :2].mean(axis=1)
    expected = errors / summary["healthy_reference_mse"]
    np.testing.assert_allclose(result["window_scores"], expected, rtol=1e-6)
    assert result["score"] == pytest.approx(np.quantile(expected, 0.95), rel=1e-6)
    assert not np.isclose(np.median(result["window_scores"]), 1)
    assert np.asarray(result["reconstruction"]["reconstructed"]).shape == (3, 2048)
    json.dumps(result, allow_nan=False)


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
    assert original["healthy_reference_mse"] != changed["healthy_reference_mse"]
