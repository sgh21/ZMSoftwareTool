"""主轴数据读取、速度频谱与从头训练的 TCN；不依赖界面。

速度转换与 TCN 源自 Self-SupervisedReconstruction。正常相容度仅使用
独立正常校准采集的特征参考，待测数据不参与参考拟合或模型训练。
"""

import csv
import json
from pathlib import Path
from uuid import uuid4

import numpy as np


SAMPLE_RATE_HZ = 2048
CHANNELS = ["ACC1", "ACC2", "ACC3"]
TEMPERATURE_CHANNELS = ["NTC1", "RTD1"]
MODEL_ARCHITECTURE = "tcn_1s"
SCORE_KIND = "normal_compatibility_v1"
PREPROCESSING = {
    "id": "velocity_2048_10_900_v1", "sample_rate_hz": SAMPLE_RATE_HZ,
    "frequency_band_hz": [10, 900], "unit": "mm/s", "channels": CHANNELS,
}
PRELIMINARY_REFERENCE_COUNT = 20
TRAINING_DEFAULTS = {
    "epochs": 150, "batch_size": 128, "learning_rate": 0.003,
    "weight_decay": 1e-5, "seed": 20260929, "device": "auto",
}


def acceleration_to_velocity(acceleration_g) -> np.ndarray:
    """完整 10 秒、25.6 kHz 的加速度 g 转为 2048 Hz、10–900 Hz 速度 mm/s。"""
    acceleration = np.asarray(acceleration_g, dtype=np.float64)
    if acceleration.shape != (3, 256000) or not np.isfinite(acceleration).all():
        raise ValueError("加速度必须是有效的三通道完整 10 秒数据 [3, 256000]")
    acceleration = acceleration * 9806.65
    acceleration -= acceleration.mean(axis=1, keepdims=True)
    frequency = np.fft.rfftfreq(256000, 1 / 25600)
    spectrum = np.fft.rfft(acceleration, axis=1)
    # 原处理链的积分通带为 10–900 Hz，带外过渡在最终存储带外，不进入输出。
    selected = (frequency >= 10) & (frequency <= 900)
    output = np.zeros((3, 10241), dtype=np.complex128)
    output_frequency = np.fft.rfftfreq(20480, 1 / SAMPLE_RATE_HZ)
    output_band = (output_frequency >= 10) & (output_frequency <= 900)
    output[:, output_band] = (
        spectrum[:, selected] / (2j * np.pi * frequency[selected]) * (20480 / 256000)
    )
    velocity = np.fft.irfft(output, n=20480, axis=1)
    velocity -= velocity.mean(axis=1, keepdims=True)
    return velocity.astype(np.float32)


def _text(value):
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _attribute(value):
    if isinstance(value, np.ndarray):
        return [_attribute(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    return value.decode("utf-8") if isinstance(value, bytes) else value


def read_run(h5_path, telemetry_path=None) -> dict:
    """读取一份 run H5；温度缺测保留 NaN，主轴遥测单独保留真实时间。"""
    import h5py

    with h5py.File(h5_path, "r") as h5:
        rate = float(h5.attrs["velocity_sample_rate_hz"])
        unit = _text(h5.attrs["velocity_unit"])
        band = np.asarray(h5.attrs["velocity_frequency_band_hz"])
        if rate != SAMPLE_RATE_HZ or unit != "mm/s" or not np.array_equal(band, [10, 900]):
            raise ValueError("模型数据需要 2048 Hz、10–900 Hz、mm/s 的振动速度")
        channel_names = [_text(value) for value in h5.attrs["velocity_channels"]]
        if set(channel_names) != set(CHANNELS):
            raise ValueError("振动数据必须包含 ACC1、ACC2、ACC3 三个通道")
        group = h5["windows_10s"]
        order = [channel_names.index(name) for name in CHANNELS]
        velocity = np.asarray(group["velocity"], dtype=np.float32)[:, order]
        if velocity.ndim != 3 or velocity.shape[1:] != (3, 20480) or not len(velocity):
            raise ValueError("H5 没有完整的三通道 10 秒速度窗口")
        if not np.isfinite(velocity).all():
            raise ValueError("振动数据包含缺失或无效数值")
        temperature = np.asarray(group["temperature"], dtype=np.float32)
        temperature_valid = np.asarray(group["temperature_valid"], dtype=bool)
        if temperature.shape != (len(velocity), 2, 100) or temperature_valid.shape != temperature.shape:
            raise ValueError("温度窗口与振动窗口不匹配")
        temperature[~temperature_valid] = np.nan
        run = {
            "metadata": {key: _attribute(value) for key, value in h5.attrs.items()},
            "schema_version": _text(h5.attrs.get("schema_version", "")),
            "run_id": _text(h5.attrs.get("run_id", h5.attrs["run_name"])),
            "run_name": _text(h5.attrs["run_name"]),
            "run_date": _text(h5.attrs["run_date"]),
            "target_speed_rpm": float(h5.attrs["target_speed_rpm"]),
            "sample_rate_hz": rate, "velocity": velocity,
            "start_time_s": np.asarray(group["start_time_s"], dtype=float),
            "temperature": temperature, "temperature_valid": temperature_valid,
            "sample_valid": np.asarray(group["sample_valid"], dtype=bool),
            "source_segment_index": np.asarray(group["source_segment_index"], dtype=int),
        }
    if run["start_time_s"].shape != (len(velocity),) or not np.isfinite(run["start_time_s"]).all():
        raise ValueError("振动窗口时间无效")
    run["telemetry"] = {key: np.asarray([], dtype=float) for key in (
        "time_s", "actual_speed_rpm", "current_a",
    )}
    if telemetry_path is not None:
        with Path(telemetry_path).open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        run["telemetry"] = {
            key: np.asarray([float(row[key]) if row[key] else np.nan for row in rows])
            for key in run["telemetry"]
        }
        for key, flag in (("actual_speed_rpm", "speed_ok"), ("current_a", "current_ok")):
            valid = np.array([str(row.get(flag, "true")).lower() in ("true", "1") for row in rows], dtype=bool)
            run["telemetry"][key][~valid] = np.nan
    return run


def _windows(run) -> np.ndarray:
    velocity = np.asarray(run["velocity"], dtype=np.float32)
    return velocity.reshape(-1, 3, 10, SAMPLE_RATE_HZ).transpose(0, 2, 1, 3).reshape(-1, 3, SAMPLE_RATE_HZ)


def velocity_rms_mm_s(velocity):
    return np.sqrt(np.mean(np.asarray(velocity, dtype=float) ** 2, axis=(0, 2)))


def fit_normal_reference(values, run_ids=None) -> dict:
    """每次独立采集一个正特征；检查对数分布，不以窗口扩大校准样本数。"""
    from scipy.stats import shapiro

    values = np.asarray(values, dtype=float)
    count = len(values)
    reference = {
        "mu": None, "s": None, "M": count, "values": _json_values(values),
        "run_ids": list(run_ids or []), "status": "invalid", "message": "",
        "calibration_level": "preliminary" if count < PRELIMINARY_REFERENCE_COUNT else "calibrated",
        "normality_p_value": None,
    }
    if count < 3:
        reference["message"] = "正常参考不足 3 次独立采集"
        return reference
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values <= 0):
        reference["message"] = "正常参考特征必须为有限正数，不能含 NaN/Inf"
        return reference
    if run_ids is not None and (len(run_ids) != count or len(set(run_ids)) != count):
        reference["message"] = "正常参考必须来自不同的完整采集，采集编号不能重复"
        return reference
    logarithms = np.log(values)
    mu = float(logarithms.mean())
    deviation = float(logarithms.std(ddof=1))
    reference.update(mu=mu, s=deviation)
    if np.ptp(logarithms) == 0 or deviation == 0 or not np.isfinite(deviation):
        reference["message"] = "正常参考对数样本标准差为 0 或无效，不能计算相容度"
        return reference
    # Shapiro 是适用性诊断，不能证明高斯分布或采集间独立性。
    normality_p = float(shapiro(logarithms).pvalue)
    reference["normality_p_value"] = normality_p
    if normality_p < 0.01:
        reference["message"] = "对数参考的高斯适用性检查未通过（p < 0.01），请检查工况或分组"
        return reference
    reference["status"] = "valid"
    prefix = "初步校准；" if count < PRELIMINARY_REFERENCE_COUNT else ""
    reference["message"] = prefix + "对数高斯检查未见明显偏离；采集独立性及工况仍需人工确认"
    return reference


def normal_compatibility(value, reference, alpha=0.05, *, two_sided=False) -> dict:
    """有限正常参考下的 t 预测检验；相容度不是设备正常的后验概率。"""
    from scipy.stats import t

    result = {**reference, "score": None, "p_value": None, "u": None,
              "alpha": alpha, "feature": value, "status": "invalid"}
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        result["alpha"] = alpha if np.isfinite(alpha) else None
        result["message"] = "alpha 必须在 0 与 1 之间"
        return result
    if value is None or not np.isfinite(value) or value <= 0:
        result["feature"] = float(value) if value is not None and np.isfinite(value) else None
        result["message"] = "当前采集特征必须为有限正数，不能含 NaN/Inf"
        return result
    if reference.get("status") != "valid":
        return result
    count, mu, deviation = reference.get("M", 0), reference.get("mu"), reference.get("s")
    if count < 3 or mu is None or deviation is None or not np.isfinite([mu, deviation]).all() or deviation <= 0:
        result["message"] = "正常参考参数无效，请重新校准"
        return result
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        u = (float(np.log(value)) - mu) / (deviation * np.sqrt(1 + 1 / count))
    if not np.isfinite(u):
        result["message"] = "当前偏离程度超出数值范围，不能计算有效相容度"
        return result
    p_value = float(2 * t.sf(abs(u), df=count - 1) if two_sided else t.sf(u, df=count - 1))
    result.update(status="valid", u=float(u), p_value=p_value, score=min(1.0, p_value / alpha))
    return result


def _calibration_context(run, preprocessing=None):
    supplied = run.get("calibration_context", {})
    return {
        "speed_rpm": supplied.get("speed_rpm", run["target_speed_rpm"]),
        "operation": supplied.get("operation", "idle"),
        "condition": supplied.get("condition", {}),
        "preprocessing": supplied.get("preprocessing", preprocessing or PREPROCESSING),
    }


def _network_id(checkpoint, model_path):
    if checkpoint.get("network_id"):
        return checkpoint["network_id"]
    path = Path(model_path).resolve()
    stat = path.stat()
    return f"legacy:{path}:{stat.st_mtime_ns}:{stat.st_size}"


def _build_normal_reference(validation_runs, run_errors, network_id, preprocessing):
    groups = {}
    for run, errors in zip(validation_runs, run_errors):
        context = _calibration_context(run, preprocessing)
        count = len(_windows(run))
        key = json.dumps([context, count], sort_keys=True, ensure_ascii=False)
        group = groups.setdefault(key, {"context": context, "one_second_window_count": count,
                                        "run_ids": [], "vibration_values": [], "network_values": []})
        group["run_ids"].append(run["run_id"])
        group["vibration_values"].append(float(np.linalg.norm(velocity_rms_mm_s(run["velocity"])[:2])))
        group["network_values"].append(float(np.quantile(errors[:, :2].mean(axis=1), 0.95)))
    for group in groups.values():
        for feature in ("vibration", "network"):
            group[feature] = fit_normal_reference(group.pop(f"{feature}_values"), group["run_ids"])
        condition = group["context"]["condition"]
        if not condition or any(value is None for value in condition.values()):
            for feature in ("vibration", "network"):
                group[feature]["message"] += "；工况信息缺失，需人工确认匹配"
    return {"score_kind": SCORE_KIND, "network_id": network_id,
            "preprocessing": preprocessing, "groups": list(groups.values())}


def _matched_reference(normal_reference, run, network_id, preprocessing):
    if not normal_reference:
        return None, "未建立独立正常采集参考，请重新校准"
    if normal_reference.get("score_kind") != SCORE_KIND or normal_reference.get("network_id") != network_id:
        return None, "正常参考与当前网络版本不匹配，请重新校准"
    if normal_reference.get("preprocessing") != preprocessing:
        return None, "正常参考与当前预处理版本不匹配，请重新校准"
    context = _calibration_context(run, preprocessing)
    for group in normal_reference.get("groups", []):
        if group["context"] == context and group["one_second_window_count"] == len(_windows(run)):
            return group, ""
    return None, "没有工况、采集长度及预处理匹配的正常参考，请重新校准"


def _device(torch, name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def _infer(model, windows, center, scale, device, batch_size):
    import torch

    model.eval()
    errors = []
    preview = None
    with torch.inference_mode():
        for start in range(0, len(windows), batch_size):
            original = torch.from_numpy(windows[start:start + batch_size]).to(device)
            normalized = (original - center) / scale
            reconstructed = model(normalized)
            errors.append((reconstructed - normalized).square().mean(dim=2).cpu().numpy())
            if preview is None:
                preview = (reconstructed[0] * scale[0] + center[0]).cpu().numpy()
    return np.concatenate(errors), preview


def train_model(train_runs, validation_runs, output_path, config, progress=None) -> dict:
    """从随机初始化训练，冻结网络后仅用隔离校准采集建立正常参考。"""
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from core.algorithms.spindle_network import TCNAutoencoder

    settings = {**TRAINING_DEFAULTS, **config}
    train_ids = [run["run_id"] for run in train_runs]
    validation_ids = [run["run_id"] for run in validation_runs]
    if not train_ids or not validation_ids or set(train_ids) & set(validation_ids):
        raise ValueError("训练集和健康校准集必须非空且按完整 run 隔离")
    if len(set(train_ids)) != len(train_ids) or len(set(validation_ids)) != len(validation_ids):
        raise ValueError("训练和正常校准采集编号不能重复")
    if any(run["target_speed_rpm"] != 7000 for run in train_runs + validation_runs):
        raise ValueError("当前模型只使用 7000 rpm 数据训练和校准")
    if int(settings["epochs"]) < 1 or int(settings["batch_size"]) < 1:
        raise ValueError("训练轮数和批大小必须为正整数")
    torch.manual_seed(int(settings["seed"]))
    np.random.seed(int(settings["seed"]))
    device = _device(torch, settings["device"])
    train = np.concatenate([_windows(run) for run in train_runs])
    validation = np.concatenate([_windows(run) for run in validation_runs])
    center_np = train.mean(axis=(0, 2), dtype=np.float64).reshape(1, 3, 1)
    scale_np = train.std(axis=(0, 2), dtype=np.float64).reshape(1, 3, 1)
    if not np.isfinite(train).all() or np.any(scale_np <= 0):
        raise ValueError("训练数据无效，或某振动通道没有波动")
    center = torch.tensor(center_np, dtype=torch.float32, device=device)
    scale = torch.tensor(scale_np, dtype=torch.float32, device=device)
    # 每次新建随机网络；不读取任何旧 checkpoint，不隐式继续训练。
    model = TCNAutoencoder().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(settings["learning_rate"]),
        weight_decay=float(settings["weight_decay"]),
    )
    epochs = int(settings["epochs"])
    batch_size = int(settings["batch_size"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=float(settings["learning_rate"]) * 0.01,
    )
    loader = DataLoader(TensorDataset(torch.from_numpy(train)), batch_size=batch_size, shuffle=True)
    history = []
    if progress:
        progress(0, f"从随机初始化训练 TCN：{len(train_ids)} 个训练 run")
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        for (batch,) in loader:
            normalized = (batch.to(device) - center) / scale
            optimizer.zero_grad(set_to_none=True)
            loss = (model(normalized) - normalized).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_sum += float(loss.item()) * len(batch)
        train_loss = loss_sum / len(train)
        if not np.isfinite(train_loss):
            raise ValueError("训练出现非有限误差，未保存模型")
        history.append({"epoch": epoch, "train_mse": train_loss})
        scheduler.step()
        if progress:
            progress(int(90 * epoch / epochs), f"训练 {epoch}/{epochs}，训练误差 {train_loss:.6g}")
    # 固定轮数后冻结网络；校准集仅用于建立参考，不能用于选择权重。
    final_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    errors, _ = _infer(model, validation, center, scale, device, batch_size)
    run_errors = np.split(errors, np.cumsum([len(_windows(run)) for run in validation_runs])[:-1])
    network_id = str(uuid4())
    preprocessing = settings.get("preprocessing", PREPROCESSING)
    normal_reference = _build_normal_reference(validation_runs, run_errors, network_id, preprocessing)
    summary = {
        "architecture": MODEL_ARCHITECTURE, "initialization": "random", "config": settings,
        "trained_epochs": epochs, "selection": "fixed_epochs", "train_run_ids": train_ids,
        "validation_run_ids": validation_ids, "train_window_count": len(train),
        "validation_window_count": len(validation), "device": str(device),
        "network_id": network_id, "preprocessing": preprocessing, "score_kind": SCORE_KIND,
        "normal_reference": normal_reference, "history": history,
        "score_definition": "min(1, t predictive tail p / alpha); acquisition radial RMS two-sided, reconstruction P95 upper-sided",
    }
    checkpoint = {
        **summary, "model_state": final_state, "center": center.cpu(), "scale": scale.cpu(),
        "sample_rate_hz": SAMPLE_RATE_HZ, "band_hz": [10, 900], "channels": CHANNELS,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)
    if progress:
        progress(100, "模型与健康校准参考已保存")
    return summary


def calibrate_model(validation_runs, model_path, preprocessing=None) -> dict:
    """冻结现有权重，重新计算独立正常采集参考；不修改 checkpoint。"""
    import torch
    from core.algorithms.spindle_network import TCNAutoencoder

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    run_ids = [run["run_id"] for run in validation_runs]
    if len(set(run_ids)) != len(run_ids) or set(run_ids) & set(checkpoint["train_run_ids"]):
        raise ValueError("正常校准采集必须独立，不能重复或与网络训练采集重叠")
    settings = {**TRAINING_DEFAULTS, **checkpoint["config"]}
    device = _device(torch, settings["device"])
    model = TCNAutoencoder().to(device)
    model.load_state_dict(checkpoint["model_state"])
    center, scale = checkpoint["center"].to(device), checkpoint["scale"].to(device)
    run_errors = [
        _infer(model, _windows(run), center, scale, device, int(settings["batch_size"]))[0]
        for run in validation_runs
    ]
    return _build_normal_reference(
        validation_runs, run_errors, _network_id(checkpoint, model_path),
        preprocessing or checkpoint.get("preprocessing", PREPROCESSING),
    )


def _json_values(array):
    values = np.asarray(array, dtype=float)
    return np.where(np.isfinite(values), values, None).tolist()


def _median(array):
    values = np.asarray(array)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None


def order_band_energy(run) -> dict:
    """10–900 Hz 内按倍频 ±0.1× 分能量，其余频点归入异步能量。"""
    starts = np.asarray(run["start_time_s"])
    telemetry = run["telemetry"]
    times = np.asarray(telemetry["time_s"])
    selected = (times >= starts.min()) & (times <= starts.max() + 10)
    speeds = np.asarray(telemetry["actual_speed_rpm"])[selected]
    speed = _median(speeds[speeds > 0])
    rotation_hz = (speed if speed is not None else run["target_speed_rpm"]) / 60
    velocity = np.asarray(run["velocity"], dtype=np.float32)
    length = velocity.shape[-1]
    frequency = np.fft.rfftfreq(length, 1 / SAMPLE_RATE_HZ)
    power = (np.abs(np.fft.rfft(velocity, axis=-1) / length) ** 2 * 2).mean(axis=0)
    analysis_band = (frequency >= 10) & (frequency <= 900)
    synchronous = np.zeros_like(analysis_band)
    orders = list(range(1, int(900 / rotation_hz) + 1))
    energies, bands = [], []
    for order in orders:
        low, high = (order - 0.1) * rotation_hz, (order + 0.1) * rotation_hz
        selected = analysis_band & (frequency >= low) & (frequency <= high)
        synchronous |= selected
        energies.append(power[:, selected].sum(axis=-1))
        bands.append([max(10, low), min(900, high)])
    asynchronous = power[:, analysis_band & ~synchronous].sum(axis=-1)
    return {
        "basis": "shaft_order", "rotation_hz": float(rotation_hz),
        "speed_source": "actual" if speed is not None else "target",
        "half_width_order": 0.1, "analysis_band_hz": [10, 900], "bands_hz": bands,
        "labels": ["异步", *[f"{order}×" for order in orders]],
        "channels": CHANNELS, "values": np.stack([asynchronous, *energies], axis=1).tolist(),
        "unit": "(mm/s)²",
    }


def evaluate_run(run, model_path=None, config=None, progress=None) -> dict:
    """保留 DSP 与重建误差，只使用匹配的独立正常采集参考计算相容度。"""
    config = config or {}
    vibration_alpha = float(config.get("vibration_alpha", config.get("alpha", 0.05)))
    network_alpha = float(config.get("network_alpha", config.get("alpha", 0.05)))
    velocity = np.asarray(run["velocity"], dtype=np.float32)
    if not np.isfinite(velocity).all():
        raise ValueError("当前采集振动速度包含 NaN/Inf，正常相容度无效")
    starts = np.asarray(run["start_time_s"])
    windows = _windows(run)
    window_times = (starts[:, None] + np.arange(10)).ravel()
    rms = velocity_rms_mm_s(velocity)
    hann = np.hanning(20480)
    amplitude = (np.abs(np.fft.rfft(velocity * hann, axis=-1)) * (2 / hann.sum())).mean(axis=0)
    frequency = np.fft.rfftfreq(20480, 1 / SAMPLE_RATE_HZ)
    band = (frequency >= 10) & (frequency <= 900)
    band_indices = np.flatnonzero(band)
    # 保留稀疏背景和每通道主要谱峰，减少历史 JSON，同时不丢转频峰。
    peaks = [band_indices[np.argsort(channel[band])[-50:]] for channel in amplitude]
    displayed = np.unique(np.concatenate([band_indices[::10], *peaks]))
    temperature = np.asarray(run["temperature"]).transpose(1, 0, 2).reshape(2, -1)
    temperature_times = (starts[:, None] + np.arange(100) / 10).ravel()
    temperature_indices = np.linspace(0, len(temperature_times) - 1, min(600, len(temperature_times)), dtype=int)
    telemetry = run["telemetry"]
    telemetry_time = np.asarray(telemetry["time_s"])
    selected_telemetry = np.flatnonzero((telemetry_time >= starts.min()) & (telemetry_time <= starts.max() + 10))
    display_telemetry = selected_telemetry[::max(1, len(selected_telemetry) // 600)]
    speed = np.asarray(telemetry["actual_speed_rpm"])
    current = np.asarray(telemetry["current_a"])
    result = {
        "run_id": run["run_id"], "run_name": run["run_name"], "run_date": run["run_date"],
        "target_speed_rpm": run["target_speed_rpm"], "actual_speed_rpm": _median(speed[selected_telemetry]),
        "current_a": _median(current[selected_telemetry]),
        "temperature_c": [_median(channel) for channel in temperature],
        "rms_mm_s": rms.tolist(), "xy_rms_mm_s": float(np.linalg.norm(rms[:2])),
        "score": None, "analysis_score": None, "score_kind": SCORE_KIND, "alpha": network_alpha,
        "network_p_value": None, "vibration_p_value": None, "reconstruction_error_p95": None,
        "normalized_mse": None, "channel_mse": None,
        "window_errors": [], "window_time_s": window_times.tolist(),
        "reference_features": {"vibration": [], "network": []},
        "window_count": len(windows), "reconstruction": None,
        "waveform": {"time_s": (starts[0] + np.arange(2048) / SAMPLE_RATE_HZ).tolist(),
                     "channels": CHANNELS, "values": windows[0].tolist()},
        "spectrum": {"frequency_hz": frequency[displayed].tolist(), "channels": CHANNELS,
                     "amplitude_mm_s": amplitude[:, displayed].tolist()},
        "band_energy": order_band_energy(run),
        "temperature": {"time_s": temperature_times[temperature_indices].tolist(),
                        "channels": TEMPERATURE_CHANNELS, "values": _json_values(temperature[:, temperature_indices])},
        "telemetry": {"time_s": telemetry_time[display_telemetry].tolist(),
                      "speed_rpm": _json_values(speed[display_telemetry]),
                      "current_a": _json_values(current[display_telemetry])},
    }
    if progress:
        progress(40, "频谱、倍频能量和温度遥测已分析")
    invalid_reference = {"status": "invalid", "message": "未建立独立正常采集参考，请训练或重新校准",
                         "mu": None, "s": None, "M": 0}
    result["compatibility"] = {
        "vibration": normal_compatibility(result["xy_rms_mm_s"], invalid_reference, vibration_alpha, two_sided=True),
        "network": {**invalid_reference, "score": None, "p_value": None, "u": None, "alpha": network_alpha},
    }
    if model_path is not None and run["target_speed_rpm"] == 7000:
        import torch
        from core.algorithms.spindle_network import TCNAutoencoder

        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
        settings = {**TRAINING_DEFAULTS, **checkpoint["config"], **config}
        device = _device(torch, settings["device"])
        model = TCNAutoencoder().to(device)
        model.load_state_dict(checkpoint["model_state"])
        errors, reconstructed = _infer(
            model, windows, checkpoint["center"].to(device), checkpoint["scale"].to(device),
            device, int(settings["batch_size"]),
        )
        primary = errors[:, :2].mean(axis=1)
        feature = float(np.quantile(primary, 0.95))
        normal_reference = config.get("normal_reference", checkpoint.get("normal_reference"))
        group, message = _matched_reference(
            normal_reference, run, _network_id(checkpoint, model_path),
            config.get("preprocessing", checkpoint.get("preprocessing", PREPROCESSING)),
        )
        invalid_reference = {"status": "invalid", "message": message, "mu": None, "s": None, "M": 0}
        vibration_reference = group["vibration"] if group else invalid_reference
        network_reference = group["network"] if group else invalid_reference
        vibration = normal_compatibility(result["xy_rms_mm_s"], vibration_reference, vibration_alpha, two_sided=True)
        network = normal_compatibility(feature, network_reference, network_alpha)
        result.update({
            "score": network["score"], "analysis_score": vibration["score"],
            "network_p_value": network["p_value"], "vibration_p_value": vibration["p_value"],
            "compatibility": {"vibration": vibration, "network": network},
            "reconstruction_error_p95": feature if np.isfinite(feature) else None,
            "normalized_mse": float(primary.mean()) if np.isfinite(primary).all() else None,
            "channel_mse": _json_values(errors.mean(axis=0)), "window_channel_mse": _json_values(errors),
            "window_errors": _json_values(primary),
            "reference_features": {"vibration": vibration_reference.get("values", []),
                                   "network": network_reference.get("values", [])},
            "reconstruction": {"time_s": result["waveform"]["time_s"],
                               "original": windows[0].tolist(), "reconstructed": _json_values(reconstructed)},
        })
    if progress:
        progress(100, "分析完成" if result["score"] is not None else "信号分析完成，正常相容度无效或待校准")
    return result
