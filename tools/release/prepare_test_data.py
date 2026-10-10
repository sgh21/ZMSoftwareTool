"""生成独立的软件验证样例；不读取或改写现场数据，不随软件本体发布。"""

import argparse
import csv
from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import cv2
import h5py
import numpy as np
from openpyxl import Workbook


ROOT = Path(__file__).resolve().parents[2]
NOTE = "软件验证样例：全部人工模拟，不是实际设备测量，不能用于设备验收或训练生产模型。"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_zip(path, files):
    """固定顺序和 ZIP 时间，包内只放明确指定的文件。"""
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        for name, content in sorted(files.items()):
            entry = ZipInfo(name, date_time=(2026, 10, 10, 0, 0, 0))
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, content)


def transform(position):
    result = np.eye(4)
    result[:3, 3] = position
    return result


def robot_samples(folder):
    # 沿用 test_charuco_pose 中已验证的 OpenCV 原生标定板及相机几何。
    camera = [[1200, 0, 500], [0, 1200, 400], [0, 0, 1]]
    board = cv2.aruco.CharucoBoard(
        (9, 7), 20, 14, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    )
    board.setLegacyPattern(False)
    pattern = board.generateImage((900, 700), marginSize=0, borderBits=1)
    parameters = {
        "schema": "ur10_simulated_camera_parameters_v1", "length_unit": "mm",
        "source_note": NOTE,
        "camera": {"camera_matrix": camera, "dist_coeffs": [0] * 5, "image_size_px": [1000, 800]},
        "charuco": {"dictionary": "DICT_5X5_1000", "squares_xy": [9, 7],
                    "square_length_mm": 20, "marker_length_mm": 14, "legacy_pattern": False},
        "system_transforms": {"E_T_C_hand_eye": np.eye(4).tolist(),
                              "B_T_M_charuco_top_left": np.eye(4).tolist()},
    }
    write_json(folder / "parameters.json", parameters)
    ideal = transform([90, 70, -240])
    for batch_id, shifts, filename in (
        ("B001", [-1, 0, 1], "baseline_observations.json"),
        ("B002", [1, 3, 5], "current_observations.json"),
    ):
        samples, frames = [], []
        for index, shift in enumerate(shifts, 1):
            sample_id = f"{batch_id}_P001_D{index:03d}"
            vision = transform([-90 + shift / 5, -70, 240])
            samples.append({
                "point_id": "P001", "direction_id": f"D{index:03d}", "sample_id": sample_id,
                "vision_pose": vision.tolist(), "ideal_pose": ideal.tolist(),
            })
            # 这些像素是确定性的标定输入，不是照片或测量记录；不使用 PIL 或生成式画图。
            image = np.pad(pattern, ((50, 50), (50 + shift, 50 - shift)), constant_values=255)
            image_path = folder / "images" / batch_id / "calibration_images" / f"{sample_id}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imencode(".png", image)[1].tofile(image_path)
            frames.append({"image": f"calibration_images/{sample_id}.png", "ideal": ideal.tolist(),
                           "actual": np.linalg.inv(vision).tolist(), "source_note": NOTE})
        write_json(folder / filename, {
            "batch_id": batch_id, "label": f"模拟验证 {batch_id}", "length_unit": "mm",
            "program_id": "software_validation", "target_id": "fixed_charuco",
            "sampling_protocol": "multidirectional", "comparison_status": "simulation",
            "source_type": "simulation", "source_note": NOTE, "samples": samples,
        })
        write_json(folder / "images" / batch_id / "record.json", {
            "batch": batch_id, "is_simulated": True, "source_note": NOTE, "frames": frames,
        })


def spindle_samples(folder):
    # 沿用 test_spindle_algorithms.make_run 的解析信号，不需要网络权重或采集资产。
    folder.mkdir(parents=True, exist_ok=True)
    preprocessing = json.loads((ROOT / "config/spindle_monitoring.json").read_text(encoding="utf-8"))["preprocessing"]
    time = np.arange(20480) / 2048
    training_files = {}
    for index, amplitude in enumerate((0.05, 0.2, 0.045, 0.048, 0.052, 0.056, 0.06), 1):
        run_id = f"software_validation_simulated_{index:03d}"
        velocity = np.stack([
            amplitude * np.sin(2 * np.pi * 120 * time + phase)
            + 0.01 * np.sin(2 * np.pi * 240 * time) for phase in (0.0, 0.7, 1.4)
        ]).astype(np.float32)[None]
        if index > 2:
            # 五份独立种子的模拟记录，保留相同工况和时长；不复制单次采集凑数量。
            rng = np.random.default_rng(20261010 + index)
            for channel in range(3):
                for frequency in (80, 320, 500):
                    velocity[0, channel] += (
                        rng.uniform(0.0005, 0.002) * np.sin(2 * np.pi * frequency * time + rng.uniform(0, 2 * np.pi))
                    ).astype(np.float32)
        content = BytesIO()
        with h5py.File(content, "w") as h5:
            h5.attrs.update({
                "schema_version": 1, "run_id": run_id, "run_name": run_id, "run_date": "20000101",
                "captured_at": f"2000-01-01T00:0{index}:00+08:00", "target_speed_rpm": 7000,
                "velocity_sample_rate_hz": 2048, "velocity_unit": "mm/s",
                "velocity_channels": ["ACC1", "ACC2", "ACC3"], "velocity_frequency_band_hz": [10, 900],
                "preprocessing_id": preprocessing["id"], "is_simulated": True, "source_note": NOTE,
            })
            group = h5.create_group("windows_10s")
            for key, value in {
                "velocity": velocity, "start_time_s": [0.0],
                "temperature": np.full((1, 2, 100), 25.0, dtype=np.float32),
                "temperature_valid": np.ones((1, 2, 100), dtype=bool),
                "sample_valid": [True], "source_segment_index": [0],
            }.items():
                group.create_dataset(key, data=value, track_times=False)
        run = {
            "run_id": run_id, "run_name": f"模拟验证：振幅 {amplitude} mm/s",
            "captured_at": f"2000-01-01T00:0{index}:00+08:00", "speed_rpm": 7000,
            "operation": "idle", "condition": {"tool_remounted": False},
            "experiment_condition": "software_validation_simulated", "is_simulated": True,
            "source_note": NOTE, "data_file": "signals.h5", "telemetry_file": "telemetry.csv",
            "source_metadata_file": "source.json",
        }
        manifest = {
            "schema_version": 1, "package_id": run_id, "captured_date": "2000-01-01",
            "source_type": "historical_replay", "is_simulated": True, "source_note": NOTE,
            "preprocessing": preprocessing, "runs": [run],
        }
        telemetry = "time_s,actual_speed_rpm,current_a,speed_ok,current_ok\n0,7000,0.4,true,true\n9,7000,0.6,true,true\n"
        files = {
            "manifest.json": json.dumps(manifest, ensure_ascii=False).encode("utf-8"),
            "signals.h5": content.getvalue(), "telemetry.csv": telemetry.encode("utf-8"),
            "source.json": json.dumps({"is_simulated": True, "source_note": NOTE,
                                       "timestamp_note": "2000-01-01 为协议用模拟时间。"}, ensure_ascii=False).encode("utf-8"),
        }
        if index <= 2:
            write_zip(folder / f"simulated_{index:03d}.zip", files)
        else:
            package = BytesIO()
            write_zip(package, files)
            training_files[f"simulated_training_{index - 2:03d}.zip"] = package.getvalue()
    write_zip(folder / "小型建模.zip", training_files)


def feed_samples(folder):
    folder.mkdir(parents=True, exist_ok=True)
    fields = ["hole_id", "actual_depth_mm", "theoretical_depth_mm", "row_id", "batch_id",
              "condition_id", "data_source", "is_simulated"]
    for label, values in (("normal", [1.48, 1.49, 1.5, 1.51, 1.52]),
                          ("out_of_limit", [1.3, 1.4, 1.5, 1.6, 1.7, None])):
        rows = [[f"{index:03d}", value, 1.5, "R01", f"SIM_{label}", "software_validation",
                 "simulation", True] for index, value in enumerate(values, 1)]
        with (folder / f"simulated_{label}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            csv.writer(stream).writerows([fields, *rows])
        workbook = Workbook()
        workbook.properties.creator = "SoftwareTools software validation"
        workbook.properties.description = NOTE
        workbook.properties.created = datetime(2000, 1, 1)
        workbook.active.title = "模拟窝深"
        for row in [fields, *rows]:
            workbook.active.append(row)
        workbook.save(folder / f"simulated_{label}.xlsx")
        workbook.close()


README = """测试数据使用说明

全部文件均为人工构造的软件验证样例，不是实际设备测量，不用于设备精度验收或生产模型训练。
请先完整解压测试数据.zip，并在独立验收账户/空白软件存储中操作，避免混入生产记录。

1. 机器人末端定位精度
点击“加载参数”，选 01_机器人/parameters.json。
通过“导入观测”的观测文件入口选 baseline_observations.json，完成后“建立基准”。
再导入 current_observations.json，点击“评估精度”。
当前重复定位精度 X/空间均约 0.3266 mm；Y/Z 为 0。
重复定位精度退化 X/空间约 0.1633 mm；绝对定位精度退化 X/空间均为 0.6000 mm。
这组 JSON 使用解析矩阵，不检查图像检测。验证图片导入时请另建基准：
导入 images/B001/record.json，建立另一基准，再导入 images/B002/record.json 并评估。
六张 PNG 均为 OpenCV 确定性生成的 ChArUco 标定图；图片解算有像素量化误差，不能要求与解析结果完全相等。
不要混用 JSON 组与图片组的基准。图片目录名应保留，禁止复制一张图充当多个独立测量。

2. 主轴回转精度
“日常导入”依次选 02_主轴/simulated_001.zip、simulated_002.zip，不要先解开这两个内层 ZIP。
每包只有一次模拟采集、一个 10 秒原始窗口（分析得到 10 个 1 秒窗口），三路振动、两路温度和遥测。
频谱主峰为 120 Hz；第一包 ACC1 RMS 约 0.03606 mm/s，第二包约 0.14160 mm/s。
“评估精度”可重新分析。未训练网络时相容度/网络评分为空属于预期；这里不提供预训练模型。
人工标签保持“未判定”，振幅变化本身不能证明设备故障。
包内 source_type=historical_replay 是现有协议枚举；is_simulated=true 和 source_note 明确标明人工模拟。
2000-01-01 是协议用模拟时间，不是真实采集日期。

可选的发布包训练检查（仅在独立验收账户/空白模型存储操作）：
“批量导入”选择 02_主轴/小型建模.zip，确认整批正常以验证软件流程；这不是对实际设备的正常判定。
该包有五次独立种子、不同振幅的人工模拟采集，同工况、同长度；软件可分为两次训练和三次校准。
将训练轮数设为 1，点击“训练网络”，完成后“日常导入”两个 simulated ZIP 并“评估精度”。
预期生成模型版本、重建结果和历史记录。仅检查训练/推理能完成，不要求评分达到某个数值。
五次极小样例和一轮训练不能证明模型有效；生成的验收模型不要用于生产。

3. 主轴轴向进给精度
“导入数据”选择 03_窝深/simulated_normal.csv，点击“评估精度”；也可重复使用同名 XLSX。
采用逐孔理论窝深，设置误差下限 -0.1 mm、上限 0.1 mm。
normal：5 个有效孔，平均 1.5000 mm，总体方差 0.0002 mm²，0 个超限。
out_of_limit：6 行、5 个有效孔，平均 1.5000 mm，总体方差 0.0200 mm²，2 个超限。
其中 1.4/1.6 mm 位于边界不超限，最后一行缺测保留为空、不补零。
CSV 与 XLSX 内容相同；切换文件用于验证覆盖导入，不视为两次独立采集。

这些样例只验证导入、计算、资源和保存等软件行为，不能代替现场传感器、标定精度或模型有效性验收。
"""


def prepare_test_data(output):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="test_data_", dir=output) as temporary:
        folder = Path(temporary)
        robot_samples(folder / "01_机器人")
        spindle_samples(folder / "02_主轴")
        feed_samples(folder / "03_窝深")
        (folder / "使用说明.txt").write_text(README, encoding="utf-8-sig")
        write_json(folder / "manifest.json", {
            "schema_version": 1, "is_simulated": True, "source_note": NOTE,
            "robot": {
                "parameters": "01_机器人/parameters.json",
                "baseline": "01_机器人/baseline_observations.json",
                "current": "01_机器人/current_observations.json",
                "image_baseline": "01_机器人/images/B001/record.json",
                "image_current": "01_机器人/images/B002/record.json",
                "expected": {"repeatability": [0.326598632371, 0, 0, 0.326598632371],
                             "repeatability_change": [0.1632993161855, 0, 0, 0.1632993161855],
                             "absolute_change": [0.6, 0, 0, 0.6]},
            },
            "spindle": {
                "packages": ["02_主轴/simulated_001.zip", "02_主轴/simulated_002.zip"],
                "training_package": "02_主轴/小型建模.zip",
                "training": {"label": "healthy", "epochs": 1, "candidate_count": 5,
                             "expected_train_count": 2, "expected_calibration_count": 3,
                             "note": "仅检查从随机初始化训练及推理完成，不验证设备或模型有效性。"},
                "expected": {"run_count": 2, "model": None, "spectrum_peak_hz": 120,
                             "rms_acc1_mm_s": [float(np.sqrt(0.0013)), float(np.sqrt(0.02005))]},
            },
            "feed": {
                "normal": ["03_窝深/simulated_normal.csv", "03_窝深/simulated_normal.xlsx"],
                "out_of_limit": ["03_窝深/simulated_out_of_limit.csv", "03_窝深/simulated_out_of_limit.xlsx"],
                "settings": {"depth_mode": "per_hole", "error_lower_mm": -0.1, "error_upper_mm": 0.1},
                "expected": {"valid_count": 5, "mean_depth_mm": 1.5,
                             "normal": {"variance_mm2": 0.0002, "alarm_count": 0},
                             "out_of_limit": {"variance_mm2": 0.02, "alarm_count": 2, "missing_count": 1}},
            },
        })
        destination = output / "测试数据.zip"
        write_zip(destination, {path.relative_to(folder).as_posix(): path.read_bytes()
                                for path in folder.rglob("*") if path.is_file()})
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "dist/release_inputs")
    args = parser.parse_args()
    print(prepare_test_data(args.output))
