"""机器人定位业务：参数版本、观测导入、基准、评估和历史；不依赖 Qt。"""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from core.algorithms.board_pose import (
    estimate_board_pose, estimate_charuco_pose, make_charuco_board,
)
from core.algorithms.pose_fields import rotation_to_rpy_degrees
from core.algorithms.position_monitoring import (
    evaluate_current_repeatability,
    evaluate_multidirectional,
    evaluate_position_monitoring,
    validate_transform,
)
from core.runtime_paths import bundle_root, data_root
from core.services.position_image_input import IMAGE_SUFFIXES, image_batch, simulation_parameters
from core.services.position_persistence import PositionStore, observation_time, parse_timestamp, write_document


METRIC_LABELS = {
    "absolute_change": "绝对定位精度退化",
    "repeatability_change": "重复定位精度退化",
    "repeatability": "当前重复定位精度",
}
METRIC_AXES = ("X", "Y", "Z", "distance")
PARAMETER_METADATA = {"version", "source_path", "source_note", "created_at", "updated_at", "description", "notes"}
def metric_values(result, mode, group=None):
    """统一卡片、逐点结果、历史曲线和阈值判定的四列取值。"""
    source = result["summary"] if group is None else group
    rms = result.get("metric_definition") == "patent_v6_rms"
    if mode == "repeatability":
        source = source if group is None else group["current"]
        axes_key = "axis_rms_base" if rms else "axis_3sigma_base"
        scalar_key = "scatter_rms" if rms else "rp"
        if group is None:
            scalar_key += "_current"
    elif mode == "repeatability_change":
        axes_key = "axis_rms_change_base" if rms else "axis_3sigma_change_base"
        scalar_key = "scatter_rms_change" if rms else "rp_change"
    elif mode == "absolute_change":
        axes_key = "centroid_shift_abs_base" if rms else "absolute_axis_change"
        scalar_key = "centroid_shift_distance" if rms else "absolute_ap_change"
    else:
        raise ValueError(f"未知评价指标：{mode}")
    axes = source.get(axes_key)
    return [*(axes if axes is not None else [None, None, None]), source.get(scalar_key)]


def assess_metric(result, mode):
    """按结果定义逐点或逐方向判定；有符号退化量不取绝对值。"""
    if mode not in METRIC_LABELS:
        raise ValueError(f"未知评价指标：{mode}")
    stored = result.get("metric_thresholds", {}).get(mode, {})
    thresholds = {axis: stored.get(axis) for axis in METRIC_AXES}
    alarms = []
    values = []
    groups = result.get("groups", [])
    if mode == "absolute_change" and result.get("metric_definition") != "patent_v6_rms":
        groups = [entry for group in groups for entry in group.get("directions", [group])]
    for group in groups:
        group_values = metric_values(result, mode, group)
        values.extend(group_values)
        for axis, value in zip(METRIC_AXES, group_values):
            threshold = thresholds[axis]
            if value is not None and threshold is not None and value > threshold:
                alarms.append({
                    "metric": mode, "point_id": group["point_id"],
                    "direction_id": group["direction_id"], "axis": axis,
                    "value": value, "threshold": threshold,
                })
    if alarms:
        status = "超限"
    elif not any(value is not None for value in values):
        status = "指标不可计算"
    elif all(value is None for value in thresholds.values()):
        status = "未设置阈值"
    elif any(value is None for value in values):
        status = "部分指标不可判定"
    elif any(value is None for value in thresholds.values()):
        status = "已设阈值项内"
    else:
        status = "阈值内"
    assessment_status = status
    # 旧历史只有状态文字，仍需保留其调试限制。
    if (result.get("comparison_status") == "debug_unverified"
            or str(result.get("status", "")).startswith("调试比较")):
        status = "调试比较 · 采样对应待确认"
    return {"status": status, "assessment_status": assessment_status,
            "alarms": alarms, "thresholds": thresholds}


def read_document(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    document = yaml.safe_load(text) if path.suffix.lower() in (".yaml", ".yml") else json.loads(text)
    if not isinstance(document, dict):
        raise ValueError("输入文件必须是 JSON/YAML 对象")
    return document


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _newer_observation(candidate, previous):
    """调试同一程序按模拟日，实测按采集/导入时间；重评时间仅用于同次观测。"""
    if previous is None:
        return True
    same_program = all(candidate.get(key) == previous.get(key) for key in ("program_id", "target_id"))
    days = [item.get("debug_day_index") for item in (candidate, previous)]
    if same_program and all(day is not None for day in days) and days[0] != days[1]:
        return days[0] > days[1]
    return (parse_timestamp(candidate["observed_at"]), parse_timestamp(candidate["created_at"])) >= (
        parse_timestamp(previous["observed_at"]), parse_timestamp(previous["created_at"]))


def _pose_mm(value, name, factor):
    pose = validate_transform(value, name).copy()
    pose[:3, 3] *= factor
    return pose.tolist()


def _rotation(value, name):
    transform = np.eye(4)
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"{name} 必须为 3×3 矩阵")
    transform[:3, :3] = rotation
    return validate_transform(transform, name)[:3, :3].tolist()


def _baseline_content(batch, parameters):
    """精确比较计算输入；保留参考姿态与重复次数，不用汇总值或文件来源去重。"""
    references = {}
    samples = []
    for sample in batch["samples"]:
        references.setdefault(sample["point_id"], sample["vision_pose"])
        samples.append({key: sample.get(key) for key in ("point_id", "direction_id", "vision_pose", "ideal_pose")})
    samples.sort(key=lambda item: (item["point_id"], item["direction_id"],
                 tuple(np.asarray(item["vision_pose"]).flat),
                 tuple(np.asarray(item["ideal_pose"]).flat) if item["ideal_pose"] is not None else ()))
    return {
        "parameters": {key: value for key, value in parameters.items()
                       if key not in PARAMETER_METADATA or key == "version"},
        "conditions": {key: batch.get(key) for key in (
            "parameter_version", "length_unit", "program_id", "target_id", "sampling_protocol",
            "comparison_status", "orientation_source")},
        "base_rotations": batch.get("base_rotations", {}),
        "initial_errors": batch.get("initial_errors", {}),
        "references": references, "samples": samples,
    }


class PositionMonitoringService:
    def __init__(self, root=None, *, defer_restore=False):
        self.root = Path(root if root is not None else data_root()).resolve()
        self.storage = self.root / "storage" / "position_monitoring"
        self._store = PositionStore(self.storage)
        self.parameter_path = self.storage / "parameters" / "current.json"
        self.previous_parameter_path = self.storage / "parameters" / "previous.json"
        defaults = bundle_root() / "config" / "robot_position.json"
        configured = self.root / "config" / "robot_position.json"
        self.parameters = read_document(configured if configured.exists() else defaults)
        self.settings = {
            "thresholds": dict.fromkeys(("X", "Y", "Z", "distance")),
            "metric_thresholds": {mode: dict.fromkeys(METRIC_AXES) for mode in METRIC_LABELS},
            "multidirectional_thresholds": {mode: dict.fromkeys(METRIC_AXES) for mode in METRIC_LABELS},
            "processing_points": [],
        }
        self.baseline = None
        self.current_batch = None
        self.latest_batch = None
        self.latest_result = None
        self.latest_comparison_error = ""
        self.history_comparison_warnings = []
        self._latest_evaluation = None
        self._latest_result_ref = None
        self._latest_comparison_key = None
        self._history_cache = None
        self._history_revision = 0
        if not defer_restore:
            self.restore()

    def restore(self, progress=None):
        """恢复磁盘状态；桌面可在工作线程中调用，构造空页面不读取大观测。"""
        if progress:
            progress(0, "读取定位监控参数与状态")
        state_path = self.storage / "state.json"
        state = read_document(state_path) if state_path.exists() else {}
        if self.parameter_path.exists():
            self.parameters = read_document(self.parameter_path)
        else:
            self.parameters = state.get("parameters", self.parameters)
            write_document(self.parameter_path, self.parameters)
        self.settings.update(state.get("settings", {}))
        if progress:
            progress(15, "恢复所选基准与当前观测")
        if state.get("baseline_path"):
            self.baseline = self._read_baseline(state["baseline_path"])
        if state.get("current_batch_path"):
            self.current_batch = read_document(state["current_batch_path"])
            self.current_batch["saved_path"] = state["current_batch_path"]
        # 旧版切换基准会清空 state 的结果指针，仍从实际已评估观测恢复。
        # 不按最后评估时间挑选，避免早期观测重评或建立基准覆盖最新采集。
        if progress:
            progress(40, "读取已评估历史，确定最新观测")
        for record in self.list_history():
            if record.get("current_batch_path") and _newer_observation(record, self._latest_evaluation):
                self._latest_evaluation = record
        if self._latest_evaluation:
            if progress:
                progress(60, "恢复最新已评估观测")
            record = self._latest_evaluation
            self.latest_batch = self._read_evaluated_batch(record)
            legacy = self.storage / "history" / f"{record['id']}.json"
            day = record.get("debug_day_index")
            date = parse_timestamp(record["observed_at"]).astimezone().date().isoformat()
            path = legacy if legacy.exists() else (
                self.storage / "debug" / f"day_{day:04d}.json" if day is not None
                else self.storage / "daily" / f"{date}.json")
            self._latest_result_ref = {"path": str(path), "id": record["id"]}
            if self.current_batch is None:
                self.current_batch = deepcopy(self.latest_batch)
        if self.current_batch and self.current_batch.get("parameter_version") != self.parameters.get("version"):
            self.current_batch = None
        if self.current_batch:
            observation_time(self.current_batch, self._latest_evaluation["created_at"] if self._latest_evaluation else None)
        self.compare_latest((lambda value, message: progress(80 + round(value * 0.19), message)) if progress else None)
        if "current_batch_path" not in state or "parameters" in state:
            self._save_state()
        if progress:
            progress(100, "定位监控状态恢复完成")

    def _save_state(self):
        write_document(self.storage / "state.json", {
            "settings": self.settings,
            "baseline_path": self.baseline["path"] if self.baseline else None,
            "current_batch_path": self.current_batch.get("saved_path") if self.current_batch else None,
            "latest_result": self._latest_result_ref,
        })

    @staticmethod
    def _read_baseline(path):
        baseline = read_document(path)
        if "batch_path" in baseline:
            baseline["batch"] = read_document(baseline["batch_path"])
            baseline["batch"]["saved_path"] = baseline["batch_path"]
        if "batch" in baseline:
            observation_time(baseline["batch"], baseline.get("created_at"), "baseline_created_at")
        baseline["path"] = str(Path(path).resolve())
        return baseline

    @staticmethod
    def _read_evaluated_batch(result):
        batch = read_document(result["current_batch_path"])
        batch["saved_path"] = result["current_batch_path"]
        observation_time(batch, result["created_at"])
        return batch

    def load_evaluation_samples(self, result):
        current = read_document(result["current_batch_path"])
        baseline = self._read_baseline(result["baseline_path"])["batch"] if result.get("baseline_path") else None
        return {"current": current, "baseline": baseline}

    def append_log(self, message, level="INFO"):
        return self._store.append_log(message, level)

    def list_logs(self):
        return self._store.read_logs()

    def load_parameters(self, path):
        document = read_document(path)
        if document.get("schema") == "ur10_simulated_camera_parameters_v1":
            document = simulation_parameters(document)
        # 旧 master 的手眼输出 T_tool_cam 明确使用 m；只在此导入边界换算。
        if "T_tool_cam" in document:
            document = {**self.parameters,
                        "hand_eye": _pose_mm(document["T_tool_cam"], "旧手眼参数", 1000), "length_unit": "mm"}
            document["source_note"] = "旧 master T_tool_cam 导入，平移由 m 转为 mm"
        elif ("camera_matrix" in document and "hand_eye" in document
              and any(key in document for key in ("board_grid", "charuco", "board_type"))):
            # 完整配置替换测量条件；未声明的新字段不能继承上一次仿真配置。
            defaults = read_document(bundle_root() / "config" / "robot_position.json")
            defaults.update({
                "board_type": "checkerboard", "charuco": {}, "target_pose_base": None,
                "image_size_px": None, "end_frame": None, "target_id": "configured_target",
                "source_note": "",
            })
            reset = {key: value for key, value in defaults.items()
                     if key in self.parameters or key in document}
            document = {**reset, **document}
        document["source_path"] = str(Path(path).resolve())
        return self.save_parameters(document)

    def save_parameters(self, document):
        parameters = deepcopy({**self.parameters, **document})
        if parameters.get("length_unit", "mm") not in ("mm", "m"):
            raise ValueError("参数长度单位只支持 mm 或 m")
        if parameters.get("transform_convention", "E_T_C") != "E_T_C":
            raise ValueError("手眼变换必须声明为 E_T_C（相机到被监测末端）")
        if parameters.get("hand_eye") is not None:
            factor = 1000 if "hand_eye" in document and document.get("length_unit") == "m" else 1
            parameters["hand_eye"] = _pose_mm(parameters["hand_eye"], "手眼参数", factor)
        if parameters.get("camera_matrix") is not None:
            camera = np.asarray(parameters["camera_matrix"], dtype=float)
            if (camera.shape != (3, 3) or not np.isfinite(camera).all()
                    or camera[0, 0] <= 0 or camera[1, 1] <= 0):
                raise ValueError("相机内参必须为有效 3×3 矩阵，焦距为正")
            parameters["camera_matrix"] = camera.tolist()
        distortion = np.asarray(parameters["dist_coeffs"], dtype=float).reshape(-1)
        if len(distortion) not in (0, 4, 5, 8, 12, 14) or not np.isfinite(distortion).all():
            raise ValueError("相机畸变参数数量应为 0、4、5、8、12 或 14")
        parameters["dist_coeffs"] = distortion.tolist()
        grid = parameters["board_grid"]
        if len(grid) != 2 or any(int(n) != n or n < 2 for n in grid):
            raise ValueError("棋盘内角点列数和行数必须为至少 2 的整数")
        parameters["board_grid"] = [int(n) for n in grid]
        size = float(parameters["square_size_mm"])
        if not np.isfinite(size) or size <= 0:
            raise ValueError("棋盘格长必须为正数，单位 mm")
        parameters["square_size_mm"] = size
        if parameters.get("board_type", "checkerboard") not in ("checkerboard", "charuco"):
            raise ValueError("标定板类型只支持普通棋盘或 ChArUco")
        if parameters.get("board_type") == "charuco":
            make_charuco_board(parameters.get("charuco", {}), size)
        if parameters.get("target_pose_base") is not None:
            factor = 1000 if "target_pose_base" in document and document.get("length_unit") == "m" else 1
            parameters["target_pose_base"] = _pose_mm(parameters["target_pose_base"], "固定靶标基座位姿", factor)
        if parameters.get("reference_rotation") is not None:
            parameters["reference_rotation"] = _rotation(parameters["reference_rotation"], "棋盘参考旋转")
        limit = parameters.get("max_reprojection_error_px")
        if limit is not None and (not np.isfinite(float(limit)) or float(limit) <= 0):
            raise ValueError("重投影误差限值必须为正数或 null")
        parameters["max_reprojection_error_px"] = float(limit) if limit is not None else None
        parameters["length_unit"] = "mm"
        parameters["transform_convention"] = "E_T_C"
        effective = {key: value for key, value in parameters.items() if key not in PARAMETER_METADATA}
        previous_effective = {key: value for key, value in self.parameters.items() if key not in PARAMETER_METADATA}
        if effective == previous_effective:
            return deepcopy(self.parameters)
        parameters["version"] = _stamp()
        write_document(self.previous_parameter_path, self.parameters)
        write_document(self.parameter_path, parameters)
        self.parameters = parameters
        self.current_batch = None
        self.compare_latest()
        self._save_state()
        return deepcopy(parameters)

    def save_settings(self, settings, progress=None):
        updated = deepcopy({**self.settings, **settings})
        threshold_groups = [updated["thresholds"]]
        for key in ("metric_thresholds", "multidirectional_thresholds"):
            if key in settings:
                updated[key] = deepcopy(self.settings[key])
                for mode, thresholds in settings[key].items():
                    if mode not in METRIC_LABELS:
                        raise ValueError(f"未知评价指标：{mode}")
                    updated[key][mode].update(thresholds)
            threshold_groups.extend(updated[key].values())
        for thresholds in threshold_groups:
            for axis in METRIC_AXES:
                value = thresholds.get(axis)
                if value is not None:
                    value = float(value)
                    if not np.isfinite(value) or value < 0:
                        raise ValueError(f"{axis} 阈值必须为非负数或留空")
                thresholds[axis] = value
        previous = (self.settings, deepcopy(self.latest_result), self._latest_comparison_key, self.latest_comparison_error)
        self.settings = updated
        try:
            self.compare_latest((lambda value, message: progress(round(value * 0.95), message)) if progress else None)
            self._save_state()
        except Exception:
            self.settings, self.latest_result, self._latest_comparison_key, self.latest_comparison_error = previous
            raise
        if progress:
            progress(100, "设置已保存")

    def load_observations(self, path, progress=None):
        """导入单批图片/目录/观测文件，逐图解算后立即保存可复用的观测结果。"""
        if isinstance(path, (list, tuple)) or Path(path).is_dir() or Path(path).suffix.lower() in IMAGE_SUFFIXES:
            batch = image_batch(path, self.parameters)
            source = Path(batch["source_path"])
        else:
            source = Path(path).resolve()
            batch = read_document(source)
            if "frames" in batch:
                batch = image_batch(source, self.parameters)
        if batch.get("parameter_version") and batch["parameter_version"] != self.parameters["version"]:
            raise ValueError("观测结果的参数版本与当前参数不一致，请恢复对应参数或重新导入原图")
        if batch.get("length_unit") not in ("mm", "m"):
            raise ValueError("观测清单需声明 length_unit 为 mm 或 m")
        for field in ("program_id", "target_id"):
            if not batch.get(field):
                raise ValueError(f"观测清单缺少 {field}，无法确认跨期测量条件")
        samples = batch.get("samples", [])
        if not samples:
            raise ValueError("观测清单没有 samples")
        factor = 1000 if batch["length_unit"] == "m" else 1
        resolved = []
        rejected = []
        identities = set()
        references = {}
        if (self.baseline and batch.get("comparison_status") != "calibration_only"
                and all(batch[key] == self.baseline["batch"][key]
                        for key in ("program_id", "target_id"))):
            for item in self.baseline["batch"]["samples"]:
                references.setdefault(item["point_id"], np.asarray(item["vision_pose"])[:3, :3])
        for index, original in enumerate(samples):
            sample = deepcopy(original)
            for field in ("point_id", "direction_id", "sample_id"):
                if not str(sample.get(field, "")).strip():
                    raise ValueError(f"样本 {index + 1} 缺少 {field}")
                sample[field] = str(sample[field])
            identity = tuple(sample[field] for field in ("point_id", "direction_id", "sample_id"))
            if identity in identities:
                raise ValueError(f"重复样本编号 {identity}；连拍不能冒充重新到达")
            identities.add(identity)
            image_path = sample.get("image_path")
            if image_path:
                image_path = Path(image_path)
                if not image_path.is_absolute():
                    image_path = source.parent / image_path
                sample["image_path"] = str(image_path.resolve())
            if "vision_pose" in sample:
                sample["vision_pose"] = _pose_mm(sample["vision_pose"], "视觉位姿", factor)
            else:
                if not image_path or not image_path.is_file():
                    raise ValueError(f"样本 {identity} 缺少视觉位姿或有效图像")
                if self.parameters.get("camera_matrix") is None:
                    raise ValueError("从图像解算位姿前请设置相机内参")
                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"无法读取图像 {image_path}")
                image_size = self.parameters.get("image_size_px")
                if image_size and list(image.shape[:2][::-1]) != image_size:
                    raise ValueError(f"图像 {image_path.name} 尺寸与相机内参对应分辨率不一致")
                reference = references.get(sample["point_id"])
                if reference is None:
                    reference = self.parameters.get("reference_rotation")
                try:
                    if self.parameters.get("board_type") == "charuco":
                        estimate = estimate_charuco_pose(
                            image, self.parameters["camera_matrix"], self.parameters["dist_coeffs"],
                            self.parameters["charuco"], self.parameters["square_size_mm"],
                        )
                    else:
                        estimate = estimate_board_pose(
                            image, self.parameters["camera_matrix"], self.parameters["dist_coeffs"],
                            self.parameters["board_grid"], self.parameters["square_size_mm"],
                            reference_rotation=reference,
                        )
                    limit = self.parameters.get("max_reprojection_error_px")
                    if limit is not None and estimate["reprojection_error_px"] > float(limit):
                        raise ValueError(f"重投影误差 {estimate['reprojection_error_px']:.4f}px 超出设置限值")
                except ValueError as error:
                    # 标定清单可显式记录剔除；监测数据不静默删掉失效到达。
                    if batch.get("comparison_status") != "calibration_only":
                        raise ValueError(f"{image_path.name}: {error}") from error
                    rejected.append({
                        "point_id": sample["point_id"], "direction_id": sample["direction_id"],
                        "sample_id": sample["sample_id"], "image_path": str(image_path.resolve()),
                        "reason": str(error),
                    })
                    if progress:
                        progress(round(75 * (index + 1) / len(samples)), f"跳过 {image_path.name}: {error}")
                    continue
                sample.update({key: value.tolist() if isinstance(value, np.ndarray) else value
                               for key, value in estimate.items()
                               if key not in ("image_points", "projected_points")})
            references.setdefault(sample["point_id"], np.asarray(sample["vision_pose"])[:3, :3])
            vision = np.asarray(sample["vision_pose"])
            sample["vision_xyz_mm"] = vision[:3, 3].tolist()
            sample["vision_rpy_deg"] = rotation_to_rpy_degrees(vision[:3, :3]).tolist()
            if "ideal_pose" in sample:
                sample["ideal_pose"] = _pose_mm(sample["ideal_pose"], "指令目标位姿", factor)
            if self.parameters.get("target_pose_base") is not None and self.parameters.get("hand_eye") is not None:
                end_pose = (np.asarray(self.parameters["target_pose_base"]) @ np.linalg.inv(vision)
                            @ np.linalg.inv(np.asarray(self.parameters["hand_eye"])))
                sample["end_pose"] = end_pose.tolist()
                sample["position_base_mm"] = end_pose[:3, 3].tolist()
                if "ideal_pose" in sample:
                    sample["error_base_mm"] = (end_pose[:3, 3] - np.asarray(sample["ideal_pose"])[:3, 3]).tolist()
            if "robot_pose" in sample:
                sample["robot_pose"] = _pose_mm(sample["robot_pose"], "机器人记录位姿", factor)
            resolved.append(sample)
            if progress:
                progress(round(75 * (index + 1) / len(samples)), f"已解算/读取 {index + 1}/{len(samples)}")
        if not resolved:
            raise ValueError("没有可用观测，请检查图像、棋盘和相机参数")
        batch["base_rotations"] = {
            str(point): _rotation(rotation, f"测点 {point} 的 Q")
            for point, rotation in batch.get("base_rotations", {}).items()
        }
        for directions in batch.get("initial_errors", {}).values():
            for direction, vector in directions.items():
                value = np.asarray(vector, dtype=float)
                if value.shape != (3,) or not np.isfinite(value).all():
                    raise ValueError("初始绝对误差须为基座三轴有限数值向量")
                directions[direction] = (value * factor).tolist()
        batch.update({
            "schema_version": 2, "batch_id": str(batch.get("batch_id") or _stamp()),
            "label": batch.get("label") or source.stem, "samples": resolved,
            "length_unit": "mm", "source_path": str(source),
            "parameter_version": self.parameters["version"], "rejected_samples": rejected,
        })
        batch.setdefault("warnings", [])
        if batch.get("orientation_source") == "controller_approximation":
            batch["warnings"].append("基座方向来自控制器记录，属于近似朝向，非独立实测姿态")
        batch["parameters"] = deepcopy(self.parameters)
        batch.setdefault("imported_at", datetime.now(timezone.utc).isoformat())
        observation_time(batch)
        identifier = _stamp()
        self._store.manage_images(batch, identifier, (
            lambda value, message: progress(75 + round(value * 0.2), message)) if progress else None)
        if progress:
            progress(95, "保存观测结果")
        saved_path = self.storage / "observations" / f"{identifier}.json"
        batch["saved_path"] = str(saved_path)
        write_document(saved_path, batch)
        self._store.record("observations", {
            "id": identifier, "batch_id": batch["batch_id"], "path": str(saved_path),
            "observed_at": batch["observed_at"], "time_source": batch["time_source"],
        }, batch, self.parameters)
        previous = self.current_batch
        self.current_batch = batch
        try:
            self._save_state()
        except OSError:
            self.current_batch = previous
            raise
        if progress:
            progress(100, "图像托管与观测保存完成")
        return deepcopy(batch)

    def create_baseline(self, name="", *, evaluate=True, progress=None):
        if self.current_batch is None:
            raise ValueError("请先导入初始观测")
        if self.parameters.get("hand_eye") is None:
            raise ValueError("建立基准前请加载相机到末端的手眼参数")
        if self.current_batch.get("comparison_status") == "calibration_only":
            raise ValueError("手眼标定数据包含不同目标位姿，不能直接建立监控基准")
        if self.current_batch["parameter_version"] != self.parameters["version"]:
            raise ValueError("观测与当前参数版本不一致，请重新导入")
        content = _baseline_content(self.current_batch, self.parameters)
        existing = self._distinct_baselines((
            lambda value, message: progress(round(value * 0.6), message)) if progress else None)
        for document, previous_content in existing:
            if content == previous_content:
                if progress:
                    progress(100, "相同测量内容的基准已存在，未新建")
                return {"duplicate": True, "existing_baseline": {
                    key: document[key] for key in ("id", "label", "path", "created_at")}}
        if progress:
            progress(65, "检查完成，保存新基准")
        identifier = _stamp()
        path = self.storage / "baselines" / f"{identifier}.json"
        baseline = {
            "id": identifier, "label": name or self.current_batch["label"],
            "created_at": datetime.now(timezone.utc).isoformat(), "path": str(path),
            "parameters": deepcopy(self.parameters), "batch_path": self.current_batch["saved_path"],
        }
        write_document(path, baseline)
        baseline["batch"] = deepcopy(self.current_batch)
        self.baseline = baseline
        self._store.record("baselines", {
            "id": identifier, "label": baseline["label"], "path": str(path),
            "batch_path": baseline["batch_path"], "created_at": baseline["created_at"],
        }, self.current_batch, self.parameters)
        if evaluate:
            self.evaluate((lambda value, message: progress(70 + round(value * 0.29), message)) if progress else None)
        else:
            self.current_batch = None
            self.compare_latest()
        self._save_state()
        if progress:
            progress(100, "基准建立完成")
        return deepcopy(baseline)

    def _distinct_baselines(self, progress=None):
        """折叠相同输入的旧基准；当前选中项作代表，磁盘文件保持原样。"""
        paths = sorted((self.storage / "baselines").glob("*.json"))
        distinct = []
        if progress:
            progress(0, "检查已保存基准的完整测量内容")
        for index, path in enumerate(paths):
            document = self._read_baseline(path)
            content = _baseline_content(document["batch"], document["parameters"])
            match = next((position for position, (_, previous) in enumerate(distinct) if content == previous), None)
            if match is None:
                distinct.append((document, content))
            elif self.baseline and document["id"] == self.baseline["id"]:
                distinct[match] = (document, content)
            if progress:
                progress(round(100 * (index + 1) / len(paths)), f"已检查基准 {index + 1}/{len(paths)}")
        if progress and not paths:
            progress(100, "尚无已保存基准")
        return distinct

    def list_baselines(self, progress=None):
        return [{key: document[key] for key in ("id", "label", "path", "created_at")}
                for document, _ in self._distinct_baselines(progress)]

    def select_baseline(self, path, progress=None):
        if progress:
            progress(0, "读取所选基准")
        baseline = self._read_baseline(path)
        if not all(key in baseline for key in ("id", "batch", "parameters")):
            raise ValueError("所选文件不是定位监控基准")
        previous = (self.baseline, deepcopy(self.latest_result), self._latest_comparison_key, self.latest_comparison_error)
        self.baseline = baseline
        try:
            self.compare_latest((lambda value, message: progress(25 + round(value * 0.7), message)) if progress else None)
            self._save_state()
        except Exception:
            self.baseline, self.latest_result, self._latest_comparison_key, self.latest_comparison_error = previous
            raise
        if progress:
            progress(100, "基准切换完成")
        return deepcopy(baseline)

    def _has_comparable_baseline(self):
        if self.baseline is None or self.current_batch is None:
            return False
        if self.baseline["parameters"]["version"] != self.parameters["version"]:
            return False
        initial = self.baseline["batch"]
        current = self.current_batch
        if initial.get("sampling_protocol") != current.get("sampling_protocol"):
            return False
        if any(initial[field] != current[field] for field in ("program_id", "target_id")):
            return False
        initial_groups = {(item["point_id"], item["direction_id"]) for item in initial["samples"]}
        current_groups = {(item["point_id"], item["direction_id"]) for item in current["samples"]}
        return initial_groups == current_groups

    def evaluate(self, progress=None, *, allow_current_only=False):
        if allow_current_only and not self._has_comparable_baseline():
            return self.evaluate_current(progress)
        if self.baseline is None or self.current_batch is None:
            raise ValueError("评估需要已保存的基准和本次复测观测")
        if self.current_batch["parameter_version"] != self.parameters["version"]:
            raise ValueError("参数版本已变化，请选择原基准参数并重新导入，或建立新基准")
        if progress:
            progress(0, "比较本次观测与基准")
        result = self._compare_batches(self.current_batch, self.baseline, self.parameters)
        return self._save_evaluation(result, self.current_batch, progress)

    def _compare_batches(self, current, baseline, parameters):
        """只比较已保存的视觉位姿；无图像解算、状态修改或文件写入。"""
        if baseline is None:
            return self._current_repeatability(current, parameters)
        baseline_batch = baseline["batch"]
        version = baseline["parameters"]["version"]
        if version != parameters["version"] or current["parameter_version"] != version:
            raise ValueError("参数版本已变化，所选基准与观测不能比较")
        if current.get("comparison_status") == "calibration_only":
            raise ValueError("手眼标定数据不能直接作为复测数据")
        for field in ("program_id", "target_id"):
            if baseline_batch[field] != current[field]:
                raise ValueError(f"基准与复测的 {field} 不一致，不能直接比较")
        if baseline_batch.get("sampling_protocol") != current.get("sampling_protocol"):
            raise ValueError("基准与复测的采样方式不一致，不能比较不同口径的指标")
        if current.get("sampling_protocol") == "multidirectional":
            result = evaluate_multidirectional(
                current["samples"], parameters["hand_eye"],
                baseline_samples=baseline_batch["samples"],
                base_rotations=baseline_batch.get("base_rotations"),
                target_pose_base=parameters.get("target_pose_base"),
            )
        else:
            result = evaluate_position_monitoring(
                baseline_batch["samples"], current["samples"], parameters["hand_eye"],
                base_rotations=baseline_batch.get("base_rotations"),
                initial_errors=baseline_batch.get("initial_errors"),
            )
        result["warnings"] = list(dict.fromkeys(
            baseline_batch.get("warnings", []) + current.get("warnings", []) + result["warnings"]
        ))
        result.update({
            "baseline_id": baseline["id"], "parameter_version": version,
            "baseline_label": baseline["label"],
            "baseline_created_at": baseline.get("created_at"),
            "orientation_source": baseline_batch.get("orientation_source", "unspecified"),
            "baseline_source": baseline_batch["source_path"],
            "baseline_path": baseline["path"],
        })
        if current.get("sampling_protocol") != "multidirectional" and not baseline_batch.get("initial_errors"):
            result["warnings"].append("所选基准没有初始绝对误差，绝对定位精度退化不可计算")
        return self._complete_result(result, current, baseline_batch)

    def evaluate_current(self, progress=None):
        """无需基准计算当前重复定位；不生成虚假的零退化量。"""
        current = self.current_batch
        if current is None:
            raise ValueError("请先导入本次观测")
        if progress:
            progress(0, "计算本次重复定位指标")
        result = self._current_repeatability(current, self.parameters)
        return self._save_evaluation(result, current, progress)

    def _current_repeatability(self, current, parameters):
        if parameters.get("hand_eye") is None:
            raise ValueError("评估前请加载相机到末端的手眼参数")
        if current["parameter_version"] != parameters["version"]:
            raise ValueError("参数版本已变化，请重新导入本次观测")
        if current.get("comparison_status") == "calibration_only":
            raise ValueError("手眼标定数据不能直接作为重复定位观测")
        if current.get("sampling_protocol") == "multidirectional":
            result = evaluate_multidirectional(
                current["samples"], parameters["hand_eye"],
                base_rotations=current.get("base_rotations"),
                target_pose_base=parameters.get("target_pose_base"),
            )
        else:
            result = evaluate_current_repeatability(
                current["samples"], parameters["hand_eye"],
                base_rotations=current.get("base_rotations"),
            )
        result["warnings"] = list(dict.fromkeys(current.get("warnings", []) + result["warnings"]))
        result.update({
            "baseline_id": None, "baseline_label": None, "baseline_path": None,
            "baseline_created_at": None,
            "baseline_source": None, "parameter_version": parameters["version"],
            "orientation_source": current.get("orientation_source", "unspecified"),
        })
        return self._complete_result(result, current, None)

    def _complete_result(self, result, current, baseline_batch):
        batches = [current] + ([baseline_batch] if baseline_batch is not None else [])
        comparison_status = (
            "debug_unverified" if any(batch.get("comparison_status") == "debug_unverified"
                                      for batch in batches)
            else current.get("comparison_status", "unspecified")
        )
        if comparison_status == "debug_unverified":
            result["warnings"].append("此数据仅供调试，不能将比较结果认定为真实精度退化")
        result.update({
            "batch_id": current["batch_id"], "batch_label": current["label"],
            "program_id": current["program_id"], "target_id": current["target_id"],
            "current_source": current["source_path"], "comparison_status": comparison_status,
            "observed_at": current["observed_at"], "time_source": current["time_source"],
            "debug_day_index": current.get("debug_day_index"),
            "baseline_observed_at": baseline_batch.get("observed_at") if baseline_batch else None,
            "baseline_time_source": baseline_batch.get("time_source") if baseline_batch else None,
            "baseline_debug_day_index": baseline_batch.get("debug_day_index") if baseline_batch else None,
            "role": "baseline" if baseline_batch and current.get("saved_path") == baseline_batch.get("saved_path") else "measurement",
            "current_batch_path": current["saved_path"],
        })
        return self._apply_thresholds(result)

    def _apply_thresholds(self, result):
        """仅重判当前阈值，不重新计算位姿统计。"""
        result["metric_thresholds"] = deepcopy(self.settings[
            "multidirectional_thresholds" if result.get("sampling_protocol") == "multidirectional" else "metric_thresholds"])
        assessments = {mode: assess_metric(result, mode) for mode in METRIC_LABELS}
        result["metric_assessments"] = assessments
        result["alarms"] = [alarm for item in assessments.values() for alarm in item["alarms"]]
        considered = list(assessments.values()) if result.get("baseline_id") else [assessments["repeatability"]]
        statuses = {item["status"] for item in considered}
        if result.get("comparison_status") == "debug_unverified":
            status = "调试比较 · 采样对应待确认"
        elif result["alarms"]:
            status = "超限"
        elif len(statuses) == 1:
            status = next(iter(statuses))
        elif all(all(value is None for value in item["thresholds"].values()) for item in considered):
            status = "未设置阈值"
        else:
            status = "部分指标不可判定"
        result["status"] = status
        return result

    def _save_evaluation(self, result, current, progress):
        if progress:
            progress(65, "保存评估结果")
        result.update({"id": _stamp(), "created_at": datetime.now(timezone.utc).isoformat()})
        # 复用已保存观测；同日的多次评估集中保存，不复制图像与观测大文件。
        daily_path = self._store.record("evaluations", result, current, self.parameters)
        self._history_revision += 1
        if _newer_observation(result, self._latest_evaluation):
            self.latest_batch = deepcopy(current)
            self._latest_evaluation = deepcopy(result)
            self._latest_result_ref = {"path": str(daily_path), "id": result["id"]}
        self.compare_latest((lambda value, message: progress(80 + round(value * 0.19), message)) if progress else None)
        self._save_state()
        if progress:
            progress(100, "评估完成，已保存结果和观测快照")
        return result

    def _batch_parameters(self, batch):
        candidates = [batch.get("parameters"), self.parameters,
                      self.baseline.get("parameters") if self.baseline else None]
        for parameters in candidates:
            if parameters and parameters.get("version") == batch["parameter_version"]:
                return parameters
        raise ValueError("已保存观测缺少对应的参数快照，无法重新比较")

    @staticmethod
    def _comparison_metadata(result, archived):
        # 保留原评估身份和时间，当前视图不是一次新评估。
        result.update({key: archived[key] for key in ("id", "created_at")})
        day, baseline_day = result.get("debug_day_index"), result.get("baseline_debug_day_index")
        if day is not None and baseline_day is not None:
            elapsed = day - baseline_day
        elif day is None and baseline_day is None and result.get("baseline_observed_at"):
            elapsed = (parse_timestamp(result["observed_at"])
                       - parse_timestamp(result["baseline_observed_at"])).total_seconds() / 86400
        else:
            elapsed = None
        result["comparison_days"] = elapsed
        result["before_baseline"] = elapsed is not None and elapsed < 0
        return result

    def compare_latest(self, progress=None):
        """最新已评估观测相对当前基准的即时视图；不追加或改写任何历史。"""
        if progress:
            progress(0, "比较最新已评估观测")
        key = (self.baseline["id"] if self.baseline else None,
               self._latest_evaluation["id"] if self._latest_evaluation else None)
        if key == self._latest_comparison_key and self.latest_result is not None:
            self._apply_thresholds(self.latest_result)
            if progress:
                progress(100, "已复用最新观测统计并更新阈值判定")
            return deepcopy(self.latest_result)
        self.latest_result = None
        self.latest_comparison_error = ""
        if self.latest_batch is None:
            if progress:
                progress(100, "尚无已评估观测")
            return None
        try:
            parameters = self._batch_parameters(self.latest_batch)
        except ValueError as error:
            self.latest_comparison_error = str(error)
            if progress:
                progress(100, str(error))
            return None
        try:
            result = self._compare_batches(self.latest_batch, self.baseline, parameters)
        except ValueError as error:
            self.latest_comparison_error = str(error)
            # 当前散布仍由该观测自己的参数计算，不能换成更早的可比观测。
            result = self._current_repeatability(self.latest_batch, parameters)
            result["comparison_error"] = str(error)
            result["warnings"].append(str(error))
        self.latest_result = self._comparison_metadata(result, self._latest_evaluation)
        self._latest_comparison_key = key
        if progress:
            progress(100, "最新观测比较完成")
        return deepcopy(self.latest_result)

    def history_comparisons(self, include_before=False, progress=None):
        """每次已评估观测只取一次，用当前基准和阈值重算曲线，档案保持原样。"""
        self.history_comparison_warnings = []
        if self.latest_batch is None:
            if progress:
                progress(100, "尚无已评估历史")
            return []
        key = (self.baseline["id"] if self.baseline else None,
               self._latest_evaluation["id"], self._history_revision)
        if self._history_cache and self._history_cache["key"] == key:
            self.history_comparison_warnings = list(self._history_cache["warnings"])
            results = [self._apply_thresholds(deepcopy(result)) for result in self._history_cache["results"]
                       if include_before or not result["before_baseline"]]
            if progress:
                progress(100, "已复用历史统计并更新阈值判定")
            return results
        if progress:
            progress(0, "读取已保存评估历史")
        records = {record.get("current_batch_path"): record for record in self.list_history()
                   if record.get("current_batch_path")}
        comparisons = []
        for index, record in enumerate(records.values()):
            try:
                if progress:
                    progress(10 + round(90 * index / len(records)), f"读取批次 {record['batch_id']}（{index + 1}/{len(records)}）")
                batch = self._read_evaluated_batch(record)
                if progress:
                    progress(10 + round(90 * (index + 0.5) / len(records)), f"比较批次 {record['batch_id']}（{index + 1}/{len(records)}）")
                if self.baseline is None:
                    for field in ("parameter_version", "program_id", "target_id", "sampling_protocol"):
                        if batch.get(field) != self.latest_batch.get(field):
                            raise ValueError(f"{field} 与最新观测不一致，不能合并趋势")
                    groups = [{(sample["point_id"], sample["direction_id"]) for sample in item["samples"]}
                              for item in (batch, self.latest_batch)]
                    if groups[0] != groups[1]:
                        raise ValueError("测点/接近方向与最新观测不匹配，不能合并趋势")
                    if batch.get("sampling_protocol") == "multidirectional":
                        ideals = [{sample["point_id"]: sample["ideal_pose"] for sample in item["samples"]
                                   if sample.get("ideal_pose") is not None} for item in (batch, self.latest_batch)]
                        for point in ideals[0].keys() & ideals[1].keys():
                            if not np.allclose(ideals[0][point], ideals[1][point], atol=1e-6, rtol=0):
                                raise ValueError(f"测点 {point} 理想位姿与最新观测不一致，不能合并趋势")
                result = self._compare_batches(batch, self.baseline, self._batch_parameters(batch))
                self._comparison_metadata(result, record)
            except ValueError as error:
                self.history_comparison_warnings.append(f"{record['batch_id']}：{error}")
            else:
                comparisons.append(result)
            if progress:
                progress(10 + round(90 * (index + 1) / len(records)), f"已处理批次 {index + 1}/{len(records)}")
        comparisons.sort(key=lambda item: (
            item["debug_day_index"] if item["debug_day_index"] is not None else 0,
            parse_timestamp(item["observed_at"])))
        self._history_cache = {"key": key, "results": comparisons, "warnings": list(self.history_comparison_warnings)}
        if progress:
            progress(100, "历史比较完成")
        return deepcopy([result for result in comparisons if include_before or not result["before_baseline"]])

    def list_history(self, baseline_id=None):
        records = self._store.history()
        return [record for record in records if baseline_id is None or record["baseline_id"] == baseline_id]

    @staticmethod
    def export_result(path, result):
        write_document(path, result)
