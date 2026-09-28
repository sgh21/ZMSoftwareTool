"""机器人定位业务：参数版本、观测导入、基准、评估和历史；不依赖 Qt。"""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from core.algorithms.board_pose import calibrate_hand_eye, estimate_board_pose
from core.algorithms.position_monitoring import (
    evaluate_position_monitoring,
    validate_transform,
)


def read_document(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    document = yaml.safe_load(text) if path.suffix.lower() in (".yaml", ".yml") else json.loads(text)
    if not isinstance(document, dict):
        raise ValueError("输入文件必须是 JSON/YAML 对象")
    return document


def write_document(path, document):
    """仅写本模块的输出；原始图像和输入文件不作任何修改。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(path)


def _stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _rotation(value, name):
    transform = np.eye(4)
    rotation = np.asarray(value, dtype=float)
    if rotation.shape != (3, 3):
        raise ValueError(f"{name} 必须为 3×3 矩阵")
    transform[:3, :3] = rotation
    return validate_transform(transform, name)[:3, :3].tolist()


class PositionMonitoringService:
    def __init__(self, root=None):
        self.root = Path(root or Path(__file__).resolve().parents[2]).resolve()
        self.storage = self.root / "storage" / "position_monitoring"
        defaults = Path(__file__).resolve().parents[2] / "config" / "robot_position.json"
        configured = self.root / "config" / "robot_position.json"
        self.parameters = read_document(configured if configured.exists() else defaults)
        self.settings = {
            "thresholds": dict.fromkeys(("X", "Y", "Z", "distance")),
            "processing_points": [],
        }
        self.baseline = None
        self.current_batch = None
        state_path = self.storage / "state.json"
        if state_path.exists():
            state = read_document(state_path)
            self.parameters = state["parameters"]
            self.settings = state["settings"]
            if state.get("baseline_path"):
                self.baseline = read_document(state["baseline_path"])
                self.baseline["path"] = state["baseline_path"]

    def _save_state(self):
        write_document(self.storage / "state.json", {
            "parameters": self.parameters,
            "settings": self.settings,
            "baseline_path": self.baseline["path"] if self.baseline else None,
        })

    def load_parameters(self, path):
        document = read_document(path)
        # 旧 master 的手眼输出 T_tool_cam 明确使用 m；只在此导入边界换算。
        if "T_tool_cam" in document:
            legacy = validate_transform(document["T_tool_cam"], "旧手眼参数").copy()
            legacy[:3, 3] *= 1000
            document = {**self.parameters, "hand_eye": legacy.tolist(), "length_unit": "mm"}
            document["source_note"] = "旧 master T_tool_cam 导入，平移由 m 转为 mm"
        document["source_path"] = str(Path(path).resolve())
        return self.save_parameters(document)

    def save_parameters(self, document):
        parameters = deepcopy(self.parameters)
        parameters.update(deepcopy(document))
        if parameters.get("length_unit", "mm") not in ("mm", "m"):
            raise ValueError("参数长度单位只支持 mm 或 m")
        if parameters.get("transform_convention", "E_T_C") != "E_T_C":
            raise ValueError("手眼变换必须声明为 E_T_C（相机到被监测末端）")
        if parameters.get("hand_eye") is not None:
            hand_eye = validate_transform(parameters["hand_eye"], "手眼参数").copy()
            if "hand_eye" in document and document.get("length_unit") == "m":
                hand_eye[:3, 3] *= 1000
            parameters["hand_eye"] = hand_eye.tolist()
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
        if parameters.get("reference_rotation") is not None:
            parameters["reference_rotation"] = _rotation(parameters["reference_rotation"], "棋盘参考旋转")
        limit = parameters.get("max_reprojection_error_px")
        if limit is not None and (not np.isfinite(float(limit)) or float(limit) <= 0):
            raise ValueError("重投影误差限值必须为正数或 null")
        parameters["length_unit"] = "mm"
        parameters["transform_convention"] = "E_T_C"
        parameters["version"] = _stamp()
        write_document(self.storage / "parameters" / f"{parameters['version']}.json", parameters)
        self.parameters = parameters
        self.current_batch = None
        self._save_state()
        return deepcopy(parameters)

    def save_settings(self, settings):
        updated = deepcopy(self.settings)
        updated.update(deepcopy(settings))
        thresholds = updated["thresholds"]
        for axis in ("X", "Y", "Z", "distance"):
            value = thresholds.get(axis)
            if value is not None:
                value = float(value)
                if not np.isfinite(value) or value < 0:
                    raise ValueError(f"{axis} 阈值必须为非负数或留空")
            thresholds[axis] = value
        self.settings = updated
        self._save_state()

    def load_observations(self, path, progress=None):
        """读取标准观测清单；图像在后台逐幅 PnP，矩阵输入无须相机内参。"""
        self.current_batch = None
        source = Path(path).resolve()
        batch = read_document(source)
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
                pose = validate_transform(sample["vision_pose"], "视觉位姿").copy()
                pose[:3, 3] *= factor
                sample["vision_pose"] = pose.tolist()
            else:
                if not image_path or not image_path.is_file():
                    raise ValueError(f"样本 {identity} 缺少视觉位姿或有效图像")
                if self.parameters.get("camera_matrix") is None:
                    raise ValueError("从图像解算位姿前请设置相机内参")
                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
                if image is None:
                    raise ValueError(f"无法读取图像 {image_path}")
                reference = references.get(sample["point_id"])
                if reference is None:
                    reference = self.parameters.get("reference_rotation")
                try:
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
                        progress(round(100 * (index + 1) / len(samples)), f"跳过 {image_path.name}: {error}")
                    continue
                sample.update({key: value.tolist() if isinstance(value, np.ndarray) else value
                               for key, value in estimate.items()
                               if key not in ("image_points", "projected_points")})
            references.setdefault(sample["point_id"], np.asarray(sample["vision_pose"])[:3, :3])
            if "robot_pose" in sample:
                robot = validate_transform(sample["robot_pose"], "机器人记录位姿").copy()
                robot[:3, 3] *= factor
                sample["robot_pose"] = robot.tolist()
            resolved.append(sample)
            if progress:
                progress(round(100 * (index + 1) / len(samples)), f"已读取 {index + 1}/{len(samples)}")
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
            "schema_version": 1, "batch_id": str(batch.get("batch_id") or _stamp()),
            "label": batch.get("label") or source.stem, "samples": resolved,
            "length_unit": "mm", "source_path": str(source),
            "parameter_version": self.parameters["version"], "rejected_samples": rejected,
        })
        batch.setdefault("warnings", [])
        if batch.get("orientation_source") == "controller_approximation":
            batch["warnings"].append("基座方向来自控制器记录，属于近似朝向，非独立实测姿态")
        self.current_batch = batch
        return deepcopy(batch)

    def create_baseline(self, name=""):
        if self.current_batch is None:
            raise ValueError("请先导入初始观测")
        if self.parameters.get("hand_eye") is None:
            raise ValueError("建立基准前请加载相机到末端的手眼参数")
        if self.current_batch.get("comparison_status") == "calibration_only":
            raise ValueError("手眼标定数据包含不同目标位姿，不能直接建立监控基准")
        if self.current_batch["parameter_version"] != self.parameters["version"]:
            raise ValueError("观测与当前参数版本不一致，请重新导入")
        identifier = _stamp()
        path = self.storage / "baselines" / f"{identifier}.json"
        baseline = {
            "id": identifier, "label": name or self.current_batch["label"],
            "created_at": datetime.now(timezone.utc).isoformat(), "path": str(path),
            "parameters": deepcopy(self.parameters), "batch": deepcopy(self.current_batch),
        }
        write_document(path, baseline)
        self.baseline = baseline
        self.current_batch = None
        self._save_state()
        return deepcopy(baseline)

    def list_baselines(self):
        return [
            {key: document[key] for key in ("id", "label", "path", "created_at")}
            for path in sorted((self.storage / "baselines").glob("*.json"))
            for document in [read_document(path)]
        ]

    def select_baseline(self, path):
        baseline = read_document(path)
        if not all(key in baseline for key in ("id", "batch", "parameters")):
            raise ValueError("所选文件不是定位监控基准")
        baseline["path"] = str(Path(path).resolve())
        self.baseline = baseline
        self.parameters = deepcopy(baseline["parameters"])
        self.current_batch = None
        self._save_state()
        return deepcopy(baseline)

    def calibrate_hand_eye(self, method="PARK", progress=None):
        if self.current_batch is None:
            raise ValueError("请先导入手眼标定观测")
        samples = self.current_batch["samples"]
        if any("robot_pose" not in item for item in samples):
            raise ValueError("手眼标定的每次观测都需要对应机器人 B_T_E 位姿")
        result = calibrate_hand_eye(
            [item["robot_pose"] for item in samples],
            [item["vision_pose"] for item in samples], method=method,
        )
        result = {key: value.tolist() if isinstance(value, np.ndarray) else value
                  for key, value in result.items()}
        validate_transform(result["hand_eye"], "标定得到的手眼变换")
        result["batch_id"] = self.current_batch["batch_id"]
        result["source_path"] = self.current_batch["source_path"]
        result["parameters"] = deepcopy(self.parameters)
        result["rejected_samples"] = deepcopy(self.current_batch.get("rejected_samples", []))
        result["pnp_summary"] = [{
            key: sample.get(key) for key in (
                "point_id", "direction_id", "sample_id", "image_path",
                "reprojection_error_px", "reprojection_mean_px", "corner_count", "corner_order",
            )
        } for sample in samples if sample.get("image_path")] or None
        identifier = _stamp()
        path = self.storage / "calibrations" / f"{identifier}.json"
        snapshot_path = self.storage / "calibrations" / f"{identifier}_observations.json"
        result["input_batch_path"] = str(snapshot_path)
        write_document(snapshot_path, self.current_batch)
        write_document(path, result)
        result["path"] = str(path)
        # 返回待审阅结果，不自动替换当前手眼及已建立基准。
        if progress:
            progress(100, "手眼标定完成，请查看固定靶标一致性残差后应用参数")
        return result

    def evaluate(self, progress=None):
        if self.baseline is None or self.current_batch is None:
            raise ValueError("评估需要已保存的基准和本次复测观测")
        baseline_batch = self.baseline["batch"]
        current = self.current_batch
        version = self.baseline["parameters"]["version"]
        if version != self.parameters["version"] or current["parameter_version"] != version:
            raise ValueError("参数版本已变化，请选择原基准参数并重新导入，或建立新基准")
        if current.get("comparison_status") == "calibration_only":
            raise ValueError("手眼标定数据不能直接作为复测数据")
        for field in ("program_id", "target_id"):
            if baseline_batch[field] != current[field]:
                raise ValueError(f"基准与复测的 {field} 不一致，不能直接比较")
        result = evaluate_position_monitoring(
            baseline_batch["samples"], current["samples"], self.parameters["hand_eye"],
            base_rotations=baseline_batch.get("base_rotations"),
            initial_errors=baseline_batch.get("initial_errors"),
        )
        result["warnings"] = list(dict.fromkeys(
            baseline_batch.get("warnings", []) + current.get("warnings", []) + result["warnings"]
        ))
        alarms = []
        missing_axes = False
        thresholds = self.settings["thresholds"]
        for group in result["groups"]:
            axes = dict(zip(("X", "Y", "Z"), group["drift_base"] or [None] * 3))
            axes["distance"] = group["drift_distance"]
            for axis, value in axes.items():
                threshold = thresholds[axis]
                if threshold is None:
                    continue
                if value is None:
                    missing_axes = True
                elif abs(value) > threshold:
                    alarms.append({
                        "point_id": group["point_id"], "direction_id": group["direction_id"],
                        "axis": axis, "value": value, "threshold": threshold,
                    })
        status = "超限" if alarms else "阈值内"
        if all(value is None for value in thresholds.values()):
            status = "未设置阈值"
        elif missing_axes and not alarms:
            status = "部分指标不可判定"
        if any(batch.get("comparison_status") == "debug_unverified"
               for batch in (baseline_batch, current)):
            status = "调试比较 · 采样对应待确认"
            result["warnings"].append("此数据仅供调试，不能将比较结果认定为真实精度退化")
        result.update({
            "id": _stamp(), "created_at": datetime.now(timezone.utc).isoformat(),
            "baseline_id": self.baseline["id"], "parameter_version": version,
            "baseline_label": self.baseline["label"],
            "batch_id": current["batch_id"], "batch_label": current["label"],
            "status": status, "alarms": alarms, "thresholds": deepcopy(thresholds),
            "orientation_source": baseline_batch.get("orientation_source", "unspecified"),
            "current_source": current["source_path"],
            "baseline_source": baseline_batch["source_path"],
            "baseline_path": self.baseline["path"],
        })
        # 保存实际参与计算的观测快照，历史无需重跑图像或依赖被改写的输入清单。
        snapshot_path = self.storage / "observations" / f"{result['id']}.json"
        result["current_batch_path"] = str(snapshot_path)
        write_document(snapshot_path, current)
        write_document(self.storage / "history" / f"{result['id']}.json", result)
        if progress:
            progress(100, "评估完成，已保存结果和观测快照")
        return result

    def list_history(self, baseline_id=None):
        records = [read_document(path) for path in sorted((self.storage / "history").glob("*.json"))]
        return [record for record in records if baseline_id is None or record["baseline_id"] == baseline_id]

    @staticmethod
    def export_result(path, result):
        write_document(path, result)
