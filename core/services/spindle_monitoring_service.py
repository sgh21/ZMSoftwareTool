"""主轴 ZIP 导入、人工标签、从头训练与分版本评价记录。"""

from copy import deepcopy
import csv
from datetime import datetime, timezone
from io import TextIOWrapper
import json
import math
from pathlib import Path, PurePosixPath
import random
import shutil
from tempfile import TemporaryDirectory
from uuid import uuid4
from zipfile import ZipFile

from core.algorithms import spindle_monitoring as algorithm
from core.services.position_persistence import read_json, write_document


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _now():
    return datetime.now(timezone.utc).isoformat()


def _relative_member(name):
    """ZIP 只接受包内 POSIX 相对路径，拒绝 Windows 盘符和路径穿越。"""
    path = PurePosixPath(name)
    if not name or "\\" in name or ":" in name or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"ZIP 中存在不安全的路径：{name}")
    if not path.parts:
        raise ValueError("ZIP 成员路径不能为空")
    return path


def assess_score(score, thresholds):
    """相容度越低越可疑；报警阈值独立于统计检验的 alpha。"""
    warning, fault = thresholds.get("warning"), thresholds.get("fault")
    if score is None or not math.isfinite(score) or not 0 <= score <= 1:
        status, message = "unavailable", "正常相容度无效或尚未校准"
    elif fault is not None and score <= fault:
        status, message = "fault", "故障 / 建议检修"
    elif warning is not None and score <= warning:
        status, message = "warning", "预警 / 建议复测"
    elif warning is None and fault is None:
        status = "unconfigured"
        message = "偏离正常参考 · 未设置报警阈值" if score < 1 else "未设置阈值"
    elif score < 1:
        status, message = "deviation", "偏离正常参考，未达报警阈值"
    else:
        status, message = "normal", "阈值内"
    return {"status": status, "message": message}


def assess_compatibility(result, thresholds, review_required=False):
    scores = [result.get("analysis_score"), result.get("score")]
    assessments = [assess_score(score, thresholds.get(feature, thresholds))
                   for feature, score in zip(("vibration", "network"), scores)]
    invalid = any(item["status"] == "unavailable" for item in assessments)
    reasons = [item.get("message", "正常参考无效") for item in result.get("compatibility", {}).values()
               if item.get("status") == "invalid"]
    invalid_message = "；".join(dict.fromkeys(reasons)) or "正常相容度无效或尚未校准"
    if not review_required:
        for status in ("fault", "warning"):
            for item in assessments:
                if item["status"] == status:
                    return {**item, "message": item["message"] + (f"；另一项无效：{invalid_message}" if invalid else "")}
    if invalid:
        return {"status": "unavailable", "message": invalid_message}
    if review_required:
        return {"status": "review_required", "message": "新模型阈值待复核"}
    for status in ("deviation", "unconfigured", "normal"):
        for _, item in sorted(zip(scores, assessments), key=lambda pair: pair[0]):
            if item["status"] == status:
                return item


def result_metric_settings(result):
    """旧的共用参数等值展开；历史快照不采用当前参数。"""
    if "metric_settings" in result:
        return deepcopy(result["metric_settings"])
    common = {"alpha": result.get("alpha", 0.05), **result.get("thresholds", {"warning": None, "fault": None})}
    return {feature: deepcopy(common) for feature in ("vibration", "network")}


def _validated_metric_settings(metrics):
    settings = {}
    for feature, title in (("vibration", "振动"), ("network", "网络")):
        values = metrics[feature]
        alpha = float(values["alpha"])
        if not math.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError(f"{title} alpha 必须在 0 与 1 之间")
        thresholds = {key: None if values[key] is None else float(values[key]) for key in ("warning", "fault")}
        if any(value is not None and (not math.isfinite(value) or not 0 <= value < 1) for value in thresholds.values()):
            raise ValueError(f"{title}相容度报警阈值必须在 0（含）与 1（不含）之间")
        if thresholds["warning"] is not None and thresholds["fault"] is not None and thresholds["fault"] >= thresholds["warning"]:
            raise ValueError(f"{title}故障阈值必须低于预警阈值")
        settings[feature] = {"alpha": alpha, **thresholds}
    return settings


class SpindleMonitoringService:
    def __init__(self, root=None, config=None):
        if config is None:
            config = read_json(PROJECT_ROOT / "config" / "spindle_monitoring.json")
        elif isinstance(config, (str, Path)):
            config = read_json(config)
        self.config = deepcopy(config)
        self.root = Path(root) if root is not None else PROJECT_ROOT / self.config["storage_root"]
        self.root = self.root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.state = read_json(self.state_path) if self.state_path.exists() else {
            "schema_version": 1,
            "settings": {"thresholds": deepcopy(self.config["thresholds"]),
                         "score_kind": "normal_compatibility_v1", "alpha": self.config.get("alpha", 0.05),
                         "thresholds_model_version": None},
            "runs": {}, "packages": [], "models": [], "current_model_version": None,
            "results": [],
        }
        # 旧倍率阈值保留备查，但不能解释为 0–1 相容度阈值。
        if self.settings.get("score_kind") != "normal_compatibility_v1":
            self.settings["legacy_thresholds"] = deepcopy(self.settings["thresholds"])
            self.settings.update(thresholds={"warning": None, "fault": None},
                                 thresholds_model_version=None, score_kind="normal_compatibility_v1",
                                 alpha=self.config.get("alpha", 0.05))
        self.settings.setdefault("metric_settings", result_metric_settings(self.settings))
        # 旧版初始数据曾允许未判定入训；恢复状态时统一收紧到正常标签。
        eligibility_changed = False
        for run in self.runs.values():
            if run["manual_label"] != "healthy" and run["training_eligible"]:
                run["training_eligible"] = False
                eligibility_changed = True
        if eligibility_changed:
            self._save()

    @property
    def runs(self):
        return self.state["runs"]

    @property
    def settings(self):
        return self.state["settings"]

    @property
    def metric_settings(self):
        return result_metric_settings(self.settings)

    result_metric_settings = staticmethod(result_metric_settings)

    @property
    def models(self):
        return self.state["models"]

    @property
    def current_model(self):
        version = self.state["current_model_version"]
        return next((model for model in self.models if model["version"] == version), None)

    @property
    def thresholds_review_required(self):
        model = self.current_model
        return bool(model and any(values[key] is not None for values in self.metric_settings.values()
                                  for key in ("warning", "fault"))
                    and self.settings.get("thresholds_model_version") != model["version"])

    def _save(self):
        write_document(self.state_path, self.state)

    def list_runs(self):
        return sorted(self.runs.values(), key=lambda run: (run["captured_at"], run["run_id"]))

    def training_candidates(self):
        return [run for run in self.list_runs() if run["training_eligible"]
                and run["manual_label"] == "healthy"]

    def _read_run(self, record, root=None):
        root = self.root if root is None else root
        telemetry = record.get("telemetry_file")
        data = algorithm.read_run(root / record["data_file"], root / telemetry if telemetry else None)
        if data["target_speed_rpm"] != record["speed_rpm"]:
            raise ValueError("H5 转速与数据包声明不一致")
        data["run_id"] = record["run_id"]
        data["calibration_context"] = {
            "speed_rpm": record["speed_rpm"], "operation": record["operation"],
            "condition": deepcopy(record.get("condition", {})),
            "preprocessing": deepcopy(record.get("preprocessing", self.config["preprocessing"])),
        }
        return data

    def order_band_energy(self, run_id):
        """旧评价的倍频图从托管信号重算，不改评价记录或网络评分。"""
        return algorithm.order_band_energy(self._read_run(self.runs[run_id]))

    def analysis_metrics(self, result):
        if not result or result.get("score_kind") != "normal_compatibility_v1":
            return None, None
        return result.get("analysis_score"), result.get("compatibility", {}).get("vibration")

    def display_result(self, result, current_settings=False, metric_settings=None):
        """显示设置只作用于副本；历史评分、alpha 和当时判定不改写。"""
        if result is None:
            return None
        shown = deepcopy(result)
        if shown.get("score_kind") != "normal_compatibility_v1":
            shown.setdefault("legacy_assessment", deepcopy(shown.get("assessment")))
            shown.update(score=None, analysis_score=None,
                         assessment={"status": "legacy", "message": "旧倍率记录，请重新评估相容度"})
            return shown
        if current_settings or metric_settings is not None:
            metrics = self.metric_settings if metric_settings is None else _validated_metric_settings(metric_settings)
            review_required = self.thresholds_review_required if metric_settings is None else False
            shown["metric_settings"] = metrics
            shown["alpha"] = metrics["network"]["alpha"]
            for key, p_key, feature in (("analysis_score", "vibration_p_value", "vibration"),
                                        ("score", "network_p_value", "network")):
                alpha = metrics[feature]["alpha"]
                p = shown.get(p_key)
                valid = shown.get("compatibility", {}).get(feature, {}).get("status") == "valid"
                shown[key] = min(1.0, p / alpha) if valid and p is not None and math.isfinite(p) and 0 <= p <= 1 else None
                if feature in shown.get("compatibility", {}):
                    shown["compatibility"][feature]["score"] = shown[key]
                    shown["compatibility"][feature]["alpha"] = alpha
            shown["thresholds"] = {key: metrics["network"][key] for key in ("warning", "fault")}
            shown["thresholds_model_version"] = (self.settings.get("thresholds_model_version")
                                                  if metric_settings is None else self.state["current_model_version"])
            shown["thresholds_review_required"] = review_required
            shown["assessment"] = assess_compatibility(shown, metrics, review_required)
        return shown

    def _infer_package(self, archive):
        """无清单时按 H5 内容识别采集；同目录的遥测按 CSV 列名识别。"""
        import h5py

        names = [name for name in archive.namelist() if not archive.getinfo(name).is_dir()]
        runs = []
        for name in sorted(names):
            if PurePosixPath(name).suffix.lower() not in (".h5", ".hdf5"):
                continue
            with archive.open(name) as stream, h5py.File(stream, "r") as h5:
                metadata = {key: algorithm._attribute(value) for key, value in h5.attrs.items()}
            run_id = metadata.get("run_id") or metadata.get("run_name")
            if not run_id:
                raise ValueError(f"{name} 缺少采集编号 run_id / run_name")
            captured_at = metadata.get("captured_at") or metadata.get("run_date")
            if not captured_at:
                raise ValueError(f"{name} 缺少采集时间 captured_at / run_date")
            captured_at = datetime.fromisoformat(str(captured_at).replace("Z", "+00:00")).isoformat()
            if not metadata.get("captured_at"):
                captured_at = captured_at[:10]
            parent = PurePosixPath(name).parent
            telemetry = []
            for member in names:
                member_path = PurePosixPath(member)
                if member_path.parent != parent or member_path.suffix.lower() != ".csv":
                    continue
                with archive.open(member) as stream, TextIOWrapper(stream, encoding="utf-8-sig", newline="") as text:
                    fields = next(csv.reader(text), [])
                if {"time_s", "actual_speed_rpm", "current_a"}.issubset(fields):
                    telemetry.append(member)
            if len(telemetry) > 1:
                matching = [member for member in telemetry if PurePosixPath(member).stem == PurePosixPath(name).stem]
                if len(matching) != 1:
                    raise ValueError(f"{name} 同目录有多份遥测 CSV，无法确定对应关系")
                telemetry = matching
            source = next((member for member in names if PurePosixPath(member).parent == parent
                           and PurePosixPath(member).name in ("source_manifest.json", "metadata.json", "manifest.json")), None)
            runs.append({
                "run_id": str(run_id), "run_name": metadata.get("run_name", str(run_id)),
                "captured_at": captured_at,
                "captured_time_precision": "timestamp" if metadata.get("captured_at") else "date",
                "speed_rpm": metadata.get("target_speed_rpm"),
                "operation": metadata.get("operation", "idle"),
                "operation_source": "h5" if "operation" in metadata else "current_acquisition_plan",
                "data_file": name, "telemetry_file": telemetry[0] if telemetry else None,
                "source_metadata_file": source,
            })
        if not runs:
            raise ValueError("ZIP 内未找到 H5 样本数据")
        return {
            "schema_version": 1, "package_id": "h5_" + runs[0]["run_id"],
            "captured_date": runs[0]["captured_at"][:10], "source_type": "acquisition",
            "preprocessing": deepcopy(self.config["preprocessing"]), "runs": runs,
            "metadata_source": "h5",
        }

    def _validate_package(self, archive):
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("ZIP 存在重名成员，无法确定数据来源")
        for name in names:
            _relative_member(name)
        manifest = (json.loads(archive.read("manifest.json").decode("utf-8-sig"))
                    if "manifest.json" in names else self._infer_package(archive))
        if manifest.get("schema_version") != 1:
            raise ValueError("不支持的数据包版本")
        if not manifest.get("package_id") or not manifest.get("captured_date"):
            raise ValueError("数据包缺少 package_id 或 captured_date")
        if manifest.get("source_type") not in ("historical_replay", "acquisition"):
            raise ValueError("数据来源必须是 historical_replay 或 acquisition")
        preprocessing = manifest.get("preprocessing", {})
        for key, value in self.config["preprocessing"].items():
            if preprocessing.get(key) != value:
                raise ValueError(f"数据包预处理参数不匹配：{key}")
        runs = manifest.get("runs", [])
        if not runs or len({run["run_id"] for run in runs}) != len(runs):
            raise ValueError("数据包无 run，或 run_id 重复")
        for run in runs:
            if not run["run_id"] or not run.get("captured_at"):
                raise ValueError("run 缺少编号或采集时间")
            datetime.fromisoformat(run["captured_at"].replace("Z", "+00:00"))
            if run.get("speed_rpm") != self.config["target_speed_rpm"]:
                raise ValueError(f"{run['run_id']} 的转速不属于当前建模工况")
            if run.get("operation") != "idle":
                raise ValueError(f"{run['run_id']} 不是空转工况")
            for field in ("data_file", "telemetry_file", "source_metadata_file"):
                if field != "data_file" and not run.get(field):
                    continue
                member = str(_relative_member(run[field]))
                if member not in names or archive.getinfo(member).is_dir():
                    raise ValueError(f"数据包缺少文件：{member}")
        return manifest

    def import_batch(self, paths, progress=None, *, label="unconfirmed"):
        """递归找出最小数据包，复用单包导入；一包失败不阻断其余包。"""
        if self.current_model is not None:
            raise ValueError("模型已建立；新数据请通过日常导入，并人工确认是否参与重训")
        if label not in ("unconfirmed", "healthy", "abnormal"):
            raise ValueError("整批标签必须为未确认、健康或异常")
        before_batch = set(self.runs)
        report = {"runs": [], "packages": [], "new_count": 0, "duplicate_count": 0, "failed_count": 0}
        batch_id = uuid4().hex
        candidates = []

        def failed(source, error):
            report["failed_count"] += 1
            message = f"{source}：{error}"
            report["packages"].append({"source": source, "status": "failed", "message": message})
            return message

        with TemporaryDirectory(prefix="spindle_batch_", dir=self.root) as temporary:
            staging = Path(temporary)

            def discover(path, source):
                try:
                    with ZipFile(path) as archive:
                        members = [member for member in archive.infolist() if not member.is_dir()]
                        if any(PurePosixPath(member.filename).suffix.lower() in (".h5", ".hdf5") for member in members):
                            candidates.append((path, source))
                            if progress:
                                progress(0, f"检索：已找到 {len(candidates)} 个样本包")
                            return True
                        children = [member for member in members if member.filename.lower().endswith(".zip")]
                        if not children:
                            raise ValueError("未找到 H5 样本或内层 ZIP")
                        for member in children:
                            child_source = source + "!/" + member.filename
                            child = staging / (uuid4().hex + ".zip")
                            try:
                                _relative_member(member.filename)
                                with archive.open(member) as stream, child.open("wb") as output:
                                    shutil.copyfileobj(stream, output)
                                if not discover(child, child_source):
                                    child.unlink()
                            except Exception as error:
                                message = failed(child_source, error)
                                if progress:
                                    progress(0, message)
                except Exception as error:
                    message = failed(source, error)
                    if progress:
                        progress(0, message)

            if progress:
                progress(0, "检索 ZIP 内的样本包")
            for path in paths:
                discover(Path(path), Path(path).name)
            for index, (path, source) in enumerate(candidates):
                def package_progress(value, message):
                    if progress:
                        progress(round(99 * (index + value / 100) / len(candidates)),
                                 f"样本包 {index + 1}/{len(candidates)}：{message}")

                before = set(self.runs)
                try:
                    runs = self.import_packages([path], "initial", package_progress, source_name=source, batch_id=batch_id)
                except Exception as error:
                    added = set(self.runs) - before
                    report["new_count"] += len(added)
                    report["runs"].extend(self.runs[run_id] for run_id in sorted(added))
                    detail = f"已保存 {len(added)} 条，分析未完成；可重新导入补算。{error}" if added else str(error)
                    package_progress(100, failed(source, detail))
                else:
                    report["new_count"] += sum(run["run_id"] not in before for run in runs)
                    report["duplicate_count"] += sum(run["run_id"] in before for run in runs)
                    report["runs"].extend(runs)
                    report["packages"].append({"source": source, "status": "complete"})
                if path.parent == staging:
                    path.unlink()
            if label != "unconfirmed":
                self.set_labels(sorted(set(self.runs) - before_batch), label, "批量导入时整批判定")
            report["summary"] = (f"批量处理完成：新增 {report['new_count']} 条采集，"
                                 f"重复 {report['duplicate_count']} 条，失败 {report['failed_count']} 个包")
            if progress:
                progress(100, report["summary"])
        return report

    def import_packages(self, paths, purpose="daily", progress=None, *, source_name=None, batch_id=None):
        if purpose not in ("initial", "daily"):
            raise ValueError("导入用途必须为 initial 或 daily")
        if purpose == "initial" and self.current_model is not None:
            raise ValueError("模型已建立；新数据请通过日常导入，并人工确认是否参与重训")
        paths = [Path(path) for path in paths]
        batch_id = (batch_id or uuid4().hex) if purpose == "initial" else None
        imported = []
        for package_index, path in enumerate(paths):
            if progress:
                progress(round(100 * package_index / len(paths)), f"导入 {PurePosixPath(source_name or path.name).name}")
            with ZipFile(path) as archive:
                manifest = self._validate_package(archive)
                existing = [self.runs[run["run_id"]] for run in manifest["runs"]
                            if run["run_id"] in self.runs]
                for entry in manifest["runs"]:
                    previous = self.runs.get(entry["run_id"])
                    if previous and any(previous[key] != entry[key] for key in
                                        ("captured_at", "speed_rpm", "operation", "condition", "experiment_condition")
                                        if key in previous and key in entry):
                        raise ValueError(f"采集编号 {entry['run_id']} 已存在，但时间或工况不同")
                new_runs = [run for run in manifest["runs"] if run["run_id"] not in self.runs]
                records = []
                if new_runs:
                    package_folder = Path("packages") / uuid4().hex
                    self.root.joinpath("packages").mkdir(exist_ok=True)
                    # 先完整读取包内数据，成功后才托管和登记，坏包不会留下半条 run。
                    with TemporaryDirectory(prefix="spindle_import_", dir=self.root) as temporary:
                        staging = Path(temporary)
                        write_document(staging / "manifest.json", manifest)
                        for entry in new_runs:
                            for field in ("data_file", "telemetry_file", "source_metadata_file"):
                                if not entry.get(field):
                                    continue
                                member = str(_relative_member(entry[field]))
                                destination = staging / member
                                destination.parent.mkdir(parents=True, exist_ok=True)
                                with archive.open(member) as source, destination.open("wb") as output:
                                    shutil.copyfileobj(source, output)
                            self._read_run(entry, staging)
                        shutil.move(str(staging), str(self.root / package_folder))
                    now = _now()
                    for entry in new_runs:
                        record = deepcopy(entry)
                        record.update({
                            "run_name": entry.get("run_name", entry["run_id"]),
                            "package_id": manifest["package_id"], "captured_date": entry["captured_at"][:10],
                            "source_filename": source_name or path.name,
                            "import_batch_id": batch_id,
                            "source_type": manifest["source_type"], "imported_at": now,
                            "purpose": purpose, "manual_label": "unconfirmed", "label_note": "",
                            "label_updated_at": None, "label_history": [],
                            "training_eligible": False,
                            "preprocessing": deepcopy(manifest["preprocessing"]),
                        })
                        for field in ("data_file", "telemetry_file", "source_metadata_file"):
                            record[field] = (package_folder / entry[field]).as_posix() if entry.get(field) else None
                        self.runs[record["run_id"]] = record
                        records.append(record)
                    self.state["packages"].append({
                        "package_id": manifest["package_id"], "source_filename": source_name or path.name,
                        "imported_at": now, "purpose": purpose,
                        "manifest_file": (package_folder / "manifest.json").as_posix(),
                        "run_ids": [record["run_id"] for record in records],
                    })
                    self._save()
            package_records = existing + records
            imported.extend(package_records)
            pending = [record for record in package_records if self.latest_result(record["run_id"]) is None]
            for index, record in enumerate(pending):
                if progress:
                    progress(round(100 * (package_index + index / len(pending)) / len(paths)),
                             f"分析 {record['run_name']}")
                self.evaluate(record["run_id"], reason="import")
        if progress:
            progress(100, "导入与分析完成")
        return imported

    def import_group(self, run_id):
        """初始建模按导入批次分组；旧递归导入记录沿用外层包来源。"""
        selected = self.runs[run_id]
        if selected["purpose"] != "initial":
            return [run_id]
        batch_id = selected.get("import_batch_id")
        source = selected.get("source_filename", "").split("!/")[0]
        if not batch_id and not source:
            return [run_id]
        return [run["run_id"] for run in self.list_runs() if run["purpose"] == "initial" and
                (run.get("import_batch_id") == batch_id if batch_id else
                 not run.get("import_batch_id") and run.get("source_filename", "").split("!/")[0] == source)]

    def set_label(self, run_id, label, note="", include_in_training=True):
        self.set_labels([run_id], label, note, include_in_training)
        return self.runs[run_id]

    def trained_run_ids(self):
        """所有历史模型的训练和健康校准采集都锁定人工判定。"""
        return {run_id for model in self.models
                for key in ("training_run_ids", "calibration_run_ids")
                for run_id in model.get(key, [])}

    def set_labels(self, run_ids, label, note="", include_in_training=True):
        """整批标签一次保存；写入失败时一并恢复，避免只改一部分。"""
        if label not in ("unconfirmed", "healthy", "abnormal"):
            raise ValueError("人工标签必须为未确认、健康或异常")
        selected = {}
        for run_id in run_ids:
            if run_id not in selected:
                selected.update(dict.fromkeys(self.import_group(run_id)))
        if selected.keys() & self.trained_run_ids():
            raise ValueError("已参与网络训练或健康校准的样本不能修改人工判定。")
        runs = [self.runs[run_id] for run_id in selected]
        if not runs:
            return []
        previous = deepcopy(runs)
        update = {"manual_label": label, "label_note": str(note),
                  "training_eligible": bool(include_in_training and label == "healthy"),
                  "label_updated_at": _now()}
        for run in runs:
            run.update(update)
            run["label_history"].append(deepcopy(update))
        try:
            self._save()
        except Exception:
            for run, original in zip(runs, previous):
                run.clear()
                run.update(original)
            raise
        return runs

    def set_thresholds(self, warning=None, fault=None, alpha=None):
        """旧调用入口沿用共用参数，新界面分别设置两项。"""
        values = {"alpha": self.settings["alpha"] if alpha is None else alpha,
                  "warning": warning, "fault": fault}
        self.set_metric_settings({feature: values for feature in ("vibration", "network")})
        return self.settings["thresholds"]

    def set_metric_settings(self, metrics):
        metrics = _validated_metric_settings(metrics)
        previous = deepcopy(self.settings)
        self.settings["metric_settings"] = metrics
        # 旧字段仅保留网络参数，供旧调用和历史格式使用；新计算使用独立参数。
        self.settings["thresholds"] = {key: metrics["network"][key] for key in ("warning", "fault")}
        self.settings["alpha"] = metrics["network"]["alpha"]
        self.settings["thresholds_model_version"] = self.state["current_model_version"]
        try:
            self._save()
        except Exception:
            self.settings.clear()
            self.settings.update(previous)
            raise
        return deepcopy(metrics)

    def train(self, progress=None, training_config=None):
        candidates = deepcopy(self.training_candidates())
        if len(candidates) < 5:
            raise ValueError("至少需要 5 个已判定正常的完整采集：至少 2 个训练、3 个独立正常校准")
        if any(run.get("preprocessing") != self.config["preprocessing"] for run in candidates):
            raise ValueError("正常候选的预处理版本与当前配置不匹配，请先重新处理采集数据")
        config = deepcopy(self.config["training"])
        config.update(training_config or {})
        config["preprocessing"] = deepcopy(self.config["preprocessing"])
        fraction = float(config.get("validation_fraction", 0.2))
        if not 0 < fraction < 1:
            raise ValueError("健康校准比例必须在 0 与 1 之间")
        data = {run["run_id"]: self._read_run(run) for run in candidates}
        groups = {}
        for run in candidates:
            loaded = data[run["run_id"]]
            key = json.dumps([loaded["calibration_context"], len(loaded["velocity"])], sort_keys=True)
            groups.setdefault(key, []).append(run)
        calibration, training = [], []
        rng = random.Random(config["seed"])
        for group in groups.values():
            rng.shuffle(group)
            count = min(len(group) - 1, max(3, math.ceil(len(group) * fraction))) if len(group) >= 4 else 0
            calibration.extend(group[:count])
            training.extend(group[count:])
        if len(training) < 2:
            needed = 2 - len(training)
            training.extend(calibration[-needed:])
            del calibration[-needed:]
        if len(training) < 2 or len(calibration) < 3:
            raise ValueError("正常参考不足：需要至少 3 次同工况、同长度的独立校准，并保留至少 2 次训练采集")
        version = "model_" + uuid4().hex[:12]
        model_path = Path("models") / version / "model.pt"
        self.root.joinpath(model_path).parent.mkdir(parents=True, exist_ok=True)
        if progress:
            progress(0, f"读取 {len(training)} 个训练 run 和 {len(calibration)} 个校准 run")
        train_data = [data[run["run_id"]] for run in training]
        calibration_data = [data[run["run_id"]] for run in calibration]

        def training_progress(percent, message):
            if progress:
                progress(round(percent * 0.8), message)

        summary = algorithm.train_model(train_data, calibration_data, self.root / model_path,
                                        config, progress=training_progress)
        created_at = _now()
        architecture = summary.get("architecture", algorithm.MODEL_ARCHITECTURE)
        epochs = summary.get("trained_epochs", config["epochs"])
        timestamp = datetime.fromisoformat(created_at).astimezone().strftime("%Y%m%d_%H%M%S_%f")
        version = f"{architecture}_{epochs}ep_{timestamp}"
        final_path = Path("models") / version / "model.pt"
        self.root.joinpath(model_path).parent.rename(self.root / final_path.parent)
        model_path = final_path
        model = deepcopy(summary)
        model.update({
            "version": version, "created_at": created_at, "model_path": model_path.as_posix(),
            "training_run_ids": [run["run_id"] for run in training],
            "calibration_run_ids": [run["run_id"] for run in calibration],
            "training_samples": [{key: run[key] for key in
                                  ("run_id", "purpose", "manual_label", "training_eligible", "label_updated_at")}
                                 for run in candidates],
            "training_config": config, "initialization": "random",
            "preprocessing": deepcopy(self.config["preprocessing"]),
            "thresholds_at_training": deepcopy(self.settings["thresholds"]),
            "metric_settings_at_training": self.metric_settings,
            "thresholds_model_version_at_training": self.settings.get("thresholds_model_version"),
            "reanalysis_status": "running",
        })
        model["normal_reference"]["model_version"] = version
        self.models.append(model)
        self.state["current_model_version"] = version
        self._save()
        def reanalysis_progress(percent, message):
            if progress:
                progress(80 + round(percent * 0.2), message)

        return self.reanalyze_history(progress=reanalysis_progress)

    def reanalyze_history(self, progress=None):
        """沿用当前模型补齐缺失评价，可在历史重算中断后继续。"""
        model = self.current_model
        if model is None:
            raise ValueError("尚未建立模型，无法补算历史")
        pending = [run for run in self.list_runs()
                   if (self.latest_result(run["run_id"]) or {}).get("score_kind") != "normal_compatibility_v1"]
        model["reanalysis_status"] = "running"
        model.pop("reanalysis_error", None)
        self._save()
        model_file = self.root / Path(model["model_path"]).parent / "model.json"
        try:
            if progress:
                progress(0, f"补算 {len(pending)} 次缺失的历史评价")
            for index, run in enumerate(pending):
                self.evaluate(run["run_id"], reason="model_reanalysis")
                if progress:
                    progress(round(99 * (index + 1) / len(pending)),
                             f"历史重算 {index + 1}/{len(pending)}")
        except Exception as error:
            model["reanalysis_status"] = "failed"
            model["reanalysis_error"] = str(error)
            write_document(model_file, model)
            self._save()
            raise RuntimeError("模型已保存，但历史重算未全部完成；可重试历史补算：" + str(error)) from error
        model["reanalysis_status"] = "complete"
        write_document(model_file, model)
        self._save()
        if progress:
            progress(100, "历史评价已完整")
        return model

    def _normal_reference(self, model):
        """旧模型只使用原来留出的校准采集重建参考，待测采集不会加入。"""
        reference = model.get("normal_reference")
        if reference and reference.get("network_id", "").startswith("legacy:"):
            if reference["network_id"] != algorithm._network_id({}, self.root / model["model_path"]):
                reference = None
        if reference is not None:
            return reference
        ids = model.get("calibration_run_ids", [])
        if (len(ids) != len(set(ids)) or set(ids) & set(model.get("training_run_ids", []))
                or any(run_id not in self.runs or self.runs[run_id]["manual_label"] != "healthy" for run_id in ids)):
            return {"groups": [], "message": "原模型正常校准采集缺失、标签无效或与训练集重叠"}
        reference = algorithm.calibrate_model(
            [self._read_run(self.runs[run_id]) for run_id in ids], self.root / model["model_path"],
            preprocessing=model.get("preprocessing", self.config["preprocessing"]),
        )
        reference["model_version"] = model["version"]
        model["normal_reference"] = reference
        write_document(self.root / Path(model["model_path"]).parent / "normal_reference.json", reference)
        self._save()
        return reference

    def evaluate(self, run_id, progress=None, reason="manual"):
        record = self.runs[run_id]
        model = self.current_model
        model_path = self.root / model["model_path"] if model else None
        metrics = self.metric_settings
        config = {**self.config, "alpha": metrics["network"]["alpha"],
                  "network_alpha": metrics["network"]["alpha"], "vibration_alpha": metrics["vibration"]["alpha"]}
        if model:
            config["normal_reference"] = self._normal_reference(model)
        result = algorithm.evaluate_run(self._read_run(record), model_path=model_path,
                                        config=config, progress=progress)
        role = "independent"
        if model and run_id in model["training_run_ids"]:
            role = "training"
        elif model and run_id in model["calibration_run_ids"]:
            role = "calibration"
        result.update({
            "id": uuid4().hex, "run_id": run_id, "run_name": record["run_name"],
            "captured_at": record["captured_at"], "captured_date": record["captured_date"],
            "evaluated_at": _now(), "model_version": model["version"] if model else None,
            "source_type": record["source_type"], "role": role, "reason": reason,
            "manual_label_at_evaluation": record["manual_label"],
            "metric_settings": deepcopy(metrics),
            "thresholds": deepcopy(self.settings["thresholds"]),
            "thresholds_model_version": self.settings.get("thresholds_model_version"),
            "thresholds_review_required": self.thresholds_review_required,
        })
        result["assessment"] = assess_compatibility(result, metrics, result["thresholds_review_required"])
        result_path = Path("evaluations") / f"{result['id']}.json"
        write_document(self.root / result_path, result)
        summary = {key: result[key] for key in (
            "id", "run_id", "run_name", "captured_at", "captured_date", "evaluated_at",
            "model_version", "role", "reason", "score", "thresholds", "assessment",
            "thresholds_model_version", "thresholds_review_required",
            "analysis_score", "score_kind", "alpha", "vibration_p_value", "network_p_value",
            "xy_rms_mm_s", "reconstruction_error_p95",
            "metric_settings",
        )}
        summary["result_file"] = result_path.as_posix()
        self.state["results"].append(summary)
        try:
            self._save()
        except Exception:
            self.state["results"].pop()
            raise
        return result

    def history(self, run_id=None, model_version=None):
        entries = self.state["results"]
        return [read_json(self.root / entry["result_file"]) for entry in entries
                if (run_id is None or entry["run_id"] == run_id)
                and (model_version is None or entry["model_version"] == model_version)]

    def latest_result(self, run_id, model_version=None):
        if model_version is None:
            model_version = self.state["current_model_version"]
        for entry in reversed(self.state["results"]):
            if entry["run_id"] == run_id and entry["model_version"] == model_version:
                return read_json(self.root / entry["result_file"])
        return None

    def ensure_latest_result(self, run_id, progress=None):
        """最新视图补齐缺失或旧倍率评价；已有相容度结果直接复用，包括无效状态。"""
        result = self.latest_result(run_id)
        if result is not None and result.get("score_kind") == "normal_compatibility_v1":
            return result
        return self.evaluate(run_id, progress=progress, reason="selection_refresh")
