"""定位监控的日记录、图像托管与日志；原始输入只读。"""

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil


def write_document(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def observation_time(batch, fallback_time=None, fallback_source="evaluated_at"):
    """保留真实时间；调试相对日是显式元数据，不伪造日期。"""
    samples = batch["samples"]
    captured = batch.get("captured_at") or batch.get("captured_at_utc")
    def parse(value):
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return stamp if stamp.tzinfo else stamp.astimezone()

    if captured is None and samples and all(sample.get("captured_at") for sample in samples):
        captured = max((sample["captured_at"] for sample in samples), key=parse)
    imported = batch.get("imported_at")
    parsed = parse(captured or imported or fallback_time)
    batch["observed_at"] = parsed.isoformat()
    batch["time_source"] = "captured_at" if captured else "imported_at" if imported else fallback_source
    if batch.get("comparison_status") == "simulation" or batch.get("source_type") == "simulation":
        batch_id = batch["batch_id"]
        number = int(batch_id[1:]) if batch_id.startswith("B") and batch_id[1:].isdigit() else 0
        batch["debug_day_index"] = number - 1 if number > 0 else None
    else:
        batch.pop("debug_day_index", None)


class PositionStore:
    def __init__(self, root):
        self.root = Path(root)

    def record(self, section, value, batch, parameters):
        debug_day = batch.get("debug_day_index")
        date = datetime.fromisoformat(batch["observed_at"]).astimezone().date().isoformat()
        path = (self.root / "debug" / f"day_{debug_day:04d}.json" if debug_day is not None
                else self.root / "daily" / f"{date}.json")
        document = read_json(path) if path.exists() else {
            "schema_version": 1, "date": date if debug_day is None else None,
            "debug_day_index": debug_day, "parameters": {},
            "observations": [], "baselines": [], "evaluations": [],
        }
        document["parameters"][parameters["version"]] = deepcopy(parameters)
        document[section].append(deepcopy(value))
        write_document(path, document)
        return path

    def history(self):
        records = []
        for folder in ("daily", "debug"):
            for path in (self.root / folder).glob("*.json"):
                records.extend(read_json(path).get("evaluations", []))
        for path in (self.root / "history").glob("*.json"):
            records.append(self._complete_times(read_json(path)))
        return sorted(records, key=lambda item: (item["created_at"], item["id"]))

    def result(self, reference):
        document = read_json(reference["path"])
        if "evaluations" not in document:
            return self._complete_times(document)
        return next(item for item in document["evaluations"] if item["id"] == reference["id"])

    @staticmethod
    def _complete_times(result):
        """旧评估只补时间来源，不重算指标或写回已有文件。"""
        fields = ("observed_at", "time_source", "debug_day_index")
        if all(key in result and f"baseline_{key}" in result for key in fields):
            return result

        def snapshot(path):
            return read_json(path) if path and Path(path).is_file() else {}

        current = snapshot(result.get("current_batch_path"))
        current.setdefault("samples", [])
        current.setdefault("batch_id", result.get("batch_id", ""))
        current.setdefault("comparison_status", result.get("comparison_status"))
        observation_time(current, result["created_at"])
        for key in fields:
            result.setdefault(key, current.get(key))

        if result.get("baseline_id"):
            baseline = snapshot(result.get("baseline_path"))
            batch = snapshot(baseline.get("batch_path")) if baseline.get("batch_path") else baseline.get("batch", {})
            batch.setdefault("samples", [])
            batch.setdefault("batch_id", "")
            created = baseline.get("created_at") or result.get("baseline_created_at")
            observation_time(batch, created or result["created_at"],
                             "baseline_created_at" if created else "evaluated_at")
            for key in fields:
                result.setdefault(f"baseline_{key}", batch.get(key))
        else:
            for key in fields:
                result.setdefault(f"baseline_{key}", None)
        return result

    def manage_images(self, batch, identifier):
        folder = self.root / "images"
        index_path = folder / "index.json"
        index = read_json(index_path) if index_path.exists() else {}
        changed = False
        for sample in batch["samples"]:
            if not sample.get("image_path"):
                continue
            source = Path(sample["image_path"]).resolve()
            if source.is_relative_to(folder.resolve()) or not source.is_file():
                continue
            stat = source.stat()
            signature = [stat.st_size, stat.st_mtime_ns]
            entry = index.get(str(source))
            if entry and entry["signature"] == signature and Path(entry["path"]).is_file():
                destination = Path(entry["path"])
            else:
                destination = folder / identifier / source.name
                # 同一观测可包含不同目录下的同名图片，按序号区分。
                if destination.exists():
                    destination = destination.with_name(f"{len(index)}_{source.name}")
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                index[str(source)] = {"signature": signature, "path": str(destination)}
                changed = True
            sample["source_image_path"] = sample.get("source_image_path", str(source))
            sample["image_path"] = str(destination)
        if changed:
            write_document(index_path, index)

    def append_log(self, message, level="INFO"):
        now = datetime.now(timezone.utc)
        record = {"timestamp": now.isoformat(), "level": level, "message": str(message)}
        path = self.root / "daily" / f"{now.astimezone().date().isoformat()}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record

    def read_logs(self):
        return [json.loads(line) for path in sorted((self.root / "daily").glob("*.jsonl"))
                for line in path.read_text(encoding="utf-8").splitlines() if line]
