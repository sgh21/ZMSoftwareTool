"""对话框及其共用的数据读取。"""

from core.services.position_monitoring_service import read_document


def read_batch_records(dataset):
    parameters = read_document(dataset / "parameters.json")
    records = {}
    for run in parameters.get("runs", []):
        batch_id = str(run["batch_id"])
        records[batch_id] = dataset / run.get("record", f"{batch_id}/record.json")
    records.update({path.parent.name: path for path in sorted(dataset.glob("B[0-9][0-9][0-9]/record.json"))})
    if not records:
        raise ValueError("目录缺少批次 record.json 或 runs 信息")
    return records
