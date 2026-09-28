"""图片批次与仿真交付格式的读取边界；原始文件只读。"""

from copy import deepcopy
import json
from pathlib import Path
import re


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
IMAGE_NAME = re.compile(r"(B\d{3})_(P\d{3})_(D\d{3})", re.IGNORECASE)


def simulation_parameters(document):
    """只提取测量配置，逐图真值不进入参数或视觉测量输入。"""
    camera, board, transforms = (document[key] for key in ("camera", "charuco", "system_transforms"))
    return {
        "camera_matrix": camera["camera_matrix"], "dist_coeffs": camera["dist_coeffs"],
        "image_size_px": camera["image_size_px"],
        "hand_eye": transforms["E_T_C_hand_eye"],
        "target_pose_base": transforms["B_T_M_charuco_top_left"],
        "board_type": "charuco", "square_size_mm": board["square_length_mm"],
        "charuco": {key: deepcopy(board[key]) for key in (
            "dictionary", "squares_xy", "marker_length_mm", "marker_ids", "legacy_pattern",
        ) if key in board},
        "reference_rotation": None, "max_reprojection_error_px": None,
        "length_unit": "mm", "transform_convention": "E_T_C", "end_frame": "tool0",
        "target_id": "fixed_charuco", "source_note": "仿真已知标定参数；末端为 tool0",
    }


def image_identity(path):
    match = IMAGE_NAME.fullmatch(Path(path).stem)
    if not match:
        raise ValueError(f"图片名应为 B000_P001_D001.png：{Path(path).name}")
    return tuple(value.upper() for value in match.groups())


def _json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def image_batch(source, parameters):
    """读取单批次图片、目录或精简 record.json；测量始终从图片解算。"""
    record = None
    if isinstance(source, (list, tuple)):
        images = [Path(path).resolve() for path in source]
        if not images:
            raise ValueError("请选择本次采集的图片")
        folder = images[0].parent
        folder = folder.parent if folder.name == "calibration_images" else folder
    else:
        path = Path(source).resolve()
        if path.suffix.lower() in IMAGE_SUFFIXES:
            return image_batch([path], parameters)
        folder = path if path.is_dir() else path.parent
        if folder.name == "calibration_images":
            folder = folder.parent
        if path.is_file():
            record = _json(path)
        image_folder = folder / "calibration_images"
        if not image_folder.is_dir():
            image_folder = folder
        images = sorted(path for path in image_folder.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if record is None and (folder / "record.json").is_file():
        record = _json(folder / "record.json")
    frames = {}
    if record is not None:
        for frame in record["frames"]:
            path = (folder / frame["image"]).resolve()
            if path in frames:
                raise ValueError(f"观测记录中图片重复：{path.name}")
            frames[path] = frame
        if not isinstance(source, (list, tuple)):
            images = list(frames)
    if not images:
        raise ValueError("目录中没有观测图片；请选择 B001 这样的批次目录")
    identities = [image_identity(path) for path in images]
    batch_ids = {identity[0] for identity in identities}
    if len(batch_ids) != 1:
        raise ValueError("一次只能导入一个批次，请分开导入 B000、B001 等批次")
    if len(set(identities)) != len(identities):
        raise ValueError("同批次、同点、同接近方向只能有一张观测图片")
    batch_id = next(iter(batch_ids))
    if record is not None and record["batch"] != batch_id:
        raise ValueError("记录批次与图片文件名不一致")
    samples = []
    for image_path, (_, point, direction) in zip(images, identities):
        sample = {
            "point_id": point, "direction_id": direction, "sample_id": image_path.stem,
            "image_path": str(image_path),
        }
        if record is not None:
            if image_path not in frames:
                raise ValueError(f"图片没有对应观测记录：{image_path.name}")
            frame = frames[image_path]
            sample["ideal_pose"] = frame["ideal"]
            if "actual" in frame:
                sample["simulation_truth"] = {"end_pose": frame["actual"]}
            if frame.get("captured_at_utc") or frame.get("captured_at"):
                sample["captured_at"] = frame.get("captured_at_utc") or frame["captured_at"]
        samples.append(sample)
    return {
        "schema_version": 2, "batch_id": batch_id, "label": batch_id,
        "program_id": folder.parent.name if record else "fixed_image_points",
        "target_id": parameters.get("target_id", "configured_target"),
        "length_unit": "mm", "sampling_protocol": "multidirectional",
        "comparison_status": "simulation" if record and any("simulation_truth" in item for item in samples) else "observed",
        "samples": samples, "base_rotations": {}, "initial_errors": {},
        "warnings": [], "source_path": str(folder / "record.json" if record else folder),
        **({"captured_at": record.get("captured_at_utc") or record["captured_at"]}
           if record and (record.get("captured_at_utc") or record.get("captured_at")) else {}),
    }
