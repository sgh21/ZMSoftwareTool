"""Legacy adapters preserve raw files and keep demo matching explicit."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from core.services import position_monitoring_service
from experiments.robot_position_debug import build_legacy_batch, make_debug_manifests
from experiments import robot_position_debug


def write_legacy_folder(folder):
    images = folder / "calibration_images"
    images.mkdir(parents=True)
    names = ["calib_00.jpg", "calib_01.jpg"]
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[:, :3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    poses[0, :3, 3] = [0.12, -0.34, 0.56]
    poses[1, :3, 3] = [0.14, -0.31, 0.52]
    np.savez(folder / "robot_poses.npz", filenames=names, T_base_tool=poses)
    for name in names:
        # The adapter only records paths; image decoding belongs to the service.
        (images / name).write_bytes(b"raw image fixture")
    (folder / "camera_board_poses_none.npz").write_bytes(b"legacy cache, do not read or overwrite")
    return poses


def read_file_bytes(folder):
    return {path.relative_to(folder): path.read_bytes() for path in folder.rglob("*") if path.is_file()}


def test_legacy_units_and_controller_orientation_are_explicit(tmp_path):
    original = write_legacy_folder(tmp_path / "legacy")
    batch = build_legacy_batch(tmp_path / "legacy", include_controller_rotations=True)

    pose = np.asarray(batch["samples"][0]["robot_pose"])
    np.testing.assert_allclose(pose[:3, 3], [120, -340, 560])
    np.testing.assert_array_equal(pose[:3, :3], original[0, :3, :3])
    np.testing.assert_array_equal(batch["base_rotations"]["calib_00"], original[0, :3, :3])
    assert batch["length_unit"] == "mm"
    assert batch["orientation_source"] == "controller_approximation"
    assert batch["samples"][0]["direction_id"] == "unspecified"
    assert Path(batch["samples"][0]["image_path"]).is_absolute()
    assert "vision_pose" not in batch["samples"][0]
    assert any("不能计算 RP" in warning for warning in batch["warnings"])
    assert any("控制器末端朝向近似" in warning for warning in batch["warnings"])
    with np.load(tmp_path / "legacy" / "robot_poses.npz") as source:
        np.testing.assert_array_equal(source["T_base_tool"], original)


def test_legacy_adapter_does_not_assume_base_orientation(tmp_path):
    write_legacy_folder(tmp_path / "legacy")
    batch = build_legacy_batch(tmp_path / "legacy", filenames=["calib_01.jpg"])
    assert batch["base_rotations"] == {}
    assert batch["orientation_source"] == "unspecified"
    assert [sample["point_id"] for sample in batch["samples"]] == ["calib_01"]
    with pytest.raises(ValueError, match="找不到"):
        build_legacy_batch(tmp_path / "legacy", filenames=["missing.jpg"])


def test_debug_manifests_pair_only_calib00_without_modifying_raw_files(tmp_path):
    data_root = tmp_path / "robot_error"
    for name in ("calib_data20", "camera_pos_normal50", "camera_pos_x05y-05_50"):
        write_legacy_folder(data_root / name)
    before = read_file_bytes(data_root)
    output = tmp_path / "outputs"
    paths = make_debug_manifests(data_root, output)
    manifests = {
        key: json.loads(Path(path).read_text(encoding="utf-8")) for key, path in paths.items()
    }

    assert read_file_bytes(data_root) == before
    assert set(paths) == {"handeye_manifest", "baseline_manifest", "current_manifest", "parameters"}
    assert all(Path(path).parent == output for path in paths.values())
    handeye = manifests["handeye_manifest"]
    assert handeye["comparison_status"] == "calibration_only"
    assert len(handeye["samples"]) == 2  # All fixture poses remain available for calibration.
    baseline = manifests["baseline_manifest"]
    current = manifests["current_manifest"]
    assert baseline["batch_id"] != current["batch_id"]
    assert baseline["program_id"] == current["program_id"]
    assert baseline["target_id"] == current["target_id"]
    for batch in (baseline, current):
        assert batch["comparison_status"] == "debug_unverified"
        assert len(batch["samples"]) == 1
        assert batch["samples"][0]["point_id"] == "calib_00"
        assert batch["samples"][0]["direction_id"] == "unverified"
        assert batch["initial_errors"] == {}
        assert any("不可当作实测机器人精度退化" in warning for warning in batch["warnings"])
    assert current["base_rotations"] == {}
    assert manifests["parameters"]["hand_eye"] is None


@pytest.mark.parametrize("relative_output", [".", "generated"])
def test_generated_files_cannot_be_written_inside_raw_data(tmp_path, relative_output):
    data_root = tmp_path / "robot_error"
    data_root.mkdir()
    with pytest.raises(ValueError, match="原始 robot_error 数据目录之外"):
        make_debug_manifests(data_root, data_root / relative_output)
    assert list(data_root.iterdir()) == []


def test_cli_run_recovers_hand_eye_and_metrics_without_changing_main_storage(tmp_path, monkeypatch, capsys):
    output = tmp_path / "debug_output"
    inputs = tmp_path / "input_manifests"
    inputs.mkdir()
    paths = {
        key: str(inputs / f"{key}.json")
        for key in ("parameters", "handeye_manifest", "baseline_manifest", "current_manifest")
    }
    hand_eye = np.eye(4)
    hand_eye[:3, :3] = cv2.Rodrigues(np.array([0.1, -0.2, 0.05]))[0]
    hand_eye[:3, 3] = [20, -80, 70]
    target = np.eye(4)
    target[:3, 3] = [500, 200, 700]
    rng = np.random.default_rng(14)
    calibration_samples = []
    for index in range(16):
        robot = np.eye(4)
        robot[:3, :3] = cv2.Rodrigues(rng.normal(0, 0.4, 3))[0]
        robot[:3, 3] = rng.normal(0, 100, 3)
        calibration_samples.append({
            "point_id": str(index), "direction_id": "calibration", "sample_id": str(index),
            "robot_pose": robot.tolist(),
            "vision_pose": (np.linalg.inv(robot @ hand_eye) @ target).tolist(),
        })

    def measurement_samples(x_positions):
        samples = []
        for index, x in enumerate(x_positions):
            robot = np.eye(4)
            robot[:3, 3] = [x, 0, 0]
            samples.append({
                "point_id": "P1", "direction_id": "D1", "sample_id": str(index),
                "vision_pose": (np.linalg.inv(robot @ hand_eye) @ target).tolist(),
            })
        return samples

    common = {"program_id": "fixture", "target_id": "fixture_board", "length_unit": "mm"}
    parameters = {
        "hand_eye": None, "camera_matrix": None, "dist_coeffs": [0] * 5,
        "board_grid": [9, 6], "square_size_mm": 10, "length_unit": "mm",
        "version": "fixture",
    }
    documents = {
        "parameters": parameters,
        "handeye_manifest": {**common, "samples": calibration_samples, "comparison_status": "calibration_only"},
        "baseline_manifest": {
            **common, "samples": measurement_samples([-0.2, 0, 0.2]),
            "comparison_status": "debug_unverified", "base_rotations": {"P1": np.eye(3).tolist()},
            "initial_errors": {"P1": {"D1": [1, 0, 0]}},
        },
        "current_manifest": {
            **common, "samples": measurement_samples([0.2, 0.5, 0.8]),
            "comparison_status": "debug_unverified",
        },
    }
    for key, document in documents.items():
        Path(paths[key]).write_text(json.dumps(document), encoding="utf-8")

    # Redirect the service's default app root as well, so a regression cannot
    # touch the developer's actual saved parameters or measurements.
    main_app = tmp_path / "main_app"
    (main_app / "config").mkdir(parents=True)
    (main_app / "config" / "robot_position.json").write_text(json.dumps(parameters), encoding="utf-8")
    (main_app / "storage").mkdir()
    (main_app / "storage" / "existing_record.json").write_text('{"preserve": true}', encoding="utf-8")
    before = read_file_bytes(main_app / "storage")
    monkeypatch.setattr(position_monitoring_service, "__file__", str(main_app / "core" / "services" / "position_monitoring_service.py"))
    monkeypatch.setattr(robot_position_debug, "make_debug_manifests", lambda *_args: paths)
    monkeypatch.setattr("sys.argv", [
        "run_debug.py", "--data-root", str(tmp_path / "raw"),
        "--output-dir", str(output), "--run", "--method", "PARK",
    ])
    robot_position_debug.main()

    report = json.loads((output / "debug_result.json").read_text(encoding="utf-8"))
    assert report["debug_only"] is True
    assert report["session_root"] == str(output / "session")
    np.testing.assert_allclose(report["calibration"]["hand_eye"], hand_eye, atol=1e-7)
    result = report["result"]
    assert result["status"] == "调试比较 · 采样对应待确认"
    summary = result["summary"]
    np.testing.assert_allclose(summary["drift_base"], [0.5, 0, 0], atol=1e-7)
    radii = np.array([0.2, 0, 0.2])
    expected_rp = radii.mean() + 3 * radii.std(ddof=1)
    assert summary["rp_baseline"] == pytest.approx(expected_rp)
    assert summary["rp_current"] == pytest.approx(1.5 * expected_rp)
    assert summary["absolute_ap"] == pytest.approx(1.5)
    assert summary["absolute_ap_change"] == pytest.approx(0.5)
    assert read_file_bytes(main_app / "storage") == before
    assert list((output / "session" / "storage" / "position_monitoring" / "history").glob("*.json"))
    assert "不是实测机器人精度退化" in capsys.readouterr().out


def test_cli_default_only_generates_inputs(tmp_path, monkeypatch, capsys):
    paths = {"parameters": str(tmp_path / "parameters.json")}
    monkeypatch.setattr(robot_position_debug, "make_debug_manifests", lambda *_args: paths)

    def unexpected_execution(*_args, **_kwargs):
        pytest.fail("Without --run the CLI must not execute calibration or evaluation")

    monkeypatch.setattr(robot_position_debug, "run_debug_pipeline", unexpected_execution)
    monkeypatch.setattr("sys.argv", ["run_debug.py", "--output-dir", str(tmp_path)])
    robot_position_debug.main()
    assert "尚未执行标定或评估" in capsys.readouterr().out
    assert not (tmp_path / "session").exists()
