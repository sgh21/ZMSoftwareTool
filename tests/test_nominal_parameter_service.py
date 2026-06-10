from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.calibration_persistence import load_identification_result, save_identification_result
from core.calibration_service import CalibrationService
from core.nominal_parameter_service import (
    NominalParameterService,
    nominal_after_applying_error_parameters,
)


def nominal_payload(*, a2: float = -0.612, base_x: float = 1.0) -> dict:
    return {
        "nominal_robot": {
            "base_xyz": [base_x, 2.0, 3.0],
            "base_rpy": [0.1, 0.2, 0.3],
            "tool_xyz": [0.0, 0.0, 0.039],
            "tool_rpy": [0.0, 0.0, 0.0],
            "mdh": {
                "alpha": [0.0, 1.57, 0.0, 0.0, 1.57, -1.57],
                "a": [0.0, a2, -0.5723, 0.0, 0.0, 0.0],
                "d": [0.1273, 0.0, 0.0, 0.163941, 0.1157, 0.0922],
                "theta_offset": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            },
            "joint_limits": [[-3.14, 3.14]] * 6,
        }
    }


def write_nominal(root: Path, payload: dict | None = None) -> Path:
    path = root / "config" / "nominal_robot.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload or nominal_payload(), sort_keys=False),
        encoding="utf-8",
    )
    return path


def read_nominal(root: Path) -> dict:
    data = yaml.safe_load((root / "config" / "nominal_robot.yaml").read_text(encoding="utf-8"))
    return data["nominal_robot"]


def identified_payload(root: Path, errors: dict[str, float]) -> dict:
    return nominal_after_applying_error_parameters(read_nominal(root), errors)


def test_direct_update_persists_and_keeps_one_backup(tmp_path: Path) -> None:
    write_nominal(tmp_path)
    service = NominalParameterService(tmp_path)

    updated = nominal_payload(a2=-0.5, base_x=9.0)
    result = service.update_direct(updated)

    assert result.nominal_path == tmp_path / "config" / "nominal_robot.yaml"
    assert result.backup_path == tmp_path / "storage" / "model_versions" / "nominal_robot_previous.yaml"
    assert read_nominal(tmp_path)["base_xyz"][0] == 9.0

    backup = yaml.safe_load(result.backup_path.read_text(encoding="utf-8"))["nominal_robot"]
    assert backup["base_xyz"][0] == 1.0
    assert backup["mdh"]["a"][1] == -0.612


def test_value_update_overwrites_provided_values_without_delta_math(tmp_path: Path) -> None:
    write_nominal(tmp_path)
    service = NominalParameterService(tmp_path)

    service.update_values(
        {
            "nominal_values": {
                "base_xyz": [0.25, 2.5, 3.5],
                "mdh": {
                    "a": [0.0, -0.5, -0.5723, 0.0, 0.0, 0.0],
                    "d": [0.1273, 0.0, 0.0, 0.001, 0.0, 0.0],
                },
            }
        }
    )

    saved = read_nominal(tmp_path)
    assert saved["base_xyz"] == [0.25, 2.5, 3.5]
    assert saved["mdh"]["a"][1] == pytest.approx(-0.5)
    assert saved["mdh"]["d"][3] == pytest.approx(0.001)


def test_nominal_delta_section_is_rejected(tmp_path: Path) -> None:
    write_nominal(tmp_path)
    service = NominalParameterService(tmp_path)

    with pytest.raises(KeyError, match="nominal_delta"):
        service.update_values({"nominal_delta": {"base_xyz": [9.0, 0.0, 0.0]}})


def test_identification_yaml_updates_nominal_parameters(tmp_path: Path) -> None:
    write_nominal(tmp_path)
    result_yaml = tmp_path / "config" / "calibration_result.yaml"
    errors = {"delta_a_2": 0.02, "delta_Bty": -0.5, "delta_Ttz": 0.001}
    save_identification_result(
        result_yaml,
        errors,
        nominal_robot=read_nominal(tmp_path),
        identified_robot=identified_payload(tmp_path, errors),
        fit_rmse_mm=0.1,
        fit_max_error_mm=0.2,
        position_error_rmse_mm=1.0,
        position_error_max_mm=1.5,
        sample_count=10,
    )

    NominalParameterService(tmp_path).update_from_identification_file(result_yaml)

    saved = read_nominal(tmp_path)
    assert saved["base_xyz"][1] == 1.5
    assert saved["tool_xyz"][2] == pytest.approx(0.04)
    assert saved["mdh"]["a"][1] == pytest.approx(-0.592)


def test_loading_same_identification_after_nominal_update_uses_zero_residual(
    tmp_path: Path,
) -> None:
    write_nominal(tmp_path)
    result_yaml = tmp_path / "config" / "calibration_result.yaml"
    errors = {"delta_Btx": 0.01}
    save_identification_result(
        result_yaml,
        errors,
        nominal_robot=read_nominal(tmp_path),
        identified_robot=identified_payload(tmp_path, errors),
        fit_rmse_mm=0.1,
        fit_max_error_mm=0.2,
        position_error_rmse_mm=10.0,
        position_error_max_mm=10.0,
        sample_count=10,
    )
    service = CalibrationService(project_root=tmp_path)
    service.load_active_parameters(result_yaml)
    before = service.compute_predicted_position([0.0, -58.0, 82.0, -112.0, -90.0, 0.0])
    assert before.error_norm_mm == pytest.approx(10.0)

    NominalParameterService(tmp_path).update_from_identification_file(result_yaml)
    service.reload_nominal_parameters()
    loaded = service.load_active_parameters(result_yaml)
    after = service.compute_predicted_position([0.0, -58.0, 82.0, -112.0, -90.0, 0.0])

    assert after.error_norm_mm == pytest.approx(0.0)
    assert service.active_parameter_values["delta_Btx"] == pytest.approx(0.01)


def test_identification_result_without_full_model_is_rejected(
    tmp_path: Path,
) -> None:
    write_nominal(tmp_path)
    result_yaml = tmp_path / "config" / "calibration_result.yaml"
    result_yaml.write_text(
        yaml.safe_dump(
            {
                "identification": {
                    "error_parameters": {"delta_Btx": 0.01},
                    "metrics": {"fit_rmse_mm": 0.1, "fit_max_error_mm": 0.2},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="nominal_robot"):
        load_identification_result(result_yaml)
    with pytest.raises(ValueError, match="nominal_robot"):
        CalibrationService(project_root=tmp_path).load_active_parameters(result_yaml)
    with pytest.raises(ValueError, match="nominal_robot"):
        NominalParameterService(tmp_path).update_from_identification_file(result_yaml)


def test_rollback_restores_previous_version_once(tmp_path: Path) -> None:
    write_nominal(tmp_path)
    service = NominalParameterService(tmp_path)
    service.update_direct(nominal_payload(base_x=7.0))

    service.rollback()

    assert read_nominal(tmp_path)["base_xyz"][0] == 1.0
    assert not service.has_backup()
    with pytest.raises(FileNotFoundError):
        service.rollback()


def test_direct_update_validates_required_nominal_fields(tmp_path: Path) -> None:
    service = NominalParameterService(tmp_path)

    with pytest.raises(KeyError):
        service.update_direct({"nominal_robot": {"base_xyz": [0.0, 0.0, 0.0]}})
