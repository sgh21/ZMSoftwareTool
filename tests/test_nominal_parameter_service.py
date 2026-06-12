from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.nominal_parameter_service import (
    DEFAULT_NOMINAL_ROBOT,
    NominalParameterService,
    nominal_after_applying_error_parameters,
)


def nominal_payload(*, a2: float = -0.612) -> dict:
    return {
        "nominal_robot": {
            "base_xyz": [1.0, 2.0, 3.0],
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


def test_nominal_service_loads_read_only_design_baseline(tmp_path: Path) -> None:
    write_nominal(tmp_path, nominal_payload(a2=-0.5))

    loaded = NominalParameterService(tmp_path).load_nominal()

    assert "base_xyz" not in loaded
    assert loaded["tool_xyz"] == [0.0, 0.0, 0.039]
    assert loaded["mdh"]["a"][1] == pytest.approx(-0.5)
    assert loaded["joint_limits"][0] == pytest.approx([-3.14, 3.14])


def test_nominal_service_falls_back_to_builtin_defaults(tmp_path: Path) -> None:
    loaded = NominalParameterService(tmp_path).load_nominal()

    assert loaded["tool_xyz"] == pytest.approx(DEFAULT_NOMINAL_ROBOT["tool_xyz"])
    assert loaded["mdh"]["a"][2] == pytest.approx(DEFAULT_NOMINAL_ROBOT["mdh"]["a"][2])


def test_nominal_after_applying_error_parameters_only_absorbs_mdh_errors() -> None:
    nominal = nominal_payload()["nominal_robot"]

    identified = nominal_after_applying_error_parameters(
        nominal,
        {
            "delta_a_2": 0.02,
            "delta_d_4": -0.001,
            "delta_Ttz": 0.003,
            "delta_Btx": 9.0,
        },
    )

    assert "base_xyz" not in identified
    assert identified["mdh"]["a"][1] == pytest.approx(-0.592)
    assert identified["mdh"]["d"][3] == pytest.approx(0.162941)
    assert identified["tool_xyz"][2] == pytest.approx(0.039)


def test_nominal_service_no_longer_exposes_update_or_rollback_entries(tmp_path: Path) -> None:
    service = NominalParameterService(tmp_path)

    assert not hasattr(service, "update_direct")
    assert not hasattr(service, "update_values")
    assert not hasattr(service, "update_from_identification_file")
    assert not hasattr(service, "rollback")
    assert not hasattr(service, "has_backup")
