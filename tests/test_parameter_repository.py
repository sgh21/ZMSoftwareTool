from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from core.parameter_repository import ParameterFileRepository


def test_create_timestamp_versions_never_overwrites(tmp_path: Path) -> None:
    repo = ParameterFileRepository(tmp_path)

    first = repo.create_version(
        "identified_model",
        {"identification": {"confidence_current": 90.0}},
        timestamp="20260612_120000",
    )
    second = repo.create_version(
        "identified_model",
        {"identification": {"confidence_current": 91.0}},
        timestamp="20260612_120000",
    )

    assert first.name == "identified_model_20260612_120000.yaml"
    assert second.name == "identified_model_20260612_120000_02.yaml"
    assert first.exists()
    assert second.exists()


def test_kind_mismatch_is_rejected(tmp_path: Path) -> None:
    repo = ParameterFileRepository(tmp_path)
    path = repo.create_version("camera_monitoring", {"model_monitoring": {}})

    with pytest.raises(ValueError, match="expected identified_model"):
        repo.load_document(path, expected_kind="identified_model")


def test_latest_version_uses_timestamp_and_collision_suffix(tmp_path: Path) -> None:
    repo = ParameterFileRepository(tmp_path)
    older = repo.create_version(
        "controller_model",
        {"controller_model": {"mdh": {}}},
        timestamp="20260612_115959",
    )
    newer = repo.create_version(
        "controller_model",
        {"controller_model": {"mdh": {}}},
        timestamp="20260612_120000",
    )
    newest = repo.create_version(
        "controller_model",
        {"controller_model": {"mdh": {}}},
        timestamp="20260612_120000",
    )

    assert repo.latest_version("controller_model").path == newest
    assert older in [item.path for item in repo.list_versions("controller_model")]
    assert newer in [item.path for item in repo.list_versions("controller_model")]


def test_active_parameters_round_trip(tmp_path: Path) -> None:
    repo = ParameterFileRepository(tmp_path)
    identified = repo.create_version("identified_model", {"identification": {}})
    camera = repo.create_version("camera_monitoring", {"model_monitoring": {}})

    repo.activate_version("identified_model", identified)
    repo.activate_version("camera_monitoring", camera)

    active = repo.load_active()
    assert active["nominal_robot"] == "config/nominal_robot.yaml"
    assert repo.active_path_for("identified_model") == identified
    assert repo.active_path_for("camera_monitoring") == camera


def test_confidence_update_appends_history_and_updates_current(tmp_path: Path) -> None:
    repo = ParameterFileRepository(tmp_path)
    model = repo.create_version(
        "identified_model",
        {
            "identification": {
                "confidence_current": 90.0,
                "confidence_history": [
                    {
                        "timestamp": "2026-06-12T12:00:00+08:00",
                        "value": 90.0,
                        "source": "test",
                        "reason": "initial",
                    }
                ],
                "metrics": {"position_uncertainty_rmse_mm": 0.2},
            }
        },
    )

    repo.append_confidence_history(
        model,
        80.0,
        source="model_degradation_monitoring",
        reason="test drift",
        position_uncertainty_rmse_mm=1.2,
        evaluation_record={"sample_count": 2},
    )
    document = yaml.safe_load(model.read_text(encoding="utf-8"))
    section = document["payload"]["identification"]

    assert section["confidence_current"] == pytest.approx(80.0)
    assert section["confidence"] == pytest.approx(80.0)
    assert section["confidence_history"][-1]["value"] == pytest.approx(80.0)
    assert len(section["confidence_history"]) == 2
    assert section["metrics"]["position_uncertainty_rmse_mm"] == pytest.approx(1.2)
    assert section["monitoring"]["last_degradation_evaluation"]["sample_count"] == 2
    assert document["updated_at"] == section["updated_at"]


def test_create_controller_model_from_identified_copies_fk_fields(tmp_path: Path) -> None:
    repo = ParameterFileRepository(tmp_path)
    identified = repo.create_version(
        "identified_model",
        {
            "identification": {
                "timestamp": "2026-06-12T12:00:00+08:00",
                "identified_robot": {
                    "base_xyz": [9.0, 8.0, 7.0],
                    "tool_xyz": [0.0, 0.0, 0.05],
                    "tool_rpy": [0.0, 0.0, 0.0],
                    "mdh": {
                        "alpha": [0, 1, 0, 0, 1, -1],
                        "a": [0, 0, -0.61, -0.57, 0, 0],
                        "d": [0.12, 0, 0, 0.16, 0.11, 0.09],
                        "theta_offset": [0, 0, 0, 0, 0, 0],
                    },
                },
            }
        },
    )

    controller = repo.create_controller_model_from_identified(identified)
    document = yaml.safe_load(controller.read_text(encoding="utf-8"))
    payload = document["payload"]
    model = payload["controller_model"]

    assert document["kind"] == "controller_model"
    assert model["tool_xyz"] == [0.0, 0.0, 0.05]
    assert model["mdh"]["a"][2] == pytest.approx(-0.61)
    assert "base_xyz" not in model
    assert payload["source_identified_model"].endswith(identified.name)
