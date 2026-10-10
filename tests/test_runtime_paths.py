"""发布资源只读，持久化与界面输出统一写入用户目录。"""

from pathlib import Path
import shutil
import sys

import pytest

from core.runtime_paths import bundle_root, data_root
from core.services.position_monitoring_service import PositionMonitoringService
from core.services.spindle_monitoring_service import SpindleMonitoringService


def test_source_paths_remain_at_project_root(monkeypatch, tmp_path):
    monkeypatch.delenv("ZMSOFTWARE_DATA_DIR", raising=False)
    monkeypatch.delattr(sys, "frozen", raising=False)
    monkeypatch.chdir(tmp_path)
    expected = Path(__file__).resolve().parents[1]
    assert bundle_root() == expected
    assert data_root() == expected


@pytest.mark.parametrize("platform, variable", [("win32", "LOCALAPPDATA"), ("linux", "XDG_DATA_HOME")])
def test_frozen_paths_separate_bundle_and_user_data(monkeypatch, tmp_path, platform, variable):
    bundle = tmp_path / "readonly_bundle"
    user = tmp_path / "user"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.delenv("ZMSOFTWARE_DATA_DIR", raising=False)
    monkeypatch.setenv(variable, str(user))
    assert bundle_root() == bundle
    assert data_root() == user / "SoftwareTools_PyQt"


@pytest.mark.parametrize("frozen", [False, True])
def test_data_override_supports_isolated_acceptance(monkeypatch, tmp_path, frozen):
    monkeypatch.setattr(sys, "frozen", frozen, raising=False)
    monkeypatch.setenv("ZMSOFTWARE_DATA_DIR", str(tmp_path / "验收数据"))
    assert data_root() == tmp_path / "验收数据"


def test_frozen_services_and_page_never_write_bundle(application, monkeypatch, tmp_path):
    from app.pages.feed_depth_page import FeedDepthPage
    from core.services.feed_depth_service import save_feed_depth_settings

    bundle = tmp_path / "bundle"
    shutil.copytree(bundle_root() / "config", bundle / "config")
    bundle_files = {path.relative_to(bundle): path.read_bytes() for path in bundle.rglob("*") if path.is_file()}
    writable = tmp_path / "用户数据"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(bundle), raising=False)
    monkeypatch.setenv("ZMSOFTWARE_DATA_DIR", str(writable))

    position = PositionMonitoringService()
    spindle = SpindleMonitoringService()
    page = FeedDepthPage()
    assert position.storage == writable / "storage/position_monitoring"
    assert position.parameter_path.is_file()
    assert spindle.root == writable / "storage/spindle_monitoring"
    assert page.settings_path == writable / "storage/feed_depth/settings.json"
    assert page.simulation_root == writable / "data/feed_depth_simulation"
    save_feed_depth_settings(page.settings_path, {
        "depth_mode": "uniform", "uniform_depth": 1.2,
        "error_lower_mm": -0.1, "error_upper_mm": 0.1,
    })
    assert page.settings_path.is_file()
    assert {path.relative_to(bundle): path.read_bytes() for path in bundle.rglob("*") if path.is_file()} == bundle_files


def test_explicit_service_roots_override_runtime_default(monkeypatch, tmp_path):
    monkeypatch.setenv("ZMSOFTWARE_DATA_DIR", str(tmp_path / "unused"))
    position = PositionMonitoringService(root=tmp_path / "robot")
    spindle = SpindleMonitoringService(root=tmp_path / "spindle")
    assert position.root == tmp_path / "robot"
    assert spindle.root == tmp_path / "spindle"
    assert not (tmp_path / "unused").exists()
