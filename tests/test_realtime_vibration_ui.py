"""隔离采集格式输入的实时波形交互回归，不读写主业务数据。"""

from threading import Event
from time import monotonic

import numpy as np
import pytest
from PyQt6.QtCore import QTimer
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication

from app.dialogs import realtime_monitor_dialog
from app.pages.spindle_rotation_page import SpindleRotationPage
from core.services.spindle_monitoring_service import SpindleMonitoringService
from tests.test_vibration_replay import make_run_folder


pytestmark = pytest.mark.usefixtures("styled_application")


@pytest.fixture
def monitor(application, tmp_path, monkeypatch):
    monkeypatch.setattr(realtime_monitor_dialog.QMessageBox, "warning", lambda *args: None)
    page = SpindleRotationPage(SpindleMonitoringService(tmp_path / "spindle_store"))
    page.ui_scale = 1.0
    page.resize(1500, 1000)
    page.show()
    QApplication.processEvents()
    page.realtime_button.click()
    dialog = page.realtime_dialog
    dialog.resize(1400, 1000)
    QTest.qWait(30)
    yield page, dialog
    wait_until(dialog, lambda: dialog.load_task is None)
    dialog.shutdown()
    dialog.close()
    page.close()
    page.deleteLater()
    QApplication.processEvents()


def wait_until(dialog, predicate, timeout=5):
    deadline = monotonic() + timeout
    while not predicate() and monotonic() < deadline and dialog.replay_state != "error":
        QTest.qWait(10)
    assert predicate(), dialog.replay_status.text()


def load_short_recording(dialog, tmp_path, seconds=3):
    folder = make_run_folder(tmp_path, sample_rate=25600, counts=(25600,) * seconds)
    dialog.load_recording(folder)
    wait_until(dialog, lambda: dialog.replay_state == "playing")
    wait_until(dialog, lambda: dialog.replay.samples_consumed > 0)
    return folder


def curve_data(plot):
    segments = [curve.getData() for curve in plot.curves]
    segments = [(x, y) for x, y in segments if x is not None and x.size]
    if not segments:
        return np.empty(0), np.empty(0)
    return np.concatenate([x for x, _ in segments]), np.concatenate([y for _, y in segments])


def test_loading_keeps_gui_responsive_and_replay_follows_real_time(monitor, tmp_path, monkeypatch):
    _, dialog = monitor
    folder = make_run_folder(tmp_path, sample_rate=25600, counts=(25600,))
    original_load = realtime_monitor_dialog.load_vibration_recording
    entered, release = Event(), Event()
    gui_ticks = []

    def delayed_load(folder, progress):
        entered.set()
        assert release.wait(5)
        return original_load(folder, progress)

    monkeypatch.setattr(realtime_monitor_dialog, "load_vibration_recording", delayed_load)
    try:
        dialog.load_recording(folder)
        QTimer.singleShot(20, lambda: gui_ticks.append(monotonic()))
        QTest.qWait(80)
        assert entered.is_set() and gui_ticks
        assert dialog.replay_state == "loading" and dialog.load_task is not None
        assert not dialog.simulate_button.isEnabled()
    finally:
        release.set()

    wait_until(dialog, lambda: dialog.replay_state == "playing")
    wait_until(dialog, lambda: dialog.replay.samples_consumed > 0)
    assert dialog.recording.sample_rate_hz == 25600
    first_count = dialog.replay.samples_consumed
    started = monotonic()
    QTest.qWait(300)
    elapsed = monotonic() - started
    advanced_s = (dialog.replay.samples_consumed - first_count) / 25600
    assert advanced_s == pytest.approx(elapsed, abs=0.12)
    assert dialog.replay_timer.isActive()
    for plot in dialog.wave_plots.values():
        x, y = curve_data(plot)
        assert x.size > 0 and x[-1] == 0
        assert np.isfinite(y).all() and np.ptp(y) > 0

    wait_until(dialog, lambda: dialog.replay_state == "finished")
    assert dialog.replay.samples_consumed == dialog.recording.total_samples == 25600
    assert not dialog.replay_timer.isActive()


def test_pause_close_resume_and_stop_control_the_same_replay(monitor, tmp_path):
    page, dialog = monitor
    load_short_recording(dialog, tmp_path)
    QTest.qWait(180)
    dialog.simulate_button.click()
    paused_count = dialog.replay.samples_consumed
    assert dialog.replay_state == "paused" and not dialog.replay_timer.isActive()
    QTest.qWait(150)
    assert dialog.replay.samples_consumed == paused_count

    dialog.simulate_button.click()
    QTest.qWait(180)
    assert dialog.replay_state == "playing"
    assert dialog.replay.samples_consumed > paused_count
    dialog.close()
    closed_count = dialog.replay.samples_consumed
    assert dialog.replay_state == "paused" and not dialog.isVisible()
    QTest.qWait(150)
    assert dialog.replay.samples_consumed == closed_count

    page.realtime_button.click()
    assert page.realtime_dialog is dialog and dialog.isVisible()
    assert dialog.replay_state == "paused"
    dialog.simulate_button.click()
    QTest.qWait(180)
    assert dialog.replay.samples_consumed > closed_count
    dialog.stop_button.click()
    assert dialog.replay_state == "empty" and dialog.replay is None
    assert not dialog.replay_timer.isActive() and not dialog.stop_button.isEnabled()


def test_window_switches_preserve_samples_and_use_fixed_measured_ranges(monitor, tmp_path):
    _, dialog = monitor
    assert [dialog.window_select.itemData(index) for index in range(dialog.window_select.count())] == [0.5, 1, 5, 10]
    assert dialog.window_select.currentData() == 5
    load_short_recording(dialog, tmp_path)
    QTest.qWait(1100)
    dialog.simulate_button.click()
    counts = {key: {window: buffer.count for window, buffer in plot.buffers.items()}
              for key, plot in dialog.wave_plots.items()}
    assert all(count > 0 for channel in counts.values() for count in channel.values())

    for window in (10, 1, 0.5, 5):
        dialog.window_select.setCurrentIndex(dialog.window_select.findData(window))
        QTest.qWait(30)
        assert dialog.time_axis.window_s == window
        assert dialog.trend_time_axis.window_s == window
        assert dialog.time_axis.reference_plot.viewRange()[0] == pytest.approx((-window, 0))
        for channel, (key, plot) in enumerate(dialog.wave_plots.items()):
            x_range, y_range = plot.plot.viewRange()
            assert x_range == pytest.approx((-window, 0))
            assert y_range == pytest.approx(dialog.recording.ranges[window][channel])
            assert {size: buffer.count for size, buffer in plot.buffers.items()} == counts[key]
            x, y = curve_data(plot)
            assert x.size == y.size == counts[key][window]
            assert x[-1] == 0 and np.isfinite(y).all()
            assert -window <= x[0] < x[-1] and x.size <= 4096
            if window <= 1:
                assert plot.buffer.count == plot.buffer.capacity

    ranges = {key: plot.plot.viewRange()[1] for key, plot in dialog.wave_plots.items()}
    dialog.simulate_button.click()
    QTest.qWait(250)
    for key, plot in dialog.wave_plots.items():
        assert plot.buffer.count > counts[key][5]
        assert plot.plot.viewRange()[1] == pytest.approx(ranges[key])


def test_loaded_plots_fit_and_reselected_channel_shows_current_samples(monitor, tmp_path):
    _, dialog = monitor
    load_short_recording(dialog, tmp_path)
    QTest.qWait(150)
    assert dialog.plot_scroll.horizontalScrollBar().maximum() == 0
    assert dialog.plot_scroll.verticalScrollBar().maximum() == 0
    assert all(dialog.plot_canvas.rect().contains(frame.geometry())
               for frame in dialog.plot_frames.values())
    for width, height in ((1400, 1000), (1120, 750), (700, 700)):
        dialog.resize(width, height)
        QTest.qWait(30)
        frame_heights = [dialog.plot_frames[f"{key}_waveform"].height() for key in dialog.wave_plots]
        drawing_heights = [plot.plot.getViewBox().height() for plot in dialog.wave_plots.values()]
        assert max(frame_heights) - min(frame_heights) <= 1
        assert max(drawing_heights) - min(drawing_heights) <= 1
        for key, plot in dialog.wave_plots.items():
            axis = plot.plot.getAxis("bottom")
            assert not axis.style["showValues"] and not axis.label.isVisible()
            frame = dialog.plot_frames[f"{key}_waveform"]
            assert not dialog.time_axis.geometry().intersects(frame.geometry())
            assert dialog.time_axis.geometry().top() > frame.geometry().bottom()
        assert dialog.time_axis.reference_plot is dialog.wave_plots["acc3"].plot
    dialog.resize(1400, 1000)
    QTest.qWait(30)

    plot = dialog.wave_plots["acc2"]
    previous_count = curve_data(plot)[0].size
    dialog.channel_checks["acc2"].setChecked(False)
    for kind in ("waveform", "spectrum"):
        frame = dialog.plot_frames[f"acc2_{kind}"]
        assert frame.isHidden() and dialog.plot_grid.indexOf(frame) == -1
    QTest.qWait(200)
    assert plot.buffer.count > previous_count
    dialog.simulate_button.click()
    assert dialog.replay_state == "paused"

    dialog.channel_checks["acc2"].setChecked(True)
    QTest.qWait(30)
    assert plot.isVisible()
    for kind in ("waveform", "spectrum"):
        assert dialog.plot_grid.indexOf(dialog.plot_frames[f"acc2_{kind}"]) >= 0
    x, y = curve_data(plot)
    expected = plot.buffer.display_segments()
    assert x.size == y.size == plot.buffer.count
    np.testing.assert_array_equal(x, np.concatenate([time for time, _ in expected]))
    np.testing.assert_array_equal(y, np.concatenate([values for _, values in expected]))

    dialog.channel_checks["acc3"].setChecked(False)
    QTest.qWait(30)
    assert dialog.time_axis.reference_plot is dialog.wave_plots["acc1"].plot
    assert dialog.plot_grid.getItemPosition(dialog.plot_grid.indexOf(dialog.time_axis)) == (3, 0, 1, 2)
    dialog.channel_checks["acc2"].setChecked(False)
    QTest.qWait(30)
    assert dialog.time_axis.reference_plot is dialog.wave_plots["acc1"].plot
    assert dialog.plot_grid.getItemPosition(dialog.plot_grid.indexOf(dialog.time_axis)) == (2, 0, 1, 2)
    dialog.waveform_check.setChecked(False)
    QTest.qWait(30)
    assert dialog.time_axis.isHidden() and dialog.plot_grid.indexOf(dialog.time_axis) == -1
