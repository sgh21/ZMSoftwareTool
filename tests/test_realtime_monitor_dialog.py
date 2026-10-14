"""实时监测占位窗口：独立打开、通道选择和响应式布局。"""

import pytest
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QLabel

from app.pages.spindle_rotation_page import SpindleRotationPage
from core.services.spindle_monitoring_service import SpindleMonitoringService


pytestmark = pytest.mark.usefixtures("styled_application")


def settle_layout():
    QApplication.processEvents()
    QTest.qWait(20)
    QApplication.processEvents()


@pytest.fixture
def page(application, tmp_path):
    widget = SpindleRotationPage(SpindleMonitoringService(tmp_path / "spindle_store"))
    widget.ui_scale = 1.0
    widget.resize(1500, 1000)
    widget.show()
    settle_layout()
    yield widget
    if widget.realtime_dialog is not None:
        widget.realtime_dialog.close()
    widget.close()
    widget.deleteLater()
    QApplication.processEvents()


def open_monitor(page):
    page.realtime_button.click()
    settle_layout()
    return page.realtime_dialog


def grid_positions(dialog):
    keys = {frame: key for key, frame in dialog.plot_frames.items()}
    positions = {}
    for index in range(dialog.plot_grid.count()):
        frame = dialog.plot_grid.itemAt(index).widget()
        if frame in (dialog.empty_state, dialog.time_axis, dialog.trend_time_axis, *dialog.plot_headers.values()):
            continue
        assert frame in keys, "绘图区布局存在不是占位框的残留项"
        assert not frame.isHidden(), "隐藏通道仍占用绘图区布局"
        positions[keys[frame]] = dialog.plot_grid.getItemPosition(index)
    return positions


def test_realtime_entry_keeps_original_plots_and_opens_while_busy(page):
    original_plots = dict(page.plots)
    assert set(original_plots) == {"waveform", "spectrum", "energy", "distribution"}
    assert page.realtime_dialog is None

    page._set_busy(True)
    assert page.realtime_button.isEnabled()
    dialog = open_monitor(page)

    assert dialog.isVisible() and dialog.isWindow()
    assert dialog.parentWidget() is page
    assert not dialog.isModal()
    assert page.plots == original_plots
    assert all(plot.isVisible() for plot in original_plots.values())
    assert all(check.isChecked() for check in dialog.channel_checks.values())
    assert dialog.waveform_check.isChecked() and dialog.spectrum_check.isChecked()
    assert dialog.time_axis.isVisible()
    assert dialog.time_axis.reference_plot is dialog.wave_plots["acc3"].plot
    assert [label.text() for label in dialog.plot_headers.values()] == ["波形", "频谱"]
    assert dialog.trend_time_axis.reference_widget is dialog.plot_contents[dialog.plot_frames["speed"]]
    assert all("\n" in frame.findChild(QLabel, "PlotTitle").text() for frame in dialog.plot_frames.values())
    page._set_busy(False)


def test_realtime_window_is_single_instance_and_keeps_selection_on_reopen(page):
    dialog = open_monitor(page)
    dialog.channel_checks["acc2"].setChecked(False)
    dialog.waveform_check.setChecked(False)
    dialog.showMaximized()
    settle_layout()
    assert dialog.isMaximized()

    dialog.close()
    settle_layout()
    assert not dialog.isVisible()
    assert open_monitor(page) is dialog
    assert not dialog.channel_checks["acc2"].isChecked()
    assert not dialog.waveform_check.isChecked()
    assert dialog.isMaximized()
    assert not dialog.isModal()
    assert open_monitor(page) is dialog


def test_channel_changes_remove_empty_rows_and_restore_all_placeholders(page):
    dialog = open_monitor(page)
    dialog.resize(1300, 800)
    for key in ("acc1", "acc3", "ntc1", "speed"):
        dialog.channel_checks[key].setChecked(False)
    settle_layout()

    positions = grid_positions(dialog)
    assert set(positions) == {"acc2_waveform", "acc2_spectrum", "temperature"}
    assert positions["acc2_waveform"] == (1, 0, 1, 1)
    assert positions["acc2_spectrum"] == (1, 1, 1, 1)
    assert positions["temperature"] == (3, 0, 1, 2)
    assert dialog.plot_grid.getItemPosition(dialog.plot_grid.indexOf(dialog.time_axis)) == (2, 0, 1, 2)
    assert dialog.time_axis.reference_plot is dialog.wave_plots["acc2"].plot
    assert dialog.temperature_legends["ntc1"].isHidden()
    assert dialog.temperature_legends["rtd1"].isVisible()
    assert dialog.trend_time_axis.reference_widget is dialog.plot_contents[dialog.plot_frames["temperature"]]
    assert not dialog.empty_state.isVisible()

    dialog.waveform_check.setChecked(False)
    settle_layout()
    positions = grid_positions(dialog)
    assert positions == {"acc2_spectrum": (1, 0, 1, 2), "temperature": (2, 0, 1, 2)}
    assert dialog.time_axis.isHidden() and dialog.plot_grid.indexOf(dialog.time_axis) == -1

    for check in dialog.channel_checks.values():
        check.setChecked(False)
    settle_layout()
    assert not grid_positions(dialog)
    assert dialog.empty_state.isVisible()
    assert dialog.plot_grid.count() == 1
    assert dialog.plot_grid.itemAt(0).widget() is dialog.empty_state
    assert all(frame.isHidden() for frame in dialog.plot_frames.values())
    assert dialog.trend_time_axis.isHidden() and dialog.plot_grid.indexOf(dialog.trend_time_axis) == -1
    assert all(header.isHidden() for header in dialog.plot_headers.values())

    dialog.channel_checks["ntc1"].setChecked(True)
    settle_layout()
    assert grid_positions(dialog) == {"temperature": (0, 0, 1, 2)}
    assert dialog.plot_frames["temperature"].width() == dialog.plot_canvas.width()
    assert dialog.trend_time_axis.geometry().top() > dialog.plot_frames["temperature"].geometry().bottom()

    dialog.waveform_check.setChecked(True)
    for check in dialog.channel_checks.values():
        check.setChecked(True)
    settle_layout()
    assert set(grid_positions(dialog)) == set(dialog.plot_frames)
    assert not dialog.empty_state.isVisible()


def test_narrow_window_stacks_plots_without_horizontal_overflow(page):
    dialog = open_monitor(page)
    dialog.resize(700, 700)
    settle_layout()

    assert dialog.plot_scroll.viewport().width() < 800
    positions = grid_positions(dialog)
    assert len(positions) == 8
    assert {position[0] for position in positions.values()} == {1, 2, 3, 6, 7, 8, 9, 10}
    assert all(position[1:] == (0, 1, 1) for position in positions.values())
    assert dialog.plot_grid.getItemPosition(dialog.plot_grid.indexOf(dialog.time_axis)) == (4, 0, 1, 1)
    assert dialog.plot_grid.getItemPosition(dialog.plot_grid.indexOf(dialog.trend_time_axis)) == (11, 0, 1, 1)
    assert dialog.plot_scroll.horizontalScrollBar().maximum() == 0
    assert dialog.plot_scroll.widget().width() <= dialog.plot_scroll.viewport().width()

    dialog.resize(1300, 800)
    settle_layout()
    assert dialog.plot_scroll.viewport().width() >= 800
    positions = grid_positions(dialog)
    for key in ("acc1", "acc2", "acc3"):
        waveform = positions[f"{key}_waveform"]
        spectrum = positions[f"{key}_spectrum"]
        assert waveform[0] == spectrum[0]
        assert waveform[1:] == (0, 1, 1)
        assert spectrum[1:] == (1, 1, 1)
    assert positions["temperature"][1:] == (0, 1, 2)
    assert positions["speed"][1:] == (0, 1, 2)
    assert dialog.plot_grid.getItemPosition(dialog.plot_grid.indexOf(dialog.time_axis)) == (4, 0, 1, 2)
    assert dialog.plot_grid.getItemPosition(dialog.plot_grid.indexOf(dialog.trend_time_axis)) == (7, 0, 1, 2)


def test_scale_changes_preserve_selection_and_restore_placeholder_sizes(page):
    dialog = open_monitor(page)
    dialog.channel_checks["acc2"].setChecked(False)
    dialog.channel_checks["ntc1"].setChecked(False)
    dialog.waveform_check.setChecked(False)
    original_heights = {key: frame.minimumHeight() for key, frame in dialog.plot_frames.items()}
    expected_selection = {key: check.isChecked() for key, check in dialog.channel_checks.items()}

    for scale in (0.75, 2.0, 1.0):
        dialog.sync_scale(scale)
        dialog.resize(round(700 * scale), round(700 * scale))
        settle_layout()

        assert {key: check.isChecked() for key, check in dialog.channel_checks.items()} == expected_selection
        assert not dialog.waveform_check.isChecked() and dialog.spectrum_check.isChecked()
        assert dialog.time_axis.isHidden() and dialog.plot_grid.indexOf(dialog.time_axis) == -1
        positions = grid_positions(dialog)
        assert set(positions) == {"acc1_spectrum", "acc3_spectrum", "temperature", "speed"}
        assert all(position[1:] == (0, 1, 1) for position in positions.values())
        assert dialog.plot_scroll.horizontalScrollBar().maximum() == 0
        assert dialog.plot_scroll.widget().width() <= dialog.plot_scroll.viewport().width()
        assert {key: frame.minimumHeight() for key, frame in dialog.plot_frames.items()} == {
            key: round(height * scale) for key, height in original_heights.items()
        }
