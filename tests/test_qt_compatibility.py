"""麒麟所用 Qt 6.7 与 Windows Qt 6.11 共用模型文字和星标委托。"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox

from app.pages.spindle_rotation_page import CurrentModelDelegate, ModelComboBox


def test_model_combo_paints_current_marker_without_qt_69_api(application, monkeypatch):
    monkeypatch.delattr(QComboBox, "setLabelDrawingMode", raising=False)
    combo = ModelComboBox()
    combo.setItemDelegate(CurrentModelDelegate(combo))
    combo.addItem("当前模型 v1", "v1")
    combo.setItemData(0, True, Qt.ItemDataRole.UserRole + 1)
    painted = []
    original = CurrentModelDelegate.paint

    def capture(self, painter, option, index):
        painted.append((index.data(), index.data(Qt.ItemDataRole.UserRole + 1)))
        original(self, painter, option, index)

    monkeypatch.setattr(CurrentModelDelegate, "paint", capture)
    combo.resize(260, 36)
    combo.show()
    application.processEvents()
    assert not combo.grab().isNull()
    assert ("当前模型 v1", True) in painted
    combo.setItemData(0, False, Qt.ItemDataRole.UserRole + 1)
    combo.grab()
    assert ("当前模型 v1", False) in painted
