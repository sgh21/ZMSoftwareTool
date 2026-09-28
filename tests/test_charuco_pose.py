"""ChArUco 像素检测、ID 对应、毫米原点和部分可见回归。"""

import cv2
import numpy as np
import pytest

from core.algorithms.board_pose import _solve_charuco_pose, estimate_charuco_pose


CHARUCO = {
    "dictionary": "DICT_5X5_1000", "squares_xy": [9, 7], "marker_length_mm": 14,
}
CAMERA = np.array([[1200, 0, 500], [0, 1200, 400], [0, 0, 1]], dtype=float)


def make_board(spec=CHARUCO):
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["dictionary"]))
    marker_ids = spec.get("marker_ids")
    board = cv2.aruco.CharucoBoard(
        tuple(spec["squares_xy"]), 20, spec["marker_length_mm"], dictionary,
        None if marker_ids is None else np.asarray(marker_ids, dtype=np.int32),
    )
    board.setLegacyPattern(spec.get("legacy_pattern", False))
    return board


def board_image(spec=CHARUCO):
    width, height = (100 * value for value in spec["squares_xy"])
    pattern = make_board(spec).generateImage((width, height), marginSize=0, borderBits=1)
    return np.pad(pattern, 50, constant_values=255)


@pytest.mark.parametrize("color", [False, True])
def test_generated_image_recovers_top_left_frame_in_mm(color):
    image = board_image()
    if color:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    result = estimate_charuco_pose(image, CAMERA, np.zeros(5), CHARUCO, 20)
    assert result["corner_count"] == 48
    assert result["marker_count"] == 31
    assert sorted(result["charuco_corner_ids"]) == list(range(48))
    assert result["board_origin"] == "charuco_top_left"
    assert result["corner_order"] == "charuco_ids"
    assert result["translation_unit"] == "mm"
    # 100 px per 20 mm square at f=1200 px gives depth 240 mm.
    np.testing.assert_allclose(result["vision_pose"][:3, 3], [-90, -70, 240], atol=0.15)
    np.testing.assert_allclose(result["vision_pose"][:3, :3], np.eye(3), atol=0.001)
    assert result["reprojection_error_px"] < 0.15


@pytest.mark.parametrize("ids", [np.arange(48)[::-1], np.array([47, 0, 7, 40])])
def test_known_mixed_pose_uses_ids_instead_of_image_corner_order(ids):
    board = make_board()
    rotvec = np.array([0.3, -0.2, 2.7])
    translation = np.array([80, 30, 700.0])
    distortion = np.array([-0.12, 0.03, 0.001, -0.002, 0.005])
    points = cv2.projectPoints(
        board.getChessboardCorners()[ids].astype(float), rotvec, translation, CAMERA, distortion,
    )[0]
    result = _solve_charuco_pose(points, ids, board, CAMERA, distortion)
    np.testing.assert_allclose(result["vision_pose"][:3, :3], cv2.Rodrigues(rotvec)[0], atol=1e-7)
    np.testing.assert_allclose(result["vision_pose"][:3, 3], translation, atol=1e-6)
    assert result["charuco_corner_ids"] == ids.tolist()
    assert result["reprojection_error_px"] < 1e-7


def test_partially_occluded_board_remains_measurable():
    image = board_image()
    image[:, :500] = 255
    result = estimate_charuco_pose(image, CAMERA, np.zeros(5), CHARUCO, 20)
    assert 4 <= result["corner_count"] < 48
    assert len(result["charuco_corner_ids"]) == result["corner_count"]
    np.testing.assert_allclose(result["vision_pose"][:3, 3], [-90, -70, 240], atol=0.2)
    assert result["reprojection_error_px"] < 0.15


@pytest.mark.parametrize("legacy", [False, True])
def test_custom_marker_ids_and_even_row_legacy_pattern(legacy):
    spec = {**CHARUCO, "squares_xy": [9, 8], "marker_ids": list(range(100, 136)),
            "legacy_pattern": legacy}
    image = board_image(spec)
    result = estimate_charuco_pose(image, CAMERA, np.zeros(5), spec, 20)
    assert result["corner_count"] == 56
    assert sorted(result["marker_ids"]) == list(range(100, 136))
    np.testing.assert_allclose(result["vision_pose"][:3, 3], [-90, -70, 240], atol=0.15)


@pytest.mark.parametrize("ids, message", [
    ([0, 1, 2], "至少 4"),
    ([0, 1, 2, 3], "共线"),
    ([0, 0, 8, 9], "不重复"),
    ([0, 7, 40, 48], "有效"),
])
def test_insufficient_or_degenerate_correspondences_fail(ids, message):
    points = np.array([[100, 100], [200, 100], [100, 200], [200, 200]], dtype=float)[:len(ids)]
    with pytest.raises(ValueError, match=message):
        _solve_charuco_pose(points, ids, make_board(), CAMERA, np.zeros(5))


def test_wrong_dictionary_and_blank_image_do_not_return_a_pose():
    with pytest.raises(ValueError, match="未检测到"):
        estimate_charuco_pose(board_image(), CAMERA, np.zeros(5),
                              {**CHARUCO, "dictionary": "DICT_4X4_50"}, 20)
    with pytest.raises(ValueError, match="未检测到"):
        estimate_charuco_pose(np.full((800, 1000), 255, dtype=np.uint8),
                              CAMERA, np.zeros(5), CHARUCO, 20)


@pytest.mark.parametrize("changes, message", [
    ({"dictionary": "DICT_UNKNOWN"}, "不支持"),
    ({"squares_xy": [9, 1]}, "至少 2"),
    ({"squares_xy": [9.5, 7]}, "整数"),
    ({"marker_length_mm": 20}, "小于方格边长"),
    ({"marker_ids": [0, 1]}, "编号与方格配置不匹配"),
    ({"marker_ids": [0] * 31}, "互不重复"),
    ({"marker_ids": list(range(970, 1001))}, "字典范围"),
    ({"marker_ids": [0.5, *range(1, 31)]}, "编号与方格配置不匹配"),
    ({"legacy_pattern": "false"}, "布尔值"),
])
def test_invalid_board_configuration_has_clear_error(changes, message):
    with pytest.raises(ValueError, match=message):
        estimate_charuco_pose(board_image(), CAMERA, np.zeros(5), {**CHARUCO, **changes}, 20)
