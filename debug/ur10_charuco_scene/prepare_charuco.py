"""Generate metric ChArUco textures with OpenCV (no hand-drawn markers)."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

CHARUCO = json.loads(
    Path(__file__).with_name("scene_config.json").read_text(encoding="utf-8")
)["charuco"]


def make_board(marker_start):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
    columns, rows = CHARUCO["squares_xy"]
    ids = np.arange(marker_start, marker_start + columns * rows // 2, dtype=np.int32)
    return cv2.aruco.CharucoBoard(
        (columns, rows), CHARUCO["square_length_m"], CHARUCO["marker_length_m"], dictionary, ids
    )


def generate(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "generator": "cv2.aruco.CharucoBoard.generateImage",
        "opencv_version": cv2.__version__,
        "dictionary": "DICT_5X5_1000",
        "squares_xy": CHARUCO["squares_xy"],
        "square_length_m": CHARUCO["square_length_m"],
        "marker_length_m": CHARUCO["marker_length_m"],
        "pattern_size_m": [n * CHARUCO["square_length_m"] for n in CHARUCO["squares_xy"]],
        "image_size_px": [3600, 2800],
        "margin_size_px": 0,
        "marker_border_bits": 1,
        "legacy_pattern": False,
        "coordinates": "Pattern origin: top left, +x right, +y down; lengths in metres.",
        "mounting": "PNG contains the pattern only; use a larger physical white substrate.",
        "boards": [],
    }
    board = make_board(0)
    texture = board.generateImage((3600, 2800), marginSize=0, borderBits=1)
    output_path = output_dir / "board_01.png"
    if not cv2.imwrite(str(output_path), texture):
        raise OSError(f"Could not write {output_path}")
    charuco_corners, charuco_ids, _, marker_ids = cv2.aruco.CharucoDetector(
        board
    ).detectBoard(texture)
    detected_marker_ids = sorted(marker_ids.reshape(-1).tolist())
    expected_marker_ids = board.getIds().reshape(-1).tolist()
    expected_corners = len(board.getChessboardCorners())
    if detected_marker_ids != expected_marker_ids or len(charuco_ids) != expected_corners:
        raise RuntimeError("Generated board self-check failed: board_01")
    metadata["boards"].append(
        {
            "board_id": "board_01",
            "image": output_path.name,
            "marker_ids": expected_marker_ids,
            "chessboard_corners_m": board.getChessboardCorners().tolist(),
            "self_check": {
                "detected_markers": len(marker_ids),
                "detected_charuco_corners": len(charuco_corners),
                "passed": True,
            },
        }
    )
    (output_dir / "boards.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"Created 1 OpenCV ChArUco texture in {output_dir}")
    print(
        f"Board: {len(marker_ids)} / {len(expected_marker_ids)} markers and "
        f"{len(charuco_ids)} / {expected_corners} ChArUco corners detected."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).parent / "assets" / "charuco"
    )
    generate(parser.parse_args().output)
