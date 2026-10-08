"""Check a real Blender camera render against ChArUco and scene geometry."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from prepare_charuco import make_board


def verify(geometry_path, image_path, output_path, annotate):
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise OSError(f"Could not read {image_path}")
    height, width = frame.shape[:2]
    target_id = geometry.get("target_board_id", "board_01")
    target = next(board for board in geometry["boards"] if board["board_id"] == target_id)
    board = make_board(min(target["marker_ids"]))
    board_width, board_height = np.asarray(board.getChessboardSize()) * board.getSquareLength()
    charuco_corners, charuco_ids, marker_corners, marker_ids = (
        cv2.aruco.CharucoDetector(board).detectBoard(frame)
    )
    detected_marker_ids = [] if marker_ids is None else marker_ids.reshape(-1).tolist()
    detected_charuco_ids = [] if charuco_ids is None else charuco_ids.reshape(-1).tolist()
    expected_marker_ids = board.getIds().reshape(-1).tolist()
    target_marker_ids = sorted(set(detected_marker_ids) & set(expected_marker_ids))

    polygon = np.asarray(target["projected_pattern_corners_px"], dtype=np.float32)
    area_px2 = float(abs(cv2.contourArea(polygon)))
    frame_polygon = np.asarray(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    visible_area_px2, _ = cv2.intersectConvexConvex(polygon, frame_polygon)
    coverage = area_px2 / (width * height)
    coverage_target = geometry["camera_model"]["target_pattern_area_fraction"]
    coverage_interval = [coverage_target - 0.03, coverage_target + 0.03]

    def in_frame(corners):
        return bool(np.all(corners >= 0) and np.all(corners <= [width, height]))

    entirely_visible = in_frame(polygon)
    result = {
        "image": str(image_path.resolve()),
        "geometry": str(geometry_path.resolve()),
        "opencv_version": cv2.__version__,
        "target_board_id": target_id,
        "image_size_px": [width, height],
        "expected_image_size_px": geometry["image_size_px"],
        "image_size_matches": [width, height] == geometry["image_size_px"],
        "target_markers_detected": len(target_marker_ids),
        "target_markers_expected": len(expected_marker_ids),
        "detected_target_marker_ids": target_marker_ids,
        "charuco_corners_detected": len(detected_charuco_ids),
        "charuco_corners_expected": 48,
        "projected_pattern_area_px2": area_px2,
        "pattern_area_fraction": coverage,
        "visible_pattern_area_fraction": float(visible_area_px2 / (width * height)),
        "pattern_entirely_in_frame": entirely_visible,
        "coverage_target_fraction": coverage_target,
        "coverage_acceptance_interval": coverage_interval,
        "coverage_matches_target": coverage_interval[0] <= coverage <= coverage_interval[1],
        "coverage_acceptance_note": "Target +/- 0.03 is a framing tolerance, not a measurement accuracy threshold.",
        "coverage_definition": (
            f"Area of the complete {board_width * 1000:g} x {board_height * 1000:g} mm "
            "printed pattern / image area."
        ),
    }
    if "projected_plate_corners_px" in target:
        plate = np.asarray(target["projected_plate_corners_px"], dtype=np.float32)
        result["plate_area_fraction"] = float(abs(cv2.contourArea(plate))) / (width * height)
        visible_plate_area, _ = cv2.intersectConvexConvex(plate, frame_polygon)
        result["visible_plate_area_fraction"] = float(visible_plate_area / (width * height))
        result["projected_plate_corners_px"] = plate.tolist()
        result["plate_entirely_in_frame"] = in_frame(plate)
        result["plate_coverage_definition"] = "White substrate including its border / image area."
    if len(detected_charuco_ids) >= 4:
        object_corners = board.getChessboardCorners()[detected_charuco_ids, :2]
        observed = charuco_corners.reshape(-1, 2)
        homography, _ = cv2.findHomography(object_corners, observed, method=0)
        if homography is not None:
            board_border = np.asarray(
                [[0, 0], [board_width, 0], [board_width, board_height], [0, board_height]],
                dtype=np.float32,
            )
            image_polygon = cv2.perspectiveTransform(board_border[None], homography)[0]
            image_coverage = float(abs(cv2.contourArea(image_polygon))) / (width * height)
            result["image_estimated_pattern_corners_px"] = image_polygon.tolist()
            result["image_estimated_pattern_area_fraction"] = image_coverage
            result["image_estimated_minus_projected_area_fraction"] = image_coverage - coverage
            result["image_area_estimation_note"] = "Homography from detected ChArUco corners and board dimensions; independent of Blender projection."
    roi_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(roi_mask, np.rint(polygon).astype(np.int32), 255)
    roi_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)[roi_mask != 0]
    if roi_gray.size:
        result["pattern_exposure_diagnostic"] = {
            "grayscale_percentiles": dict(zip(["p1", "p50", "p99"], np.percentile(roi_gray, [1, 50, 99]).tolist())),
            "clipped_at_0_fraction": float(np.mean(roi_gray == 0)),
            "clipped_at_255_fraction": float(np.mean(roi_gray == 255)),
            "note": "8-bit grayscale values inside the projected pattern ROI; exposure diagnostic only, not a camera response or accuracy validation.",
        }
    if "projected_charuco_corners_px" in target and detected_charuco_ids:
        expected = np.asarray(target["projected_charuco_corners_px"], dtype=np.float64)
        expected = expected[detected_charuco_ids]
        observed = charuco_corners.reshape(-1, 2).astype(np.float64)
        errors = observed - expected
        radial_errors = np.linalg.norm(errors, axis=1)
        result["corner_projection"] = {
            "rmse_px": float(np.sqrt(np.mean(radial_errors**2))),
            "max_error_px": float(np.max(radial_errors)),
            "mean_error_xy_px": np.mean(errors, axis=0).tolist(),
            "correct_corner_ids_and_orientation": bool(np.max(radial_errors) < 2),
            "acceptance_note": "Raw projection difference below 2 px checks texture orientation and correspondence, not accuracy performance.",
        }
    result["passed"] = bool(
        result["image_size_matches"]
        and result["target_markers_detected"] == len(expected_marker_ids)
        and result["charuco_corners_detected"] == 48
        and entirely_visible
        and result["coverage_matches_target"]
        and result.get("plate_entirely_in_frame", True)
        and result.get("corner_projection", {}).get("correct_corner_ids_and_orientation", True)
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if annotate:
        overlay = frame.copy()
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(overlay, marker_corners, marker_ids)
        if charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(overlay, charuco_corners, charuco_ids)
        cv2.polylines(overlay, [np.rint(polygon).astype(np.int32)], True, (0, 190, 255), 3)
        cv2.putText(
            overlay,
            f"SIMULATED | pattern area {coverage:.1%} | markers {len(target_marker_ids)}/31 | corners {len(detected_charuco_ids)}/48",
            (35, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 80, 220), 2, cv2.LINE_AA,
        )
        if not cv2.imwrite(str(output_path.with_name("wrist_camera_detection.png")), overlay):
            raise OSError("Could not write detection overlay")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result["passed"]


if __name__ == "__main__":
    directory = Path(__file__).parent / "output"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, default=directory / "geometry.json")
    parser.add_argument("--image", type=Path, default=directory / "wrist_camera.png")
    parser.add_argument("--output", type=Path, default=directory / "validation.json")
    parser.add_argument("--annotate", action="store_true")
    args = parser.parse_args()
    raise SystemExit(0 if verify(args.geometry, args.image, args.output, args.annotate) else 1)
