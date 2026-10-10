"""独立相机与手眼标定命令行入口；实现与桌面共用。"""

import argparse

from core.services.camera_calibration import calibrate_dataset


def main():
    parser = argparse.ArgumentParser(description="独立 ChArUco 相机内参与手眼标定")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-id", default="B000")
    parser.add_argument("--intrinsic-mode", choices=("calibrate", "reference"), default="calibrate")
    parser.add_argument("--hand-eye-mode", choices=("calibrate", "reference"), default="calibrate")
    parser.add_argument("--distortion-mode", choices=("estimate", "zero"), default="estimate")
    parser.add_argument("--method", choices=("PARK", "TSAI"), default="PARK")
    result = calibrate_dataset(**vars(parser.parse_args()), progress=lambda value, message: print(f"{value}% {message}"))
    print(result["report_path"])


if __name__ == "__main__":
    main()
