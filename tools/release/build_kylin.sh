#!/usr/bin/env bash
# Build on Linux, using an existing CPython 3.12 interpreter; never cross-compile.
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$project_root"
python_bin=${PYTHON:-python3}
report_dir="$project_root/dist/linux-build-report"
mkdir -p "$report_dir"
export PYTHONDONTWRITEBYTECODE=1
export QT_QPA_PLATFORM=offscreen

"$python_bin" - <<'PY'
import platform
import sys
if sys.platform != "linux" or platform.machine() != "x86_64" or sys.version_info[:2] != (3, 12):
    raise SystemExit("Use Linux x86_64 with an existing CPython 3.12 interpreter.")
PY

{
    uname -a
    cat /etc/os-release
    getconf GNU_LIBC_VERSION
    "$python_bin" --version
    git -c safe.directory="$project_root" rev-parse HEAD
    printf 'Target: Kylin V10 Desktop x86_64; target-machine validation is NOT performed here.\n'
} > "$report_dir/build-environment.txt"
if command -v rpm >/dev/null; then
    rpm -qa | sort > "$report_dir/system-packages.txt"
fi

"$python_bin" -m pip install --only-binary=:all: -r tools/release/requirements-kylin.lock \
    2>&1 | tee "$report_dir/dependency-install.log"
"$python_bin" -m pip freeze --all > "$report_dir/python-packages.txt"
"$python_bin" -m pip check 2>&1 | tee "$report_dir/pip-check.txt"

# UI geometry assertions depend on the Windows font environment. Test the same
# algorithms, real import/export services, runtime paths and delivery fixtures.
"$python_bin" -B -m pytest -q -p no:cacheprovider \
    tests/test_board_pose.py tests/test_charuco_pose.py \
    tests/test_camera_calibration_debug.py tests/test_pose_fields.py \
    tests/test_position_monitoring.py tests/test_multidirectional_monitoring.py \
    tests/test_position_independent_verification.py tests/test_position_image_input.py \
    tests/test_position_monitoring_service.py tests/test_position_persistence.py \
    tests/test_parameter_persistence.py tests/test_spindle_algorithms.py \
    tests/test_spindle_monitoring_service.py tests/test_feed_depth_service.py \
    tests/test_feed_depth_simulation.py tests/test_runtime_paths.py \
    tests/test_release_test_data.py \
    --junitxml="$report_dir/source-tests.xml" \
    2>&1 | tee "$report_dir/source-tests.log"

"$python_bin" -m PyInstaller --noconfirm --clean --distpath dist \
    --workpath build/pyinstaller tools/release/SoftwareTools.spec \
    2>&1 | tee "$report_dir/pyinstaller.log"

"$python_bin" -B tools/release/audit_bundle.py dist/SoftwareTools \
    | tee "$report_dir/bundle-audit.json"

# Inspect and launch the exact final executable, outside the repository working
# directory, without Python paths or a developer data directory in its environment.
"$python_bin" -B tools/release/verify_linux_startup.py \
    --exe "$project_root/dist/SoftwareTools/SoftwareTools" \
    --output "$report_dir/final-executable"

# tar preserves executable permissions and symlinks across artifact/ZIP transfers.
tar -czf dist/SoftwareTools-linux-x86_64.tar.gz -C dist SoftwareTools
printf 'Linux build and container startup complete. Kylin V10 hardware acceptance remains pending.\n'
