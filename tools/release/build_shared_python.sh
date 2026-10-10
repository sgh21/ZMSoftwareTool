#!/usr/bin/env bash
# manylinux's wheel-build interpreters are static; PyInstaller needs libpython.so.
set -euo pipefail

project_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)
python_version=3.12.15
python_prefix=${PYTHON_PREFIX:-/opt/softwaretools-python}
build_dir="$project_root/build/cpython-$python_version"
report_dir="$project_root/dist/linux-build-report"
mkdir -p "$build_dir" "$report_dir"
source_url="https://www.python.org/ftp/python/$python_version/Python-$python_version.tar.xz"
curl --fail --location "$source_url" --output "$build_dir/Python-$python_version.tar.xz"
tar -xf "$build_dir/Python-$python_version.tar.xz" -C "$build_dir"
cd "$build_dir/Python-$python_version"

printf 'Source: %s\nConfigure: --prefix=%s --enable-shared --with-openssl=/usr --with-ensurepip=install --disable-test-modules\n' \
    "$source_url" "$python_prefix" > "$report_dir/cpython-build-configuration.txt"
./configure --prefix="$python_prefix" --enable-shared --with-openssl=/usr \
    --with-ensurepip=install --disable-test-modules \
    2>&1 | tee "$report_dir/cpython-configure.log"
make -j"$(nproc)" 2>&1 | tee "$report_dir/cpython-compile.log"
LD_LIBRARY_PATH="$PWD" make install 2>&1 | tee "$report_dir/cpython-install.log"
cp "$report_dir/cpython-build-configuration.txt" "$python_prefix/build-configuration.txt"
LD_LIBRARY_PATH="$python_prefix/lib" "$python_prefix/bin/python3.12" - <<'PY'
import ssl
import sys
import sysconfig
assert sysconfig.get_config_var("Py_ENABLE_SHARED") == 1
print(sys.version)
print(ssl.OPENSSL_VERSION)
print("Shared CPython is ready for PyInstaller.")
PY
