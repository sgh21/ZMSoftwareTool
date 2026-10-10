param(
    [string]$Python = 'python',
    [switch]$InstallDependencies,
    [switch]$SkipTests
)
$ErrorActionPreference = 'Stop'
$projectDir = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
Push-Location $projectDir
$oldPythonPath = $env:PYTHONPATH
try {
    if ($InstallDependencies) {
        & $Python -m pip install -r tools/release/requirements-windows.lock
        if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
    }
    if (-not $SkipTests) {
        & $Python -B -m pytest -q -p no:cacheprovider tests
        if ($LASTEXITCODE -ne 0) { throw 'Source tests failed.' }
    }
    $overlayDir = Join-Path $projectDir 'build/release_deps/windows'
    if (-not (Test-Path -LiteralPath (Join-Path $overlayDir 'torch/__init__.py'))) {
        & $Python -m pip install --no-deps --target $overlayDir --index-url https://download.pytorch.org/whl/cpu 'torch==2.9.0+cpu'
        if ($LASTEXITCODE -ne 0) { throw 'CPU runtime installation failed.' }
    }
    $env:PYTHONPATH = "$overlayDir;$projectDir"
    & $Python -B tools/release/build.py
    if ($LASTEXITCODE -ne 0) { throw 'Windows build failed.' }
} finally {
    $env:PYTHONPATH = $oldPythonPath
    Pop-Location
}
