param([switch]$IncludeFirmware)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Missing virtual environment: $python"
}

Push-Location $root
try {
    & $python -m pytest -q
    if ($LASTEXITCODE -ne 0) {
        throw "Python tests failed"
    }

    & $python -m ruff check src tests
    if ($LASTEXITCODE -ne 0) {
        throw "Ruff checks failed"
    }

    & $python -m preact.trajectory_validation --scenarios 300
    if ($LASTEXITCODE -ne 0) {
        throw "Trajectory validation gate failed"
    }

    if (-not $IncludeFirmware) {
        Write-Host "PREACT Python verification passed. Use -IncludeFirmware for Arduino compilation." -ForegroundColor Green
        return
    }

    $arduino = Get-Command arduino-cli -ErrorAction SilentlyContinue
    if ($arduino) {
        $arduinoPath = $arduino.Source
    } else {
        $arduinoPath = "C:\Program Files\Arduino CLI\arduino-cli.exe"
    }
    if (-not (Test-Path -LiteralPath $arduinoPath)) {
        throw "Arduino CLI is not installed"
    }

    & $arduinoPath compile `
        --fqbn esp32:esp32:axiometa_genesis_mini `
        --output-dir artifacts\firmware-build `
        firmware\genesis_mini\preact_genesis
    if ($LASTEXITCODE -ne 0) {
        throw "Genesis firmware compilation failed"
    }

    Write-Host "PREACT verification passed." -ForegroundColor Green
} finally {
    Pop-Location
}
