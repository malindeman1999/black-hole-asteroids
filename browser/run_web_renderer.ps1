param(
    [string]$PythonExe = "python",
    [switch]$SkipInstall,
    [switch]$OpenBrowser,
    [string]$BindHost = "127.0.0.1",
    [int]$Port = 8000,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

if ($Help) {
    Write-Host "Usage:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\\run_web_renderer.ps1 [-SkipInstall] [-OpenBrowser] [-PythonExe <path>] [-BindHost <ip>] [-Port <n>]"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\\run_web_renderer.ps1 -OpenBrowser"
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\\run_web_renderer.ps1 -SkipInstall -Port 8080"
    exit 0
}

function Resolve-PrecomputePath {
    $candidates = @(
        "data/earliest_angles_precompute_10rs.npz",
        "earliest_angles_precompute_10rs.npz",
        "tests/earliest_angles_precompute_10rs.npz"
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) {
            return $p
        }
    }
    return $null
}

Write-Host "Starting web renderer launcher..."

if (-not $SkipInstall) {
    Write-Host "Installing/updating required packages (fastapi, uvicorn)..."
    & $PythonExe -m pip install --upgrade fastapi uvicorn
}

$precompute = Resolve-PrecomputePath
if (-not $precompute) {
    Write-Warning "Could not find earliest_angles_precompute_10rs.npz in data/, project root, or tests/."
    Write-Warning "The server may fail to start until the precompute file is available."
} else {
    Write-Host "Found precompute file: $precompute"
}

$url = "http://$BindHost`:$Port"
if ($OpenBrowser) {
    Write-Host "Opening browser at $url"
    Start-Process $url | Out-Null
}

Write-Host "Running server on $url"
& $PythonExe -m uvicorn web_renderer.server:app --host $BindHost --port $Port
