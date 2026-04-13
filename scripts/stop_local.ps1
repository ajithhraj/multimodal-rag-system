param(
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$pidFile = Join-Path $repoRoot ".rag_store\mmrag.pid"

$stopped = $false

if (Test-Path $pidFile) {
    $pidRaw = Get-Content -Path $pidFile -ErrorAction SilentlyContinue
    [int]$targetPid = 0
    if ([int]::TryParse(($pidRaw | Select-Object -First 1), [ref]$targetPid) -and $targetPid -gt 0) {
        $proc = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
        if ($proc) {
            Stop-Process -Id $targetPid -Force
            Write-Host "Stopped process from PID file: $targetPid"
            $stopped = $true
        }
    }
    Remove-Item $pidFile -ErrorAction SilentlyContinue
}

$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) {
    $ownerPid = $listener.OwningProcess
    $ownerProc = Get-Process -Id $ownerPid -ErrorAction SilentlyContinue
    if ($ownerProc) {
        Stop-Process -Id $ownerPid -Force
        Write-Host "Stopped process listening on port ${Port}: $ownerPid"
        $stopped = $true
    }
}

if (-not $stopped) {
    Write-Host "No running local deployment process found."
}
