param(
    [int]$Port = 8000,
    [int]$Tail = 20
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$stateDir = Join-Path $repoRoot ".rag_store"
$pidFile = Join-Path $stateDir "mmrag.pid"
$stdoutLog = Join-Path $stateDir "deploy_stdout.log"
$stderrLog = Join-Path $stateDir "deploy_stderr.log"

Write-Host "MMRAG Local Deployment Status"
Write-Host "Repo: $repoRoot"

if (Test-Path $pidFile) {
    $pidRaw = Get-Content -Path $pidFile -ErrorAction SilentlyContinue | Select-Object -First 1
    Write-Host "PID file: $pidRaw"
    [int]$statusPid = 0
    if ([int]::TryParse(($pidRaw | Out-String).Trim(), [ref]$statusPid) -and $statusPid -gt 0) {
        $proc = Get-Process -Id $statusPid -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "Process: running ($($proc.ProcessName), PID=$($proc.Id))" -ForegroundColor Green
        } else {
            Write-Host "Process: not running (stale PID file)" -ForegroundColor Yellow
        }
    } else {
        Write-Host "Process: unknown (PID file empty or invalid)" -ForegroundColor Yellow
    }
} else {
    Write-Host "PID file: missing"
}

$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) {
    Write-Host "Port ${Port}: listening (PID=$($listener.OwningProcess))" -ForegroundColor Green
} else {
    Write-Host "Port ${Port}: not listening" -ForegroundColor Yellow
}

try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -Method Get -TimeoutSec 3
    Write-Host "Health: $($health.status)" -ForegroundColor Green
} catch {
    Write-Host "Health: unavailable" -ForegroundColor Yellow
}

if (Test-Path $stdoutLog) {
    Write-Host "`n--- stdout (tail $Tail) ---"
    Get-Content -Path $stdoutLog -Tail $Tail
}
if (Test-Path $stderrLog) {
    Write-Host "`n--- stderr (tail $Tail) ---"
    Get-Content -Path $stderrLog -Tail $Tail
}
