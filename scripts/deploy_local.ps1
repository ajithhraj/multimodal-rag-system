param(
    [int]$Port = 8000,
    [string]$BindHost = "0.0.0.0",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repoRoot

$envPath = Join-Path $repoRoot ".env"
if (-not (Test-Path $envPath)) {
    Copy-Item (Join-Path $repoRoot ".env.example") $envPath
}

$stateDir = Join-Path $repoRoot ".rag_store"
if (-not (Test-Path $stateDir)) {
    New-Item -ItemType Directory -Path $stateDir | Out-Null
}

$pidFile = Join-Path $stateDir "mmrag.pid"
$stdoutLog = Join-Path $stateDir "deploy_stdout.log"
$stderrLog = Join-Path $stateDir "deploy_stderr.log"

$existingListen = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existingListen) {
    Write-Host "Port $Port already has a listening process. Run scripts/stop_local.ps1 first." -ForegroundColor Yellow
    exit 1
}

if (-not $SkipInstall) {
    python -m pip install -e .
}

$proc = Start-Process `
    -FilePath "python" `
    -ArgumentList "-m", "multimodal_rag.cli", "serve", "--host", $BindHost, "--port", "$Port" `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdoutLog `
    -RedirectStandardError $stderrLog `
    -PassThru

Set-Content -Path $pidFile -Value "$($proc.Id)"

$healthUri = "http://127.0.0.1:$Port/health"
$ready = $false
for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Milliseconds 500
    try {
        $health = Invoke-RestMethod -Uri $healthUri -Method Get -TimeoutSec 2
        if ($health.status -eq "ok") {
            $ready = $true
            break
        }
    } catch {
        # keep polling until timeout
    }
}

if (-not $ready) {
    Write-Host "Deployment started but health check failed at $healthUri" -ForegroundColor Red
    Write-Host "Check logs:"
    Write-Host "  $stdoutLog"
    Write-Host "  $stderrLog"
    exit 1
}

$listener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
if ($listener) {
    Set-Content -Path $pidFile -Value "$($listener.OwningProcess)"
}

Write-Host "Deployment successful." -ForegroundColor Green
Write-Host "PID: $(Get-Content -Path $pidFile | Select-Object -First 1)"
Write-Host "Health: $healthUri"
Write-Host "Docs: http://127.0.0.1:$Port/docs"
