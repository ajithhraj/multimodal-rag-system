param(
    [int]$Port = 8000,
    [string]$BindHost = "0.0.0.0",
    [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"

$stopScript = Join-Path $PSScriptRoot "stop_local.ps1"
$deployScript = Join-Path $PSScriptRoot "deploy_local.ps1"

& $stopScript -Port $Port
Start-Sleep -Seconds 1

if ($SkipInstall) {
    & $deployScript -Port $Port -BindHost $BindHost -SkipInstall
} else {
    & $deployScript -Port $Port -BindHost $BindHost
}
