param(
  [switch]$InstallStartupServer
)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$BinPath = Join-Path $RepoRoot 'bin'
$ApiScript = Join-Path $PSScriptRoot 'start_api_server.ps1'

[Environment]::SetEnvironmentVariable('AUSTIN_AI_BRAIN_ROOT', 'E:\AUSTIN_AI_BRAIN', 'User')

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
$parts = @()
if ($userPath) { $parts = $userPath -split ';' | Where-Object { $_ } }
if ($parts -notcontains $BinPath) {
  $parts += $BinPath
  [Environment]::SetEnvironmentVariable('Path', ($parts -join ';'), 'User')
}

if ($InstallStartupServer) {
  $taskName = 'AustinBrainLocalServer'
  $action = New-ScheduledTaskAction -Execute 'pwsh.exe' -Argument "-NoProfile -File `"$ApiScript`""
  $trigger = New-ScheduledTaskTrigger -AtLogOn
  Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Description 'Starts the Austin Brain local API server at logon.' -User $env:USERNAME -Force | Out-Null
}

[pscustomobject]@{
  repo_root = $RepoRoot
  bin_path = $BinPath
  startup_server_installed = [bool]$InstallStartupServer
} | ConvertTo-Json -Depth 4
