[CmdletBinding(PositionalBinding = $false)]
param(
  [string]$Name = 'command',
  [string]$StatusPath = '',
  [string]$LogPath = '',
  [string]$CommandLine = ''
)

$ErrorActionPreference = 'Stop'

if (-not $CommandLine.Trim()) {
  throw 'Usage: scripts\run_with_completion_signal.ps1 [-Name name] -CommandLine "<command line>"'
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$SignalRoot = Join-Path ([System.IO.Path]::GetTempPath()) 'proof_transformer-codex-signals'
New-Item -ItemType Directory -Force -Path $SignalRoot | Out-Null

$safeName = $Name -replace '[^A-Za-z0-9_.-]', '_'
$stamp = Get-Date -Format 'yyyyMMddTHHmmss'
if (-not $StatusPath) {
  $StatusPath = Join-Path $SignalRoot "$stamp-$safeName.done.json"
}
if (-not $LogPath) {
  $LogPath = Join-Path $SignalRoot "$stamp-$safeName.log"
}

$started = Get-Date
$exitCode = 0
$errorText = $null

Push-Location $RepoRoot
try {
  & $env:ComSpec /d /c $CommandLine 2>&1 | Tee-Object -FilePath $LogPath
  if ($null -ne $global:LASTEXITCODE) {
    $exitCode = [int]$global:LASTEXITCODE
  }
} catch {
  $exitCode = 1
  $errorText = $_.Exception.Message
  $_ | Out-String | Tee-Object -FilePath $LogPath -Append | Write-Host
} finally {
  Pop-Location
  $finished = Get-Date
  $status = [ordered]@{
    name = $Name
    command_line = $CommandLine
    cwd = $RepoRoot
    started_at = $started.ToUniversalTime().ToString('o')
    finished_at = $finished.ToUniversalTime().ToString('o')
    elapsed_seconds = [math]::Round(($finished - $started).TotalSeconds, 3)
    exit_code = $exitCode
    error = $errorText
    log_path = $LogPath
    status_path = $StatusPath
  }
  $status | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $StatusPath -Encoding UTF8
  Write-Host "COMPLETION_SIGNAL status_path=$StatusPath exit_code=$exitCode elapsed_seconds=$($status.elapsed_seconds)"
}

exit $exitCode
