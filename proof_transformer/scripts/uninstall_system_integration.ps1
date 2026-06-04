$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$BinPath = Join-Path $RepoRoot 'bin'

$userPath = [Environment]::GetEnvironmentVariable('Path', 'User')
$parts = @()
if ($userPath) {
  $parts = $userPath -split ';' | Where-Object { $_ -and $_ -ne $BinPath }
  [Environment]::SetEnvironmentVariable('Path', ($parts -join ';'), 'User')
}

$taskName = 'AustinBrainLocalServer'
try {
  Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction Stop
} catch {
}

[pscustomobject]@{
  repo_root = $RepoRoot
  bin_path_removed = $BinPath
  startup_server_removed = $true
} | ConvertTo-Json -Depth 4
