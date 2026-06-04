$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $RepoRoot 'src'
Push-Location $RepoRoot
try {
  python -m desktop_commander.cli ui
} finally {
  Pop-Location
}
