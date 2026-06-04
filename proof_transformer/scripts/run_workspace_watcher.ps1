$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $env:AUSTIN_AI_BRAIN_ROOT) { $env:AUSTIN_AI_BRAIN_ROOT = 'E:\AUSTIN_AI_BRAIN' }
$env:PYTHONPATH = Join-Path $RepoRoot 'src'
Push-Location $RepoRoot
try {
  if ($args.Length -lt 1) { throw 'Usage: .\\scripts\\run_workspace_watcher.ps1 PATH [extra args]' }
  $workspace = $args[0]
  $extra = @()
  if ($args.Length -gt 1) { $extra = $args[1..($args.Length-1)] }
  python -m proof_transformer.cli watch-workspace $workspace --poll-interval 2.0 @extra
} finally {
  Pop-Location
}
