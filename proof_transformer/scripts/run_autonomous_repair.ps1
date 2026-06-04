$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $env:AUSTIN_AI_BRAIN_ROOT) { $env:AUSTIN_AI_BRAIN_ROOT = 'E:\AUSTIN_AI_BRAIN' }
$env:PYTHONPATH = Join-Path $RepoRoot 'src'
Push-Location $RepoRoot
try {
  if ($args.Length -lt 2) { throw 'Usage: .\\scripts\\run_autonomous_repair.ps1 WORKSPACE COMMAND [ARGS...]' }
  $workspace = $args[0]
  $command = @()
  if ($args.Length -gt 1) { $command = $args[1..($args.Length-1)] }
  python -m proof_transformer.cli autonomous-repair $workspace -- @command
} finally {
  Pop-Location
}
