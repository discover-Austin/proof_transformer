$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $env:AUSTIN_AI_BRAIN_ROOT) { $env:AUSTIN_AI_BRAIN_ROOT = 'E:\AUSTIN_AI_BRAIN' }
$env:PYTHONPATH = Join-Path $RepoRoot 'src'
Push-Location $RepoRoot
try {
  python -m proof_transformer.cli serve-api --host 127.0.0.1 --port 8765
} finally {
  Pop-Location
}
