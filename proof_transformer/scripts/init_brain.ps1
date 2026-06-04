$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
if (-not $env:AUSTIN_AI_BRAIN_ROOT) { $env:AUSTIN_AI_BRAIN_ROOT = 'E:\AUSTIN_AI_BRAIN' }
$env:PYTHONPATH = Join-Path $RepoRoot 'src'
python -m proof_transformer.cli init
