@echo off
setlocal
set REPO_ROOT=%~dp0..
if "%AUSTIN_AI_BRAIN_ROOT%"=="" set AUSTIN_AI_BRAIN_ROOT=E:\AUSTIN_AI_BRAIN
set PYTHONPATH=%REPO_ROOT%\src
python -m proof_transformer.cli %*
