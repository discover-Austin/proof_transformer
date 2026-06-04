# Desktop Commander

Standalone desktop task control center for:
- typing in terminals
- searching the web
- opening and reading files
- launching apps and folders
- running local commands
- driving simple desktop interactions on Windows

This app is intentionally separate from `proof_transformer`.

## Run

```powershell
cd D:\AUSTIN_AI_BRAIN_DRIVE\AUSTIN_AI_BRAIN\desktop_commander
$env:PYTHONPATH = 'src'
python -m desktop_commander.cli ui
```

Or:

```powershell
.\scripts\run_desktop_commander.ps1
```

## CLI examples

```powershell
python -m desktop_commander.cli capabilities
python -m desktop_commander.cli draft-plan "open powershell and search the web for Windows terminal shortcuts"
python -m desktop_commander.cli execute --plan-file .\examples\sample_plan.json
python -m desktop_commander.cli serve-api --host 127.0.0.1 --port 8877
```

## API

- `GET /health`
- `GET /capabilities`
- `GET /history`
- `GET /presets`
- `POST /draft-plan`
- `POST /execute`
- `POST /save-preset`

## Notes

- Browser and terminal launching are real local actions.
- Window focus, typing, hotkeys, and clicks require Windows.
- Run history and presets are stored under `runtime_data`.
- The GUI restores the last drafted plan on startup.
- Starter presets are included for web search, PowerShell launch, and opening the runtime folder.
