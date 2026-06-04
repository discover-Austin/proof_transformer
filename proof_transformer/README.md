# Proof-of-Improvement Persistent Transformer

A serious local v0.3 substrate for Austin Grandstaff's **Proof-of-Improvement Persistent Transformer** architecture.

This is not a toy chatbot shell. It is a Windows-first local memory, retrieval, validation, and continual-learning governance stack rooted at **`E:\AUSTIN_AI_BRAIN`**.

## What it is

A CPU-friendly local architecture where:
- SSD = persistent cognitive substrate
- SQLite = shared memory/ledger state
- graph memory + vector memory = externalized attention
- retrieval ranker = first mutable learning target
- validators + ledger = proof-of-improvement gate
- local LLM backend = Ollama, llama.cpp, or deterministic fallback

## What it is not

- not a claim of full self-training neural autonomy
- not a remote SaaS chatbot
- not blind training from every interaction
- not a fake fine-tuning pipeline with TODO placeholders

## Core idea

Every interaction can update memory, but behavior-changing updates are treated as **learning transactions**.
They are only committed after validator consensus demonstrates enough proof of improvement.

## Text architecture diagram

```text
User / CLI
    |
    v
Agent Orchestrator
    |-- Context Compiler
    |     |-- Vector Retrieval (SQLite + deterministic embeddings)
    |     |-- Graph Expansion (SQLite memory graph)
    |     '-- Retrieval Event Log
    |
    |-- LocalLLM
    |     |-- Ollama HTTP backend
    |     |-- llama.cpp HTTP backend
    |     '-- deterministic fallback generator
    |
    |-- Workspace Indexer
    |-- Task Queue
    |-- Learning Event -> Candidate Factory
    |-- Validators
    |     |-- Preference
    |     |-- Contradiction
    |     |-- Scope
    |     |-- Replay
    |     |-- Evidence
    |     '-- Outcome
    |
    '-- Learning Ledger
          |-- policy snapshots
          |-- ranker versions
          |-- adapter registry hooks
          '-- hash-linked learning blocks

Persistent substrate on E:\AUSTIN_AI_BRAIN
    |- memory\brain.sqlite3
    |- memory\documents
    |- memory\vectors
    |- memory\graph
    |- ledger
    |- adapters
    |- checkpoints
    |- logs
    '- sessions
```

## Storage root

Default root:
- `E:\AUSTIN_AI_BRAIN`
- if `E:` is unavailable, the code falls back to the current checked-out `AUSTIN_AI_BRAIN` workspace

Override with:
- `AUSTIN_AI_BRAIN_ROOT`

The code auto-creates:
- `models`
- `memory`
- `memory\documents`
- `memory\vectors`
- `memory\graph`
- `ledger`
- `datasets`
- `evals`
- `checkpoints`
- `adapters`
- `cache`
- `temp`
- `logs`
- `sessions`
- `proof_transformer`

## Implemented components

- SSD-first path control
- SQLite schema + deterministic migrations
- graph memory nodes/edges/subgraph expansion
- code symbol extraction for Python/PowerShell/JS/TS workspaces
- plain text, markdown, and code/text-file ingestion
- deterministic local hashing embeddings
- optional sentence-transformers embedding backend with real local fallback
- SQLite-backed vector search with cosine similarity
- blended retrieval scoring
- bounded context compilation
- hash-linked learning ledger with verification
- policy snapshots
- candidate generation from learning events
- validator consensus engine
- mutable retrieval ranker with feedback training
- adapter/checkpoint registry
- local model routing with real Ollama and llama.cpp clients
- deterministic fallback generation
- workspace indexing, code symbol graphing, queued task execution, queue worker, command execution review, repair planning, local API server, interactive chat
- CLI, examples, PowerShell scripts, and unit tests

## Installation

### Option A: run from source

```powershell
cd E:\AUSTIN_AI_BRAIN\proof_transformer
$env:AUSTIN_AI_BRAIN_ROOT = 'E:\AUSTIN_AI_BRAIN'
$env:PYTHONPATH = "$PWD\src"
python -m proof_transformer.cli init
```

### Option B: local venv on the SSD

```powershell
cd E:\AUSTIN_AI_BRAIN\proof_transformer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
$env:PIP_CACHE_DIR = 'E:\AUSTIN_AI_BRAIN\cache\pip'
python -m pip install --upgrade pip
python -m pip install -e .
proof-transformer init
```

## Initialize the brain

```powershell
cd E:\AUSTIN_AI_BRAIN\proof_transformer
$env:AUSTIN_AI_BRAIN_ROOT = 'E:\AUSTIN_AI_BRAIN'
$env:PYTHONPATH = "$PWD\src"
python -m proof_transformer.cli init
```

Or:

```powershell
.\scripts\init_brain.ps1
```

## Ingest notes

Supported in v0.3:
- `.txt`
- `.md`
- `.markdown`
- `.py`
- `.ps1`
- `.toml`
- `.json`
- `.yaml` / `.yml`
- `.js` / `.ts` / `.tsx` / `.jsx`
- `.sql`
- `.html`
- `.css`
- `.ini` / `.cfg`
- `.bat` / `.sh`

```powershell
python -m proof_transformer.cli ingest .\notes\brain_notes.md
```

## Ask questions

```powershell
python -m proof_transformer.cli ask "What is the first mutable learning target?"
```

## Index a workspace

```powershell
python -m proof_transformer.cli index-workspace E:\path\to\workspace
```

Workspace indexing now builds:
- document chunks + embeddings
- `workspace_file` graph nodes
- `code_symbol` graph nodes
- `belongs_to` edges from symbols to files

## Workspace-aware assist mode

```powershell
python -m proof_transformer.cli assist "How does the ranker update work?" --workspace E:\path\to\workspace --index-workspace
python -m proof_transformer.cli assist "Where is QueueWorker defined?" --workspace E:\path\to\workspace
```

## Watch a workspace for changes

```powershell
python -m proof_transformer.cli watch-workspace E:\path\to\workspace --poll-interval 2.0 --max-cycles 10
.\scripts\run_workspace_watcher.ps1 E:\path\to\workspace --max-cycles 10
```

Workspace watching re-runs incremental indexing, updates changed files, and removes deleted files from active retrieval state.

## Interactive chat

```powershell
python -m proof_transformer.cli chat --workspace E:\path\to\workspace --index-workspace
.\scripts\run_chat.ps1 E:\path\to\workspace
```

## Local API server

```powershell
python -m proof_transformer.cli serve-api --host 127.0.0.1 --port 8765
.\scripts\start_api_server.ps1
```

Available example endpoints:
- `GET /health`
- `GET /status`
- `GET /tasks`
- `GET /executions`
- `GET /repair-plans`
- `GET /desktop/capabilities`
- `POST /ask`
- `POST /assist`
- `POST /run-command`
- `POST /desktop/execute`
- `POST /plan-repair`
- `POST /apply-repair-plan`
- `POST /autonomous-repair`

If `OLLAMA_BASE_URL` or `LLAMA_CPP_BASE_URL` is configured, the system uses that backend.
Otherwise it uses the deterministic local fallback generator and states that explicitly.

## Desktop task execution

The local API and CLI can now drive desktop-style task plans for:
- terminal commands and launching visible processes
- reading, writing, listing, and opening files
- opening URLs and browser search pages
- focusing windows, typing text, sending hotkeys, and clicking on Windows

See the example plan:
- `examples\desktop_plan.json`

Run it from the CLI:

```powershell
python -m proof_transformer.cli desktop-capabilities
python -m proof_transformer.cli desktop-execute --plan-file .\examples\desktop_plan.json
```

Or call it over HTTP:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8765/desktop/execute -ContentType 'application/json' -Body (Get-Content .\examples\desktop_plan.json -Raw)
```

Example action names:
- `run_command`
- `launch_process`
- `open_url`
- `search_web`
- `open_path`
- `read_file`
- `write_file`
- `list_files`
- `focus_window`
- `type_text`
- `send_hotkey`
- `click`
- `wait`

This surface is intentionally local and trust-based. A plan can type into the active desktop, open apps, and modify files, so expose the API only to trusted local callers.

## Record learning events

```powershell
python -m proof_transformer.cli learn --type preference --text "Prefer root-cause-first explanations and no placeholders."
python -m proof_transformer.cli learn --type project_fact --text "Project fact: The ledger is hash-linked and tamper-detecting."
python -m proof_transformer.cli learn --type bug_fix --text "Bug fix: retrieval scores now normalize before ranker updates."
```

## Give retrieval feedback

1. Ask a question.
2. Copy the `retrieval_event_id` and chunk ids from the answer payload.
3. Submit feedback.

```powershell
python -m proof_transformer.cli feedback --event-id RETRIEVAL_ID --useful chunk_a,chunk_b --irrelevant chunk_c
```

This creates a real local ranker candidate, validates it, and only commits a new ranker version if consensus accepts it.

## Install system integration

This repo can be made available from any PowerShell or Command Prompt session by adding its `bin` folder to your user PATH.

```powershell
.\scripts\install_system_integration.ps1
```

That gives you commands like:
- `austin-brain`
- `austin-brain-chat`

To also start the localhost API server automatically at logon:

```powershell
.\scripts\install_system_integration.ps1 -InstallStartupServer
```

To remove the integration later:

```powershell
.\scripts\uninstall_system_integration.ps1
```

## Queue tasks

You can queue work and run it later.

```powershell
python -m proof_transformer.cli enqueue index_workspace --payload '{"path":"E:\\path\\to\\workspace"}'
python -m proof_transformer.cli enqueue ask --payload '{"query":"What is the first mutable learning target?"}'
python -m proof_transformer.cli list-tasks
python -m proof_transformer.cli run-queue
python -m proof_transformer.cli serve-queue --poll-interval 2.0 --max-cycles 10
.\scripts\run_worker.ps1 --max-cycles 10
```

This does not make the local model identical to a frontier hosted assistant, but it does let the local substrate persist, schedule, and execute grounded tasks on its own state.

## Run local commands and ingest results

```powershell
python -m proof_transformer.cli run-command -- python -m unittest discover -s tests
python -m proof_transformer.cli list-executions
python -m proof_transformer.cli review-execution EXECUTION_ID "What did the command print?"
.\scripts\run_command.ps1 python -m unittest discover -s tests
```

Command output is stored under `E:\AUSTIN_AI_BRAIN\logs\executions` and can be ingested back into persistent memory for later retrieval and post-run questioning.

## Repair planning and autonomous repair

```powershell
python -m proof_transformer.cli plan-repair E:\path\to\workspace -- python broken.py
python -m proof_transformer.cli list-repair-plans
python -m proof_transformer.cli apply-repair-plan REPAIR_PLAN_ID
python -m proof_transformer.cli autonomous-repair E:\path\to\workspace -- python broken.py
.\scripts\run_autonomous_repair.ps1 E:\path\to\workspace python broken.py
```

Behavior:
- runs the failing command locally
- stores execution output in persistent memory
- builds a repair plan
- uses a stronger local model backend for exact patch generation when available
- falls back to a deterministic local diagnosis plan when exact safe edits cannot be proven
- only applies exact text replacements that match uniquely inside the workspace

## Health

```powershell
python -m proof_transformer.cli health
```

This reports:
- database presence
- ledger verification status
- local model backend status
- embedding backend status
- queue status
- recent execution history

## Verify the ledger

```powershell
python -m proof_transformer.cli verify-ledger
```

The verifier detects:
- missing parent
- bad parent hash
- recomputed block hash mismatch
- height discontinuity

## Export audit report

```powershell
python -m proof_transformer.cli export-audit E:\AUSTIN_AI_BRAIN\logs\audit_report.json
```

## Status

```powershell
python -m proof_transformer.cli status
```

## Demo

```powershell
python -m proof_transformer.cli demo
```

Or:

```powershell
.\scripts\run_demo.ps1
```

## Run tests

```powershell
cd E:\AUSTIN_AI_BRAIN\proof_transformer
$env:AUSTIN_AI_BRAIN_ROOT = 'E:\AUSTIN_AI_BRAIN'
$env:PYTHONPATH = "$PWD\src"
python -m unittest discover -s tests
```

Or:

```powershell
.\scripts\run_tests.ps1
```

## How this maps to the persistent transformer idea

- **documents + chunks + embeddings** act like a persistent external KV substrate
- **graph memory** captures symbolic structure and contradiction edges
- **context compiler** externalizes attention by retrieving and packing a bounded context packet
- **ranker learning** is the first real mutable learning mechanism
- **policy snapshots** act as a mutable decoder-control layer
- **ledger blocks** govern when updates become part of the persistent mind-state

## Future adapter/checkpoint path

This repo already includes an adapter registry that records:
- name
- version
- path
- parent version
- dataset hash
- eval report hash
- status

To connect a real LoRA or checkpoint trainer later, add a trainer that:
1. writes adapter artifacts under `E:\AUSTIN_AI_BRAIN\adapters`
2. writes eval reports under `E:\AUSTIN_AI_BRAIN\evals`
3. registers pending adapters in `adapter_registry`
4. creates a learning candidate with dataset/eval hashes
5. commits activation only after validator consensus and ledger acceptance

## Limitations

- deterministic hashing embeddings are local and real, but weaker than neural embeddings
- a CPU-first Windows box with no assumed GPU will not match frontier hosted model capability one-for-one
- autonomous repair is strongest when connected to a stronger local model backend; fallback mode is conservative and may stop at diagnosis instead of unsafe patching
- vector search is SQLite/Python brute force, suitable for small-to-medium local corpora
- v0.3 ingests text and markdown only
- neural adapter training is prepared but not implemented here
- fallback generation is grounded and deterministic, not a substitute for a strong model

## IP and authorship notice

See:
- `NOTICE.md`
- `PUBLIC_DISCLOSURE.md`
- `IP_SAFEGUARDING.md`

Originating concept language belongs to Austin Grandstaff, including:
- **Proof-of-Improvement Persistent Transformer**
- **Consensus-gated continual learning**
- **Learning updates as validated transactions**
- **Persistent graph-KV memory as externalized attention**
