# requirement-maker

`requirement-maker` is a CLI that turns meeting recordings into polished, source-linked requirements artifacts. It prepares audio/video with `ffmpeg`, transcribes with OpenAI, runs an explicit Anthropic-powered agentic workflow, and writes shareable planning outputs for engineers, PMs, and implementation agents.

## Agentic workflow

```text
Input media (.mp3/.mp4/.m4a/.wav/.webm/...)
  → ffmpeg/ffprobe media preparation
  → OpenAI transcription
  → Anthropic planner creates extraction units
  → Anthropic extractor captures product signals with source references
  → deterministic merge/dedupe consolidates duplicates and conflicts
  → Anthropic critic reviews traceability, ambiguity, and weak criteria
  → Anthropic writer produces final requirements
  → Markdown, trace, manifest, and optional JSON/task exports
```

The normal command is still one step: provide a supported recording and receive a requirements Markdown file plus audit artifacts. No long-running services, web servers, databases, or ports are required.

## Prerequisites

- **Python 3.10+**. The project is validated with Python 3.11.
- **ffmpeg and ffprobe**.
  - macOS: `brew install ffmpeg`
  - Ubuntu/Debian: `sudo apt install ffmpeg`
- **OpenAI API key** for transcription (`OPENAI_API_KEY`).
- **Anthropic API key** for planning, extraction, critique, and writing (`ANTHROPIC_API_KEY`).

## Installation from a clean checkout

```bash
git clone https://github.com/AdrianAcala/requirement-maker.git
cd requirement-maker

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

If your default `python3` is older than 3.10, create the virtual environment with a newer interpreter, for example `python3.11 -m venv .venv`.

## Configuration

Use environment variables or a local `.env` file. The `.env` file is for your machine only and must not be committed.

```bash
cp .env.example .env
# Edit .env and set:
# OPENAI_API_KEY=...
# ANTHROPIC_API_KEY=...
```

You can also export keys for a single shell session:

```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

The CLI never prints secret values. Verbose output reports selected models and runtime settings, not credentials.

## Sanity check

Run diagnostics before processing media:

```bash
.venv/bin/requirement-maker --doctor
```

The doctor command checks Python, the installed package version, `ffmpeg`, `ffprobe`, and whether provider credentials are configured. It does not call paid provider APIs and does not overwrite config files.

## Usage

```bash
# Basic: writes meeting_requirements.md, meeting_requirements.trace.json,
# and meeting_requirements.manifest.json
.venv/bin/requirement-maker meeting.mp4

# Custom Markdown output path
.venv/bin/requirement-maker meeting.mp3 -o spec.md

# Include structured JSON and task handoff exports
.venv/bin/requirement-maker meeting.mp4 --json --tasks

# Show stage progress and resolved non-secret runtime configuration
.venv/bin/requirement-maker meeting.mp4 --verbose

# Use lower-cost or alternate models and tuned runtime settings
.venv/bin/requirement-maker meeting.mp4 \
  --model claude-haiku-4-5-20251001 \
  --transcription-model whisper-1 \
  --concurrency 4 \
  --timeout 60 \
  --retries 2
```

## CLI options

| Flag | Default | Description |
| --- | --- | --- |
| `--version` | n/a | Print the installed `requirement-maker` version. |
| `-o`, `--output PATH` | `<input_stem>_requirements.md` | Markdown output path. |
| `--model TEXT` | `claude-sonnet-4-5-20250929` | Anthropic model for requirement-generation workflow stages. Can also be set with `REQUIREMENT_MAKER_MODEL`. |
| `--transcription-model TEXT` | `whisper-1` | OpenAI transcription model. Can also be set with `REQUIREMENT_MAKER_TRANSCRIPTION_MODEL`. |
| `--concurrency INTEGER` | `10` | Maximum concurrent transcription calls, from `1` to `50`. Can also be set with `REQUIREMENT_MAKER_CONCURRENCY`. |
| `--timeout FLOAT` | `60.0` | Provider timeout in seconds. Can also be set with `REQUIREMENT_MAKER_TIMEOUT`. |
| `--retries INTEGER` | `2` | Provider retry count, from `0` to `10`. Can also be set with `REQUIREMENT_MAKER_RETRIES`. |
| `--verbose` | off | Print pipeline stages and resolved runtime configuration. |
| `--quiet` | off | Suppress non-error progress output on success. |
| `--force` | off | Replace existing output artifacts. Without this flag, existing outputs are protected before paid work starts. |
| `--json` | off | Write a versioned structured JSON export next to Markdown. |
| `--tasks` | off | Write a versioned task handoff JSON export next to Markdown. |
| `--doctor` | off | Run safe local diagnostics and setup guidance. |
| `-h`, `--help` | n/a | Print help and examples. |

CLI flags take precedence over environment variables and `.env` values, which take precedence over built-in defaults. `--verbose` and `--quiet` are mutually exclusive.

## Supported media

- Audio: `.flac`, `.m4a`, `.mp3`, `.ogg`, `.wav`, `.webm`
- Video: `.avi`, `.mkv`, `.mov`, `.mp4`, `.webm`

Unsupported extensions, missing paths, corrupt media, invalid options, missing credentials, and existing output conflicts fail before downstream paid provider work whenever possible.

## Outputs and naming

Every successful run writes:

- `*.md` — polished requirements document.
- `*.trace.json` — audit trace for workflow stages, model metadata, retry counts, and final artifact paths.
- `*.manifest.json` — list of generated artifact paths.

Optional outputs:

- `--json` writes `*.json` with schema version `requirements-export-v1`.
- `--tasks` writes `*.tasks.json` with schema version `task-handoff-v1`.

Default names are derived from the Markdown output. For `meeting.mp4`, the default artifacts are:

```text
meeting_requirements.md
meeting_requirements.trace.json
meeting_requirements.manifest.json
meeting_requirements.json          # only with --json
meeting_requirements.tasks.json    # only with --tasks
```

The manifest stores relative public artifact paths when possible and verifies each listed artifact exists. Public Markdown, JSON, task exports, and manifest paths are sanitized to avoid API keys, authorization headers, and user-specific absolute paths.

## Markdown contents

The generated requirements document includes:

1. **Executive Summary**
2. **Background & Context**
3. **Goals & Objectives**
4. **Functional Requirements**
5. **Non-Functional Requirements**
6. **User Flows & Scenarios**
7. **Data Requirements**
8. **Dependencies & Constraints**
9. **Out of Scope**
10. **Open Questions & Ambiguities**
11. **Participants & Decisions**
12. **Action & Task Candidates**
13. **Source Traceability**

Functional and non-functional requirements include stable IDs and acceptance-style validation notes. Open questions and conflicts are preserved rather than invented away.

## Copy-paste demo path

This demo starts from a clean checkout, generates a tiny local audio file, runs the agentic workflow, and produces Markdown, JSON, task, trace, and manifest artifacts. It requires configured provider keys but no hidden services or open ports.

```bash
git clone https://github.com/AdrianAcala/requirement-maker.git
cd requirement-maker

python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .

export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."

.venv/bin/requirement-maker --doctor

ffmpeg -hide_banner -loglevel error -f lavfi \
  -i "sine=frequency=1000:duration=1" \
  -ar 16000 -ac 1 demo.wav -y

.venv/bin/requirement-maker demo.wav \
  --output demo_requirements.md \
  --json \
  --tasks \
  --verbose \
  --model claude-haiku-4-5-20251001 \
  --concurrency 1 \
  --timeout 60 \
  --retries 1 \
  --force

ls demo_requirements.md \
  demo_requirements.json \
  demo_requirements.tasks.json \
  demo_requirements.trace.json \
  demo_requirements.manifest.json
```

Do not commit `demo.wav` or generated demo artifacts unless you intentionally create sanitized fixtures. For a dry, no-provider smoke check from a clean checkout, run `.venv/bin/requirement-maker --help` and `.venv/bin/requirement-maker --doctor`.

## Validation

Local validation uses the Python 3.10+ virtual environment and default tests do not call real providers:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy src
.venv/bin/python -m compileall -q src
.venv/bin/requirement-maker --help
```

The real provider E2E check is explicit opt-in and bounded-cost:

```bash
REQUIREMENT_MAKER_RUN_REAL_E2E=1 \
  .venv/bin/python -m pytest tests/test_real_provider_e2e.py -m real_provider -q
```

## Limitations and safety notes

- Real transcription and generation require valid OpenAI and Anthropic credentials.
- Provider API usage may incur cost; use tiny media for demos and lower-cost models when appropriate.
- The CLI processes local files and writes local artifacts only. It does not run a server or require ports.
- Failed runs should not leave normal completed-looking outputs; existing outputs are protected unless `--force` is provided.
- Keep `.env`, generated recordings, real meeting transcripts, and generated requirement outputs out of commits unless they are sanitized fixtures.
