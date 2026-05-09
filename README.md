# requirement-maker

`requirement-maker` turns meeting audio or video into source-linked requirements artifacts. It prepares local media with `ffmpeg`, transcribes with OpenAI, generates requirements with Anthropic, and writes Markdown plus audit files.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .

cp .env.example .env
# Edit .env with real values:
# OPENAI_API_KEY=...
# ANTHROPIC_API_KEY=...

make doctor
make req INPUT=meeting.mp4
```

Use `OUTPUT=spec.md` to choose the Markdown filename:

```bash
make req INPUT=meeting.mp4 OUTPUT=spec.md
```

## Common commands

The Makefile is the easiest way to run the project:

| Command | What it does |
| --- | --- |
| `make help` | Show available targets. |
| `make doctor` | Check Python, package setup, `ffmpeg`, `ffprobe`, and configured credentials. |
| `make req INPUT=meeting.mp4 [OUTPUT=spec.md]` | Generate Markdown requirements. |
| `make req-json INPUT=meeting.mp4 [OUTPUT=spec.md]` | Generate Markdown plus structured JSON. |
| `make req-tasks INPUT=meeting.mp4 [OUTPUT=spec.md]` | Generate Markdown plus task handoff JSON. |
| `make req-demo` | Generate a tiny local `demo.wav` and demo artifacts. Requires API keys. |
| `make test` | Run pytest. |
| `make lint` | Run Ruff checks. |
| `make typecheck` | Run mypy over `src`. |
| `make check` | Run tests, lint, and typecheck. |

## Direct CLI usage

You can also run the installed CLI directly:

```bash
.venv/bin/requirement-maker meeting.mp4
.venv/bin/requirement-maker meeting.mp3 --output spec.md
.venv/bin/requirement-maker meeting.mp4 --json --tasks --verbose
.venv/bin/requirement-maker --doctor
```

Useful options:

| Option | Purpose |
| --- | --- |
| `-o`, `--output PATH` | Choose the Markdown output path. |
| `--json` | Write a structured JSON export next to the Markdown. |
| `--tasks` | Write a task handoff JSON export next to the Markdown. |
| `--verbose` | Print pipeline stages and non-secret runtime settings. |
| `--quiet` | Suppress non-error progress output. |
| `--force` | Overwrite existing output artifacts. |
| `--model TEXT` | Choose the Anthropic generation model. |
| `--transcription-model TEXT` | Choose the OpenAI transcription model. |
| `--concurrency INTEGER` | Set concurrent transcription calls. |
| `--timeout FLOAT` | Set provider timeout in seconds. |
| `--retries INTEGER` | Set provider retry count. |
| `--doctor` | Run safe local diagnostics. |

Run `.venv/bin/requirement-maker --help` for the full CLI help.

## Outputs

For `meeting.mp4`, the default run writes:

```text
meeting_requirements.md
meeting_requirements.trace.json
meeting_requirements.manifest.json
```

Optional exports add:

```text
meeting_requirements.json          # with --json or make req-json
meeting_requirements.tasks.json    # with --tasks or make req-tasks
```

The Markdown file is the polished requirements document. The trace and manifest files provide an audit trail of the workflow and generated artifacts.

## Requirements

- Python 3.10+
- `ffmpeg` and `ffprobe`
- Real API keys in `.env` or the shell environment:
  - `OPENAI_API_KEY`
  - `ANTHROPIC_API_KEY`

Supported inputs include common local audio and video files such as `.mp3`, `.mp4`, `.m4a`, `.wav`, `.webm`, `.mov`, and `.mkv`.

Keep `.env`, real recordings, transcripts, and generated customer artifacts out of commits unless they are intentionally sanitized fixtures.

## Validation

Run the standard local checks with:

```bash
make check
```

This runs:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
.venv/bin/python -m mypy src
```

Default validation does not call real providers. The real provider end-to-end test is opt-in:

```bash
REQUIREMENT_MAKER_RUN_REAL_E2E=1 \
  .venv/bin/python -m pytest tests/test_real_provider_e2e.py -m real_provider -q
```
