# requirement-maker

CLI tool that converts audio/video recordings into comprehensive requirement documents. Record a meeting, get a detailed spec.

## How it works

```
Input (mp3/mp4/m4a/wav/webm)
  → ffmpeg extracts audio (if video)
  → OpenAI Whisper API transcribes audio
  → Anthropic Claude generates detailed requirements
  → Output: markdown file
```

Large files (>25MB) are automatically split into chunks and transcribed in parallel.

## Prerequisites

- **Python 3.10+**
- **ffmpeg** — `brew install ffmpeg` (macOS) / `sudo apt install ffmpeg` (Ubuntu)
- **OpenAI API key** — for Whisper transcription
- **Anthropic API key** — for requirement generation

## Installation

```bash
git clone https://github.com/AdrianAcala/requirement-maker.git
cd requirement-maker
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configuration

Copy the example env file and add your API keys:

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

```bash
# Basic — outputs meeting_requirements.md
requirement-maker meeting.mp4

# Custom output path
requirement-maker meeting.mp3 -o spec.md

# Use a different Claude model
requirement-maker meeting.mp4 --model claude-haiku-4-5-20251001
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output` | `<input>_requirements.md` | Output file path |
| `--model` | `claude-sonnet-4-5-20250929` | Anthropic model to use |

### Supported formats

Audio: `.mp3`, `.wav`, `.m4a`, `.ogg`, `.flac`, `.webm`
Video: `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`

## Output

The generated requirement document includes:

1. **Executive Summary** — what was discussed and decided
2. **Background & Context** — business drivers, pain points
3. **Goals & Objectives** — measurable outcomes
4. **Functional Requirements** — every feature/behavior, grouped by area
5. **Non-Functional Requirements** — performance, security, scalability
6. **User Flows & Scenarios** — step-by-step journeys
7. **Data Requirements** — models, storage, integrations
8. **Dependencies & Constraints** — blockers, timelines
9. **Out of Scope** — explicitly deferred items
10. **Open Questions & Ambiguities** — unresolved items flagged
11. **Participants & Decisions** — who decided what

The output is intentionally a single raw document — comprehensive enough that you can feed it into other tools (Jira, Linear, Claude) to break down into individual tickets.

## How large files are handled

- Files under 25MB are sent directly to the Whisper API
- Larger files are split into 10-minute chunks via ffmpeg
- All chunks are transcribed in parallel (up to 10 concurrent API calls)
- Transcripts are stitched back in order before requirement generation
