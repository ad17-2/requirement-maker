import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest
from dotenv import find_dotenv, load_dotenv

from requirement_maker.generate import GenerationConfig, generate_requirements
from requirement_maker.transcribe import TranscriptionConfig, transcribe_chunks

load_dotenv(dotenv_path=find_dotenv(usecwd=True))


REAL_E2E_ENV = "REQUIREMENT_MAKER_RUN_REAL_E2E"
REAL_E2E_COMMAND = (
    "REQUIREMENT_MAKER_RUN_REAL_E2E=1 .venv/bin/python -m pytest "
    "tests/test_real_provider_e2e.py -m real_provider -q"
)


def _real_e2e_enabled() -> bool:
    return os.environ.get(REAL_E2E_ENV) == "1"


def _has_credentials() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY")) and bool(os.environ.get("ANTHROPIC_API_KEY"))


pytestmark = pytest.mark.real_provider


@pytest.mark.skipif(
    not _real_e2e_enabled() or not _has_credentials(),
    reason=(
        "real provider E2E is skipped by default. Opt in with: "
        f"{REAL_E2E_COMMAND}"
    ),
)
def test_real_provider_e2e_uses_tiny_media_and_safe_trace(tmp_path: Path) -> None:
    if shutil.which("say") is None:
        pytest.skip("macOS say command is required to generate tiny spoken media")

    spoken_path = tmp_path / "tiny-requirements-demo.aiff"
    media_path = tmp_path / "tiny-requirements-demo.wav"
    output_path = tmp_path / "tiny-requirements.md"

    subprocess.run(
        [
            "say",
            "-o",
            str(spoken_path),
            "Build email login. Defer social login.",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(spoken_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-y",
            str(media_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "quiet",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(media_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    duration = float(probe.stdout.strip())
    media_size = media_path.stat().st_size

    assert 0 < duration <= 4.0
    assert media_size < 250_000

    transcript = asyncio.run(
        transcribe_chunks(
            [media_path],
            os.environ["OPENAI_API_KEY"],
            config=TranscriptionConfig(
                model=os.environ.get("REQUIREMENT_MAKER_REAL_TRANSCRIPTION_MODEL", "whisper-1"),
                concurrency=1,
                timeout=60,
                retries=0,
            ),
        )
    )
    assert "login" in transcript.lower()

    requirement_model = os.environ.get(
        "REQUIREMENT_MAKER_REAL_MODEL",
        "claude-haiku-4-5-20251001",
    )
    markdown = generate_requirements(
        transcript,
        os.environ["ANTHROPIC_API_KEY"],
        GenerationConfig(model=requirement_model, timeout=60, retries=0),
    )
    assert markdown.strip()

    output_path.write_text(markdown, encoding="utf-8")
    trace_path = tmp_path / "tiny-requirements.trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": "real-provider-validation-v1",
                "media": {
                    "path": media_path.name,
                    "duration_seconds": duration,
                    "size_bytes": media_size,
                },
                "stages": [
                    {
                        "stage": "transcription",
                        "provider": "OpenAI",
                        "model": os.environ.get(
                            "REQUIREMENT_MAKER_REAL_TRANSCRIPTION_MODEL",
                            "whisper-1",
                        ),
                        "concurrency": 1,
                        "timeout_seconds": 60,
                        "retry_count": 0,
                    },
                    {
                        "stage": "requirement_generation",
                        "provider": "Anthropic",
                        "model": requirement_model,
                        "concurrency": 1,
                        "timeout_seconds": 60,
                        "retry_count": 0,
                    },
                ],
                "artifacts": {
                    "markdown": output_path.name,
                    "trace": trace_path.name,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    stages = [entry["stage"] for entry in trace["stages"]]
    assert stages == ["transcription", "requirement_generation"]
    providers = [entry["provider"] for entry in trace["stages"]]
    assert providers == ["OpenAI", "Anthropic"]
    assert all(entry["concurrency"] == 1 for entry in trace["stages"])
    assert all(entry["retry_count"] == 0 for entry in trace["stages"])

    combined_evidence = (
        transcript
        + output_path.read_text(encoding="utf-8")
        + trace_path.read_text(encoding="utf-8")
    )
    for secret_name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        secret = os.environ.get(secret_name, "")
        assert secret
        assert secret not in combined_evidence
    assert "sk-" not in combined_evidence
