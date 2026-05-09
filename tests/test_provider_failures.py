from pathlib import Path

from click.testing import CliRunner

from requirement_maker import cli
from requirement_maker.provider_errors import (
    ProviderErrorKind,
    ProviderStage,
    classify_provider_exception,
)


class FakeStatusError(Exception):
    def __init__(self, status_code: int, message: str = "provider failed") -> None:
        super().__init__(message)
        self.status_code = status_code


class AuthenticationError(Exception):
    pass


class RateLimitError(Exception):
    pass


class APITimeoutError(Exception):
    pass


class BadRequestError(Exception):
    pass


def test_provider_exception_classification_is_secret_safe():
    cases = [
        (AuthenticationError("bad key REDACTED_PROVIDER_VALUE"), ProviderErrorKind.AUTHENTICATION, False),
        (RateLimitError("slow down"), ProviderErrorKind.RATE_LIMIT, True),
        (APITimeoutError("timed out"), ProviderErrorKind.TIMEOUT, True),
        (BadRequestError("bad model"), ProviderErrorKind.INVALID_REQUEST, False),
        (FakeStatusError(500), ProviderErrorKind.PROVIDER, True),
    ]

    for exc, expected_kind, expected_transient in cases:
        error = classify_provider_exception(
            exc,
            stage=ProviderStage.TRANSCRIPTION,
            provider="OpenAI",
        )

        assert error.kind is expected_kind
        assert error.transient is expected_transient
        assert "REDACTED_PROVIDER_VALUE" not in error.detail


def test_transcription_provider_failure_is_stage_specific_and_leaves_no_output(monkeypatch):
    runner = CliRunner()

    async def fake_prepare_audio(input_file: Path, temp_dir: Path) -> list[Path]:
        return [input_file]

    async def fail_transcribe_chunks(audio_paths, openai_key, on_chunk_done, config):  # noqa: ANN001, ANN202
        raise classify_provider_exception(
            RateLimitError("quota exceeded for REDACTED_PROVIDER_VALUE"),
            stage=ProviderStage.TRANSCRIPTION,
            provider="OpenAI",
        )

    def fail_run_agentic_workflow(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("generation should not run after transcription failure")

    monkeypatch.setenv("OPENAI_API_KEY", "OPENAI_TEST_VALUE")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ANTHROPIC_TEST_VALUE")
    monkeypatch.setattr(cli, "prepare_audio", fake_prepare_audio)
    monkeypatch.setattr(cli, "transcribe_chunks", fail_transcribe_chunks)
    monkeypatch.setattr(cli, "run_agentic_workflow", fail_run_agentic_workflow)

    with runner.isolated_filesystem():
        Path("meeting.mp3").write_bytes(b"fake audio")

        result = runner.invoke(cli.main, ["meeting.mp3"])

        assert not Path("meeting_requirements.md").exists()

    assert result.exit_code != 0
    assert "Transcription failed (OpenAI rate_limit, transient)" in result.output
    assert "Repair: Wait and retry, lower --concurrency" in result.output
    assert "REDACTED_PROVIDER_VALUE" not in result.output
    assert "Traceback" not in result.output
    assert "Done:" not in result.output


def test_generation_provider_failure_is_stage_specific_and_leaves_existing_output(monkeypatch):
    runner = CliRunner()

    async def fake_prepare_audio(input_file: Path, temp_dir: Path) -> list[Path]:
        return [input_file]

    async def fake_transcribe_chunks(audio_paths, openai_key, on_chunk_done, config):  # noqa: ANN001, ANN202
        return "Transcript"

    def fail_run_agentic_workflow(transcript: str, provider, config):  # noqa: ANN001
        raise classify_provider_exception(
            BadRequestError("invalid model for request"),
            stage=ProviderStage.REQUIREMENT_GENERATION,
            provider="Anthropic",
        )

    monkeypatch.setenv("OPENAI_API_KEY", "OPENAI_TEST_VALUE")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ANTHROPIC_TEST_VALUE")
    monkeypatch.setattr(cli, "prepare_audio", fake_prepare_audio)
    monkeypatch.setattr(cli, "transcribe_chunks", fake_transcribe_chunks)
    monkeypatch.setattr(cli, "run_agentic_workflow", fail_run_agentic_workflow)

    with runner.isolated_filesystem():
        Path("meeting.mp3").write_bytes(b"fake audio")
        output = Path("out.md")
        output.write_text("original", encoding="utf-8")

        result = runner.invoke(cli.main, ["meeting.mp3", "--output", "out.md", "--force"])

        assert output.read_text(encoding="utf-8") == "original"

    assert result.exit_code != 0
    assert "Requirement generation failed (Anthropic invalid_request, permanent)" in result.output
    assert "Repair: Check the selected model/options" in result.output
    assert "Traceback" not in result.output
    assert "Done:" not in result.output
