from pathlib import Path

from click.testing import CliRunner

from requirement_maker import cli


def test_cli_missing_credentials_fails_before_pipeline(monkeypatch):
    runner = CliRunner()

    def fail_if_called(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("pipeline should not run without credentials")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.setattr(cli, "run_pipeline", fail_if_called)

    with runner.isolated_filesystem():
        Path("meeting.mp3").write_bytes(b"fake audio")

        result = runner.invoke(cli.main, ["meeting.mp3"])

    assert result.exit_code != 0
    assert "OPENAI_API_KEY not set" in str(result.exception)
    assert "ANTHROPIC_API_KEY" not in result.output
    assert "Traceback" not in result.output


def test_cli_success_path_uses_mocked_boundaries(monkeypatch):
    runner = CliRunner()
    calls: list[str] = []

    async def fake_prepare_audio(input_file: Path) -> list[Path]:
        calls.append(f"prepare:{input_file.name}")
        return [Path("prepared.mp3")]

    async def fake_transcribe_chunks(audio_paths, openai_key, on_chunk_done):  # noqa: ANN001, ANN202
        calls.append(f"transcribe:{openai_key}:{len(audio_paths)}")
        on_chunk_done(1, 1)
        return "Mocked transcript"

    def fake_generate_requirements(transcript: str, anthropic_key: str, model: str) -> str:
        calls.append(f"generate:{anthropic_key}:{model}:{transcript}")
        return "# Requirements\n\nGenerated from mocked providers.\n"

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-anthropic-key")
    monkeypatch.setattr(cli, "prepare_audio", fake_prepare_audio)
    monkeypatch.setattr(cli, "transcribe_chunks", fake_transcribe_chunks)
    monkeypatch.setattr(cli, "generate_requirements", fake_generate_requirements)

    with runner.isolated_filesystem():
        Path("meeting.mp3").write_bytes(b"fake audio")

        result = runner.invoke(cli.main, ["meeting.mp3"])

        output = Path("meeting_requirements.md")
        assert output.read_text(encoding="utf-8") == "# Requirements\n\nGenerated from mocked providers.\n"

    assert result.exit_code == 0
    assert calls == [
        "prepare:meeting.mp3",
        "transcribe:dummy-openai-key:1",
        "generate:dummy-anthropic-key:claude-sonnet-4-5-20250929:Mocked transcript",
    ]


def test_cli_custom_output_path_uses_mocked_boundaries(monkeypatch):
    runner = CliRunner()

    async def fake_prepare_audio(input_file: Path) -> list[Path]:
        return [input_file]

    async def fake_transcribe_chunks(audio_paths, openai_key, on_chunk_done):  # noqa: ANN001, ANN202
        return "Transcript"

    def fake_generate_requirements(transcript: str, anthropic_key: str, model: str) -> str:
        return "Custom output"

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-anthropic-key")
    monkeypatch.setattr(cli, "prepare_audio", fake_prepare_audio)
    monkeypatch.setattr(cli, "transcribe_chunks", fake_transcribe_chunks)
    monkeypatch.setattr(cli, "generate_requirements", fake_generate_requirements)

    with runner.isolated_filesystem():
        Path("meeting.mp3").write_bytes(b"fake audio")

        result = runner.invoke(cli.main, ["meeting.mp3", "--output", "custom.md"])

        assert Path("custom.md").read_text(encoding="utf-8") == "Custom output"
        assert not Path("meeting_requirements.md").exists()

    assert result.exit_code == 0
