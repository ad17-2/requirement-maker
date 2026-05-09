from pathlib import Path

from click.testing import CliRunner

from requirement_maker import cli


def test_cli_missing_credentials_fails_before_pipeline(monkeypatch):
    runner = CliRunner()

    def fail_if_called(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("pipeline should not run without credentials")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda **kwargs: None)
    monkeypatch.setattr(cli, "run_pipeline", fail_if_called)

    with runner.isolated_filesystem():
        Path("meeting.mp3").write_bytes(b"fake audio")

        result = runner.invoke(cli.main, ["meeting.mp3"])

    assert result.exit_code != 0
    assert "Missing required credential(s): OPENAI_API_KEY, ANTHROPIC_API_KEY" in result.output
    assert "cp .env.example .env" in result.output
    assert "Traceback" not in result.output


def test_cli_success_path_uses_mocked_boundaries(monkeypatch):
    runner = CliRunner()
    calls: list[str] = []

    async def fake_prepare_audio(input_file: Path, temp_dir: Path) -> list[Path]:
        calls.append(f"prepare:{input_file.name}")
        return [Path("prepared.mp3")]

    async def fake_transcribe_chunks(audio_paths, openai_key, on_chunk_done, config):  # noqa: ANN001, ANN202
        calls.append(
            f"transcribe:{openai_key}:{len(audio_paths)}:{config.model}:"
            f"{config.concurrency}:{config.timeout}:{config.retries}"
        )
        on_chunk_done(1, 1)
        return "Mocked transcript"

    def fake_generate_requirements(transcript: str, anthropic_key: str, config):  # noqa: ANN001
        calls.append(f"generate:{anthropic_key}:{config.model}:{transcript}")
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
        "transcribe:dummy-openai-key:1:whisper-1:10:60.0:2",
        "generate:dummy-anthropic-key:claude-sonnet-4-5-20250929:Mocked transcript",
    ]


def test_cli_custom_output_path_uses_mocked_boundaries(monkeypatch):
    runner = CliRunner()

    async def fake_prepare_audio(input_file: Path, temp_dir: Path) -> list[Path]:
        return [input_file]

    async def fake_transcribe_chunks(audio_paths, openai_key, on_chunk_done, config):  # noqa: ANN001, ANN202
        return "Transcript"

    def fake_generate_requirements(transcript: str, anthropic_key: str, config) -> str:  # noqa: ANN001
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


def test_help_version_and_doctor_do_not_require_credentials(monkeypatch):
    runner = CliRunner()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(cli, "load_dotenv", lambda **kwargs: None)

    help_result = runner.invoke(cli.main, ["--help"])
    version_result = runner.invoke(cli.main, ["--version"])
    doctor_result = runner.invoke(cli.main, ["--doctor"])

    assert help_result.exit_code == 0
    assert "Convert audio/video recordings" in help_result.output
    assert "Examples:" in help_result.output
    assert "--transcription-model" in help_result.output
    assert "--verbose" in help_result.output
    assert version_result.exit_code == 0
    assert "requirement-maker" in version_result.output
    assert doctor_result.exit_code == 0
    assert "Diagnostics" in doctor_result.output
    assert "OPENAI_API_KEY: missing" in doctor_result.output
    assert "ANTHROPIC_API_KEY: missing" in doctor_result.output


def test_runtime_options_are_validated_and_applied(monkeypatch):
    runner = CliRunner()
    seen: list[cli.RuntimeConfig] = []

    async def fake_run_pipeline(
        input_file: Path,
        output: Path,
        config: cli.RuntimeConfig,
        openai_key: str,
        anthropic_key: str,
    ) -> None:
        seen.append(config)
        output.write_text("ok", encoding="utf-8")

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-anthropic-key")
    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    with runner.isolated_filesystem():
        Path("meeting.mp3").write_bytes(b"fake audio")
        result = runner.invoke(
            cli.main,
            [
                "meeting.mp3",
                "--model",
                "claude-test",
                "--transcription-model",
                "whisper-test",
                "--concurrency",
                "3",
                "--timeout",
                "12.5",
                "--retries",
                "4",
                "--verbose",
            ],
        )

    assert result.exit_code == 0
    assert seen == [
        cli.RuntimeConfig(
            requirement_model="claude-test",
            transcription_model="whisper-test",
            concurrency=3,
            timeout=12.5,
            retries=4,
            verbose=True,
            quiet=False,
        )
    ]
    assert "Runtime configuration:" in result.output
    assert "requirement model: claude-test" in result.output
    assert "OPENAI_API_KEY" not in result.output

    invalid = runner.invoke(cli.main, ["meeting.mp3", "--concurrency", "0"])
    assert invalid.exit_code != 0
    assert "Invalid value for '--concurrency'" in invalid.output


def test_option_precedence_and_verbose_quiet_conflict(monkeypatch):
    runner = CliRunner()
    seen: list[cli.RuntimeConfig] = []

    async def fake_run_pipeline(
        input_file: Path,
        output: Path,
        config: cli.RuntimeConfig,
        openai_key: str,
        anthropic_key: str,
    ) -> None:
        seen.append(config)
        output.write_text("ok", encoding="utf-8")

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-anthropic-key")
    monkeypatch.setenv("REQUIREMENT_MAKER_MODEL", "env-claude")
    monkeypatch.setenv("REQUIREMENT_MAKER_TRANSCRIPTION_MODEL", "env-whisper")
    monkeypatch.setenv("REQUIREMENT_MAKER_CONCURRENCY", "2")
    monkeypatch.setenv("REQUIREMENT_MAKER_TIMEOUT", "9")
    monkeypatch.setenv("REQUIREMENT_MAKER_RETRIES", "1")
    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    with runner.isolated_filesystem():
        Path("meeting.mp3").write_bytes(b"fake audio")
        env_result = runner.invoke(cli.main, ["meeting.mp3"])
        cli_result = runner.invoke(
            cli.main, ["meeting.mp3", "--model", "cli-claude", "--force"]
        )
        conflict_result = runner.invoke(cli.main, ["meeting.mp3", "--verbose", "--quiet"])

    assert env_result.exit_code == 0
    assert cli_result.exit_code == 0
    assert seen[0].requirement_model == "env-claude"
    assert seen[0].transcription_model == "env-whisper"
    assert seen[0].concurrency == 2
    assert seen[0].timeout == 9.0
    assert seen[0].retries == 1
    assert seen[1].requirement_model == "cli-claude"
    assert seen[1].transcription_model == "env-whisper"
    assert conflict_result.exit_code != 0
    assert "--verbose and --quiet cannot be used together" in conflict_result.output


def test_preflight_errors_happen_before_pipeline_or_credential_loading(monkeypatch):
    runner = CliRunner()

    def fail_load_dotenv(**kwargs) -> None:  # noqa: ANN003
        raise AssertionError("credentials should not be loaded before cheap preflight failure")

    async def fail_run_pipeline(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("pipeline should not run for cheap preflight failure")

    monkeypatch.setattr(cli, "load_dotenv", fail_load_dotenv)
    monkeypatch.setattr(cli, "run_pipeline", fail_run_pipeline)

    with runner.isolated_filesystem():
        Path("notes.txt").write_text("not media", encoding="utf-8")
        unsupported = runner.invoke(cli.main, ["notes.txt"])
        missing = runner.invoke(cli.main, ["missing.mp3"])

    assert unsupported.exit_code != 0
    assert "Unsupported file type: .txt" in unsupported.output
    assert "Supported formats:" in unsupported.output
    assert "Traceback" not in unsupported.output
    assert missing.exit_code != 0
    assert "Input file not found: missing.mp3" in missing.output


def test_quiet_success_suppresses_progress_and_failures_use_stderr(monkeypatch):
    runner = CliRunner()

    async def fake_run_pipeline(
        input_file: Path,
        output: Path,
        config: cli.RuntimeConfig,
        openai_key: str,
        anthropic_key: str,
    ) -> None:
        output.write_text("ok", encoding="utf-8")

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-anthropic-key")
    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    with runner.isolated_filesystem():
        Path("meeting.mp3").write_bytes(b"fake audio")
        success = runner.invoke(cli.main, ["meeting.mp3", "--quiet"])
        failure = runner.invoke(cli.main, ["missing.mp3"])

    assert success.exit_code == 0
    assert success.output == ""
    assert failure.exit_code != 0
    assert "Input file not found" in failure.output


def test_existing_output_requires_force_before_pipeline(monkeypatch):
    runner = CliRunner()

    async def fail_run_pipeline(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("pipeline should not run when output is protected")

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-anthropic-key")
    monkeypatch.setattr(cli, "run_pipeline", fail_run_pipeline)

    with runner.isolated_filesystem():
        Path("meeting.mp3").write_bytes(b"fake audio")
        output = Path("meeting_requirements.md")
        output.write_text("original", encoding="utf-8")

        result = runner.invoke(cli.main, ["meeting.mp3"])

        assert output.read_text(encoding="utf-8") == "original"

    assert result.exit_code != 0
    assert "already exists" in result.output
    assert "--force" in result.output


def test_force_allows_existing_output_replacement(monkeypatch):
    runner = CliRunner()

    async def fake_run_pipeline(
        input_file: Path,
        output: Path,
        config: cli.RuntimeConfig,
        openai_key: str,
        anthropic_key: str,
    ) -> None:
        output.write_text("replacement", encoding="utf-8")

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-anthropic-key")
    monkeypatch.setattr(cli, "run_pipeline", fake_run_pipeline)

    with runner.isolated_filesystem():
        Path("meeting.mp3").write_bytes(b"fake audio")
        output = Path("meeting_requirements.md")
        output.write_text("original", encoding="utf-8")

        result = runner.invoke(cli.main, ["meeting.mp3", "--force"])

        assert output.read_text(encoding="utf-8") == "replacement"

    assert result.exit_code == 0


def test_output_path_edge_cases_are_validated_before_pipeline(monkeypatch):
    runner = CliRunner()

    async def fail_run_pipeline(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("pipeline should not run for invalid output path")

    monkeypatch.setattr(cli, "run_pipeline", fail_run_pipeline)

    with runner.isolated_filesystem():
        media = Path("meeting.mp3")
        media.write_bytes(b"fake audio")
        missing_parent = runner.invoke(cli.main, ["meeting.mp3", "--output", "missing/out.md"])
        same_path = runner.invoke(cli.main, ["meeting.mp3", "--output", "meeting.mp3"])
        Path("dir").mkdir()
        directory_output = runner.invoke(cli.main, ["meeting.mp3", "--output", "dir"])

    assert missing_parent.exit_code != 0
    assert "Output directory does not exist" in missing_parent.output
    assert same_path.exit_code != 0
    assert "Output path must differ from input path" in same_path.output
    assert directory_output.exit_code != 0
    assert "Output path is a directory" in directory_output.output


def test_atomic_write_preserves_previous_output_on_failure(monkeypatch):
    runner = CliRunner()

    def fake_generate_requirements(transcript: str, anthropic_key: str, config) -> str:  # noqa: ANN001
        return "new requirements"

    async def fake_prepare_audio(input_file: Path, temp_dir: Path) -> list[Path]:
        return [input_file]

    async def fake_transcribe_chunks(audio_paths, openai_key, on_chunk_done, config):  # noqa: ANN001, ANN202
        return "Transcript"

    def fail_replace(src: Path, dst: Path) -> None:
        raise OSError("simulated replace failure")

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-anthropic-key")
    monkeypatch.setattr(cli, "prepare_audio", fake_prepare_audio)
    monkeypatch.setattr(cli, "transcribe_chunks", fake_transcribe_chunks)
    monkeypatch.setattr(cli, "generate_requirements", fake_generate_requirements)
    monkeypatch.setattr(cli, "_replace_file", fail_replace)

    with runner.isolated_filesystem():
        Path("meeting.mp3").write_bytes(b"fake audio")
        output = Path("out.md")
        output.write_text("original", encoding="utf-8")

        result = runner.invoke(cli.main, ["meeting.mp3", "--output", "out.md", "--force"])

        assert output.read_text(encoding="utf-8") == "original"
        assert not list(Path(".").glob("*.tmp"))

    assert result.exit_code != 0
    assert "Failed to write output" in result.output
