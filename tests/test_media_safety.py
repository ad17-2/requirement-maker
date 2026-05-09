import subprocess
from pathlib import Path

import pytest
from click.testing import CliRunner

from requirement_maker import audio, cli


def test_corrupt_supported_media_fails_without_provider_calls(monkeypatch):
    runner = CliRunner()

    async def fail_transcribe_chunks(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("transcription should not run after media preparation failure")

    def fail_generate_requirements(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        raise AssertionError("generation should not run after media preparation failure")

    monkeypatch.setenv("OPENAI_API_KEY", "dummy-openai-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-anthropic-key")
    monkeypatch.setattr(cli, "transcribe_chunks", fail_transcribe_chunks)
    monkeypatch.setattr(cli, "generate_requirements", fail_generate_requirements)

    with runner.isolated_filesystem():
        Path("corrupt.mp3").write_bytes(b"not real mp3")

        result = runner.invoke(cli.main, ["corrupt.mp3"])

        assert not Path("corrupt_requirements.md").exists()

    assert result.exit_code != 0
    assert "Media preparation failed" in result.output
    assert "Traceback" not in result.output


def test_video_preparation_uses_temp_dir_and_cleans_up(monkeypatch, tmp_path):
    calls: list[tuple[Path, str | None]] = []

    def fake_run(command, capture_output, check, text=False):  # noqa: ANN001, ANN202
        output_path = Path(command[-1])
        output_path.write_bytes(b"prepared audio")
        calls.append((output_path, str(output_path.parent)))
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(audio.subprocess, "run", fake_run)
    monkeypatch.setattr(audio, "get_audio_duration", lambda file_path: 1.0)
    temp_root = tmp_path / "media work"
    temp_root.mkdir()
    input_video = tmp_path / "video input.mp4"
    input_video.write_bytes(b"fake video")

    chunks = audio.prepare_audio_sync(input_video, temp_root)

    assert chunks == [temp_root / "video input.mp3"]
    assert calls == [(temp_root / "video input.mp3", str(temp_root))]


def test_large_audio_split_uses_temp_dir_and_order(monkeypatch, tmp_path):
    async def fake_split_one_chunk(
        file_path: Path, start: int, duration: int, output_path: Path
    ) -> Path:
        output_path.write_bytes(f"{start}:{duration}".encode())
        return output_path

    monkeypatch.setattr(audio, "MAX_WHISPER_SIZE", 1)
    monkeypatch.setattr(audio, "get_audio_duration", lambda file_path: 1201.0)
    monkeypatch.setattr(audio, "_split_one_chunk", fake_split_one_chunk)
    media = tmp_path / "long.mp3"
    media.write_bytes(b"long audio")
    temp_root = tmp_path / "chunks with spaces"
    temp_root.mkdir()

    chunks = audio.prepare_audio_sync(media, temp_root)

    assert chunks == [
        temp_root / "chunk_000.mp3",
        temp_root / "chunk_001.mp3",
        temp_root / "chunk_002.mp3",
    ]


def test_unsupported_media_does_not_invoke_ffmpeg(monkeypatch, tmp_path):
    def fail_ensure_ffmpeg() -> str:
        raise AssertionError("ffmpeg should not be checked for unsupported media")

    monkeypatch.setattr(audio, "ensure_ffmpeg", fail_ensure_ffmpeg)

    notes = tmp_path / "notes.txt"
    notes.write_text("not media", encoding="utf-8")
    with pytest.raises(audio.MediaPreparationError, match="Unsupported file type"):
        audio.prepare_audio_sync(notes, tmp_path / "unused")
