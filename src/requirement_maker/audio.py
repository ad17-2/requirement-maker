import asyncio
import math
import os
import shutil
import subprocess
from pathlib import Path

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm"}
SUPPORTED_EXTENSIONS = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS

MAX_WHISPER_SIZE = 24 * 1024 * 1024  # 24MB (API limit is 25MB, leave margin)


class MediaPreparationError(RuntimeError):
    """Raised when local media validation or ffmpeg preparation fails."""


def ensure_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if not path:
        raise MediaPreparationError(
            "ffmpeg not found. Install it:\n"
            "  macOS:  brew install ffmpeg\n"
            "  Ubuntu: sudo apt install ffmpeg\n"
            "  Windows: https://ffmpeg.org/download.html"
        )
    return path


def validate_input(file_path: Path) -> Path:
    if not file_path.exists():
        raise MediaPreparationError(f"File not found: {file_path}")
    if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise MediaPreparationError(
            f"Unsupported file type: {file_path.suffix}\n"
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return file_path


def is_video(file_path: Path) -> bool:
    return file_path.suffix.lower() in VIDEO_EXTENSIONS


def extract_audio(video_path: Path, output_dir: Path) -> Path:
    ensure_ffmpeg()
    output_path = Path(output_dir) / f"{video_path.stem}.mp3"

    try:
        subprocess.run(
            [
                "ffmpeg", "-i", str(video_path),
                "-vn", "-acodec", "libmp3lame", "-q:a", "2",
                "-y", str(output_path),
            ],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
        detail = (stderr or "").strip().splitlines()[-1:] or ["ffmpeg could not extract audio."]
        raise MediaPreparationError(f"Could not extract audio from {video_path}: {detail[0]}") from exc
    return output_path


def get_audio_duration(file_path: Path) -> float:
    ensure_ffmpeg()
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(file_path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as exc:
        raise MediaPreparationError(f"Could not determine media duration for {file_path}.") from exc


async def _split_one_chunk(
    file_path: Path, start: int, duration: int, output_path: Path
) -> Path:
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-i", str(file_path),
        "-ss", str(start), "-t", str(duration),
        "-acodec", "libmp3lame", "-q:a", "2",
        "-y", str(output_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        detail = stderr.decode(errors="replace").strip().splitlines()[-1:] or [
            "ffmpeg could not create chunk."
        ]
        raise MediaPreparationError(f"Could not split audio chunk {output_path.name}: {detail[0]}")
    return output_path


async def split_audio(
    file_path: Path, output_dir: Path, chunk_duration: int = 600
) -> list[Path]:
    ensure_ffmpeg()
    total_duration = get_audio_duration(file_path)
    if total_duration <= 0:
        raise MediaPreparationError(f"Could not determine duration of {file_path}")

    num_chunks = math.ceil(total_duration / chunk_duration)

    tasks = []
    for i in range(num_chunks):
        start = i * chunk_duration
        chunk_path = Path(output_dir) / f"chunk_{i:03d}.mp3"
        tasks.append(_split_one_chunk(file_path, start, chunk_duration, chunk_path))

    chunks = await asyncio.gather(*tasks)
    return list(chunks)


async def prepare_audio(file_path: Path, temp_dir: Path) -> list[Path]:
    file_path = validate_input(file_path)

    if is_video(file_path):
        audio_path = extract_audio(file_path, temp_dir)
    else:
        audio_path = file_path

    get_audio_duration(audio_path)
    if os.path.getsize(audio_path) > MAX_WHISPER_SIZE:
        return await split_audio(audio_path, temp_dir)

    return [audio_path]


def prepare_audio_sync(file_path: Path, temp_dir: Path) -> list[Path]:
    return asyncio.run(prepare_audio(file_path, temp_dir))
