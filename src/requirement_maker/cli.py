import asyncio
import os
import shutil
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from requirement_maker.audio import SUPPORTED_EXTENSIONS, prepare_audio
from requirement_maker.generate import GenerationConfig, generate_requirements
from requirement_maker.transcribe import TranscriptionConfig, transcribe_chunks


DEFAULT_REQUIREMENT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_TRANSCRIPTION_MODEL = "whisper-1"
DEFAULT_CONCURRENCY = 10
DEFAULT_TIMEOUT = 60.0
DEFAULT_RETRIES = 2


@dataclass(frozen=True)
class RuntimeConfig:
    requirement_model: str = DEFAULT_REQUIREMENT_MODEL
    transcription_model: str = DEFAULT_TRANSCRIPTION_MODEL
    concurrency: int = DEFAULT_CONCURRENCY
    timeout: float = DEFAULT_TIMEOUT
    retries: int = DEFAULT_RETRIES
    verbose: bool = False
    quiet: bool = False


def _package_version() -> str:
    try:
        return version("requirement-maker")
    except PackageNotFoundError:
        return "0.1.0"


def _echo(message: str, config: RuntimeConfig, *, err: bool = False, force: bool = False) -> None:
    if force or err or not config.quiet:
        click.echo(message, err=err)


def _fail(message: str) -> None:
    raise click.ClickException(message)


def _supported_formats_text() -> str:
    return ", ".join(sorted(SUPPORTED_EXTENSIONS))


def _validate_input_path(input_file: Path) -> None:
    if not input_file.exists():
        _fail(f"Input file not found: {input_file}")
    if input_file.suffix.lower() not in SUPPORTED_EXTENSIONS:
        _fail(
            f"Unsupported file type: {input_file.suffix or '<none>'}\n"
            f"Supported formats: {_supported_formats_text()}"
        )


def _load_int_env(name: str, default: int, minimum: int, maximum: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError:
        _fail(f"{name} must be an integer between {minimum} and {maximum}.")
    if parsed < minimum or parsed > maximum:
        _fail(f"{name} must be between {minimum} and {maximum}.")
    return parsed


def _load_float_env(name: str, default: float, minimum: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        parsed = float(value)
    except ValueError:
        _fail(f"{name} must be a number greater than {minimum}.")
    if parsed <= minimum:
        _fail(f"{name} must be greater than {minimum}.")
    return parsed


def _resolve_config(
    model: str | None,
    transcription_model: str | None,
    concurrency: int | None,
    timeout: float | None,
    retries: int | None,
    verbose: bool,
    quiet: bool,
) -> RuntimeConfig:
    if verbose and quiet:
        _fail("--verbose and --quiet cannot be used together.")
    return RuntimeConfig(
        requirement_model=model
        or os.environ.get("REQUIREMENT_MAKER_MODEL", DEFAULT_REQUIREMENT_MODEL),
        transcription_model=transcription_model
        or os.environ.get("REQUIREMENT_MAKER_TRANSCRIPTION_MODEL", DEFAULT_TRANSCRIPTION_MODEL),
        concurrency=concurrency
        if concurrency is not None
        else _load_int_env("REQUIREMENT_MAKER_CONCURRENCY", DEFAULT_CONCURRENCY, 1, 50),
        timeout=timeout
        if timeout is not None
        else _load_float_env("REQUIREMENT_MAKER_TIMEOUT", DEFAULT_TIMEOUT, 0),
        retries=retries
        if retries is not None
        else _load_int_env("REQUIREMENT_MAKER_RETRIES", DEFAULT_RETRIES, 0, 10),
        verbose=verbose,
        quiet=quiet,
    )


def _print_runtime_config(config: RuntimeConfig) -> None:
    if not config.verbose:
        return
    click.echo("Runtime configuration:")
    click.echo(f"  requirement model: {config.requirement_model}")
    click.echo(f"  transcription model: {config.transcription_model}")
    click.echo(f"  concurrency: {config.concurrency}")
    click.echo(f"  timeout: {config.timeout}s")
    click.echo(f"  retries: {config.retries}")
    click.echo("  precedence: CLI flags > environment/.env > defaults")


def _load_credentials() -> tuple[str, str]:
    load_dotenv(dotenv_path=find_dotenv(usecwd=True))
    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    missing = [
        name
        for name, value in (
            ("OPENAI_API_KEY", openai_key),
            ("ANTHROPIC_API_KEY", anthropic_key),
        )
        if not value
    ]
    if missing:
        _fail(
            "Missing required credential(s): "
            f"{', '.join(missing)}\n"
            "Repair: copy the example config with `cp .env.example .env`, add provider keys, "
            "or export the variables in your shell. Secret values are never printed."
        )
    return openai_key or "", anthropic_key or ""


def _run_doctor() -> None:
    load_dotenv(dotenv_path=find_dotenv(usecwd=True))
    click.echo("Diagnostics")
    click.echo(f"  Python: {'.'.join(map(str, __import__('sys').version_info[:3]))}")
    click.echo(f"  requirement-maker: {_package_version()}")
    click.echo(f"  ffmpeg: {'found' if shutil.which('ffmpeg') else 'missing - install ffmpeg'}")
    click.echo(f"  ffprobe: {'found' if shutil.which('ffprobe') else 'missing - install ffmpeg'}")
    click.echo(
        "  OPENAI_API_KEY: "
        f"{'configured' if os.environ.get('OPENAI_API_KEY') else 'missing'}"
    )
    click.echo(
        "  ANTHROPIC_API_KEY: "
        f"{'configured' if os.environ.get('ANTHROPIC_API_KEY') else 'missing'}"
    )
    click.echo("  Repair: install with `.venv/bin/python -m pip install -e .` and configure .env.")


async def run_pipeline(
    input_file: Path,
    output: Path,
    config: RuntimeConfig,
    openai_key: str,
    anthropic_key: str,
) -> None:
    _echo(f"Processing: {input_file}", config)

    _echo("Extracting audio...", config)
    audio_chunks = await prepare_audio(input_file)
    _echo(f"  {len(audio_chunks)} audio segment(s) ready", config)

    def on_chunk_done(completed: int, total: int) -> None:
        _echo(f"  Transcribed chunk {completed}/{total}", config)

    _echo("Transcribing (parallel)...", config)
    transcript = await transcribe_chunks(
        audio_chunks,
        openai_key,
        on_chunk_done,
        TranscriptionConfig(
            model=config.transcription_model,
            concurrency=config.concurrency,
            timeout=config.timeout,
            retries=config.retries,
        ),
    )
    _echo(f"  Transcript: {len(transcript)} characters", config)

    _echo(f"Generating requirements (model: {config.requirement_model})...", config)
    requirements = generate_requirements(
        transcript,
        anthropic_key,
        GenerationConfig(
            model=config.requirement_model,
            timeout=config.timeout,
            retries=config.retries,
        ),
    )

    output.write_text(requirements, encoding="utf-8")
    _echo(f"Done: {output}", config)


@click.command(
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog=(
        "Examples:\n"
        "  requirement-maker meeting.mp4\n"
        "  requirement-maker meeting.mp3 -o spec.md --verbose\n"
        "  requirement-maker meeting.mp4 --model claude-haiku-4-5-20251001 "
        "--transcription-model whisper-1 --concurrency 4\n"
        "  requirement-maker --doctor"
    ),
)
@click.version_option(_package_version(), prog_name="requirement-maker")
@click.argument("input_file", required=False, type=click.Path(path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None, help="Output markdown file path.")
@click.option("--model", default=None, help="Anthropic requirement-generation model. Env: REQUIREMENT_MAKER_MODEL.")
@click.option("--transcription-model", default=None, help="OpenAI transcription model. Env: REQUIREMENT_MAKER_TRANSCRIPTION_MODEL.")
@click.option("--concurrency", type=click.IntRange(1, 50), default=None, help="Maximum concurrent transcription calls. Env: REQUIREMENT_MAKER_CONCURRENCY.")
@click.option("--timeout", type=click.FloatRange(min=0, min_open=True), default=None, help="Provider timeout in seconds. Env: REQUIREMENT_MAKER_TIMEOUT.")
@click.option("--retries", type=click.IntRange(0, 10), default=None, help="Provider retry count. Env: REQUIREMENT_MAKER_RETRIES.")
@click.option("--verbose", is_flag=True, help="Print pipeline stages and resolved runtime configuration.")
@click.option("--quiet", is_flag=True, help="Suppress non-error progress output on success.")
@click.option("--doctor", is_flag=True, help="Run safe environment diagnostics and setup guidance.")
def main(
    input_file: Path | None,
    output: Path | None,
    model: str | None,
    transcription_model: str | None,
    concurrency: int | None,
    timeout: float | None,
    retries: int | None,
    verbose: bool,
    quiet: bool,
    doctor: bool,
) -> None:
    """Convert audio/video recordings into comprehensive requirement documents."""
    if doctor:
        _run_doctor()
        return
    if input_file is None:
        raise click.UsageError("Missing argument 'INPUT_FILE'.")
    _validate_input_path(input_file)
    if output is None:
        output = input_file.with_name(f"{input_file.stem}_requirements.md")

    config = _resolve_config(
        model=model,
        transcription_model=transcription_model,
        concurrency=concurrency,
        timeout=timeout,
        retries=retries,
        verbose=verbose,
        quiet=quiet,
    )
    openai_key, anthropic_key = _load_credentials()
    _print_runtime_config(config)

    asyncio.run(run_pipeline(input_file, output, config, openai_key, anthropic_key))
