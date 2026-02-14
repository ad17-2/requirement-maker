import asyncio
import os
from pathlib import Path

import click
from dotenv import load_dotenv

from requirement_maker.audio import prepare_audio
from requirement_maker.generate import generate_requirements
from requirement_maker.transcribe import transcribe_chunks


async def run_pipeline(input_file: Path, output: Path, model: str, openai_key: str, anthropic_key: str):
    click.echo(f"Processing: {input_file}")

    click.echo("Extracting audio...")
    audio_chunks = await prepare_audio(input_file)
    click.echo(f"  {len(audio_chunks)} audio segment(s) ready")

    def on_chunk_done(completed: int, total: int):
        click.echo(f"  Transcribed chunk {completed}/{total}")

    click.echo("Transcribing (parallel)...")
    transcript = await transcribe_chunks(audio_chunks, openai_key, on_chunk_done)
    click.echo(f"  Transcript: {len(transcript)} characters")

    click.echo(f"Generating requirements (model: {model})...")
    requirements = generate_requirements(transcript, anthropic_key, model)

    output.write_text(requirements, encoding="utf-8")
    click.echo(f"Done: {output}")


@click.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("-o", "--output", type=click.Path(path_type=Path), default=None, help="Output markdown file path")
@click.option("--model", default="claude-sonnet-4-5-20250929", help="Anthropic model to use")
def main(input_file: Path, output: Path | None, model: str):
    """Convert audio/video recordings into comprehensive requirement documents."""
    load_dotenv()

    openai_key = os.environ.get("OPENAI_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not openai_key:
        raise SystemExit("OPENAI_API_KEY not set. Add it to .env or export it.")
    if not anthropic_key:
        raise SystemExit("ANTHROPIC_API_KEY not set. Add it to .env or export it.")

    if output is None:
        output = input_file.with_name(f"{input_file.stem}_requirements.md")

    asyncio.run(run_pipeline(input_file, output, model, openai_key, anthropic_key))
