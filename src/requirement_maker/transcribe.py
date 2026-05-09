import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from openai import AsyncOpenAI

MAX_CONCURRENT = 10


@dataclass(frozen=True)
class TranscriptionConfig:
    model: str = "whisper-1"
    concurrency: int = MAX_CONCURRENT
    timeout: float = 60.0
    retries: int = 2


async def _transcribe_one(
    client: AsyncOpenAI,
    path: Path,
    semaphore: asyncio.Semaphore,
    on_complete: Callable[[int], None] | None = None,
    index: int = 0,
    model: str = "whisper-1",
) -> str:
    async with semaphore:
        with open(path, "rb") as f:
            response = await client.audio.transcriptions.create(
                model=model,
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        if on_complete:
            on_complete(index)
        return response.text


async def transcribe_chunks(
    audio_paths: list[Path],
    api_key: str,
    on_chunk_complete: Callable[[int, int], None] | None = None,
    config: TranscriptionConfig | None = None,
) -> str:
    if config is None:
        config = TranscriptionConfig()
    client = AsyncOpenAI(api_key=api_key, timeout=config.timeout, max_retries=config.retries)
    semaphore = asyncio.Semaphore(config.concurrency)
    total = len(audio_paths)

    def _notify(index: int) -> None:
        if on_chunk_complete:
            on_chunk_complete(index + 1, total)

    tasks = [
        _transcribe_one(client, path, semaphore, _notify, i, config.model)
        for i, path in enumerate(audio_paths)
    ]
    results = await asyncio.gather(*tasks)
    return "\n\n".join(results)
