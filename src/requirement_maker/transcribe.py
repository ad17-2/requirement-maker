import asyncio
from collections.abc import Callable
from pathlib import Path

from openai import AsyncOpenAI

MAX_CONCURRENT = 10


async def _transcribe_one(
    client: AsyncOpenAI,
    path: Path,
    semaphore: asyncio.Semaphore,
    on_complete: Callable[[int], None] | None = None,
    index: int = 0,
) -> str:
    async with semaphore:
        with open(path, "rb") as f:
            response = await client.audio.transcriptions.create(
                model="whisper-1",
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
) -> str:
    client = AsyncOpenAI(api_key=api_key)
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    total = len(audio_paths)

    def _notify(index: int) -> None:
        if on_chunk_complete:
            on_chunk_complete(index + 1, total)

    tasks = [
        _transcribe_one(client, path, semaphore, _notify, i)
        for i, path in enumerate(audio_paths)
    ]
    results = await asyncio.gather(*tasks)
    return "\n\n".join(results)
