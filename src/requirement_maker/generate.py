import json
from dataclasses import dataclass
from typing import Any

import anthropic

from requirement_maker.provider_errors import (
    ProviderStage,
    classify_provider_exception,
    retry_exhausted,
)
from requirement_maker.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


@dataclass(frozen=True)
class GenerationConfig:
    model: str = "claude-sonnet-4-5-20250929"
    timeout: float = 60.0
    retries: int = 2


def generate_requirements(
    transcript: str,
    api_key: str,
    model: str | GenerationConfig = "claude-sonnet-4-5-20250929",
) -> str:
    if isinstance(model, GenerationConfig):
        config = model
    else:
        config = GenerationConfig(model=model)
    client = anthropic.Anthropic(
        api_key=api_key,
        timeout=config.timeout,
        max_retries=config.retries,
    )

    try:
        message = client.messages.create(
            model=config.model,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(transcript=transcript),
                }
            ],
        )
    except Exception as exc:
        error = classify_provider_exception(
            exc,
            stage=ProviderStage.REQUIREMENT_GENERATION,
            provider="Anthropic",
        )
        if error.transient:
            raise retry_exhausted(error) from exc
        raise error from exc

    text_blocks = [block.text for block in message.content if block.type == "text"]
    return "\n".join(text_blocks)


def generate_structured_json(
    prompt: str,
    api_key: str,
    config: GenerationConfig,
) -> dict[str, Any]:
    client = anthropic.Anthropic(
        api_key=api_key,
        timeout=config.timeout,
        max_retries=config.retries,
    )
    try:
        message = client.messages.create(
            model=config.model,
            max_tokens=8000,
            system=(
                "You are a structured workflow stage. Return only valid JSON matching "
                "the user's requested schema. Do not include markdown fences, prose, "
                "or commentary outside the JSON object."
            ),
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as exc:
        error = classify_provider_exception(
            exc,
            stage=ProviderStage.REQUIREMENT_GENERATION,
            provider="Anthropic",
        )
        if error.transient:
            raise retry_exhausted(error) from exc
        raise error from exc

    response = "\n".join(block.text for block in message.content if block.type == "text")
    response = _extract_json_object(response)
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"provider returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("provider returned JSON that is not an object")
    return parsed


def _extract_json_object(response: str) -> str:
    stripped = response.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        return stripped
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start >= 0 and end > start:
        return stripped[start : end + 1]
    return stripped
