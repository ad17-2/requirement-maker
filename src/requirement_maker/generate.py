from dataclasses import dataclass

import anthropic

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

    text_blocks = [block.text for block in message.content if block.type == "text"]
    return "\n".join(text_blocks)
