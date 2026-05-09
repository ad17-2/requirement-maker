import anthropic

from requirement_maker.prompt import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE


def generate_requirements(
    transcript: str,
    api_key: str,
    model: str = "claude-sonnet-4-5-20250929",
) -> str:
    client = anthropic.Anthropic(api_key=api_key)

    message = client.messages.create(
        model=model,
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
