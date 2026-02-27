"""Compaction — summarize old messages to stay within context limits."""

from __future__ import annotations

from mini_agent.providers.anthropic import AnthropicProvider

_SUMMARY_SYSTEM = (
    "You are a helpful assistant that summarizes conversation history concisely."
)

_SUMMARY_PROMPT = (
    "Summarize the following conversation history concisely.\n"
    "Preserve: key decisions, file paths mentioned, code changes made, errors encountered.\n"
    "Drop: verbose tool outputs, intermediate reasoning.\n\n"
    "Conversation history:\n{history}"
)


async def compact_messages(
    provider: AnthropicProvider,
    messages: list[dict],
    system_prompt: str,
    keep_recent: int = 4,
    model: str | None = None,
) -> list[dict]:
    """
    Compress the message list by summarizing older messages.

    Strategy (mirrors Pi):
    1. Keep the most recent `keep_recent` messages intact.
    2. Send the older messages to the LLM for summarization.
    3. Return [summary user message] + [recent messages].

    The returned list always starts with a user message (Anthropic requirement).
    """
    if len(messages) <= keep_recent:
        return messages  # nothing to compact

    older = messages[:-keep_recent]
    recent = messages[-keep_recent:]

    # Build a text representation of the older messages
    history_lines: list[str] = []
    for msg in older:
        role = msg.get("role", "?")
        content = msg.get("content", "")
        if isinstance(content, str):
            history_lines.append(f"[{role}] {content}")
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    btype = block.get("type", "")
                    if btype == "text":
                        history_lines.append(f"[{role}] {block.get('text', '')}")
                    elif btype == "tool_use":
                        history_lines.append(
                            f"[tool_call:{block.get('name','')}] {str(block.get('input',''))[:200]}"
                        )
                    elif btype == "tool_result":
                        history_lines.append(
                            f"[tool_result] {str(block.get('content',''))[:200]}"
                        )

    history_text = "\n".join(history_lines)
    summary_request = _SUMMARY_PROMPT.format(history=history_text)

    summary_messages = [{"role": "user", "content": summary_request}]

    response = await provider.complete(
        messages=summary_messages,
        system_prompt=_SUMMARY_SYSTEM,
        tools=None,
        model=model,
    )

    # Extract the summary text
    summary_text = ""
    for block in response.content:
        if hasattr(block, "text"):
            summary_text += block.text

    summary_message: dict = {
        "role": "user",
        "content": f"[Previous conversation summary]\n{summary_text}",
    }

    # The compacted list must start with a user message
    result = [summary_message] + list(recent)

    # Ensure first message is user (Anthropic requirement)
    # If recent starts with assistant, prepend our summary (already done above)
    return result
