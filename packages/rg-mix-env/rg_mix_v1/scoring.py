"""RG's single answer-tag extraction and zero-weight format metric."""

import re
from typing import Any

_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)


def _field(message: Any, name: str, default=None):
    return message.get(name, default) if isinstance(message, dict) else getattr(message, name, default)


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and isinstance(part.get("text"), str):
                    chunks.append(part["text"])
            elif isinstance(text := getattr(part, "text", None), str):
                chunks.append(text)
        return " ".join(chunks).strip()
    return ""


def assistant_texts(completion) -> list[str]:
    return [
        _text(_field(message, "content", "") or "") for message in completion if _field(message, "role") == "assistant"
    ]


def parse_answer(completion) -> str | None:
    if isinstance(completion, str):
        # The raw-string API takes the last tag, while message completions take
        # the first tag in the most recent assistant message containing one.
        matches = list(_ANSWER.finditer(completion))
        return matches[-1].group(1).strip() if matches else None
    for content in reversed(assistant_texts(completion)):
        if match := _ANSWER.search(content):
            return match.group(1).strip()
    return None


def response_for_scoring(completion) -> str:
    return str(parse_answer(completion)).strip()


def format_reward(completion) -> float:
    texts = assistant_texts(completion)
    if not texts:
        return 0.0
    scores = []
    for content in texts:
        # With one answer field, the parser's spacing check contributes0.2
        # even when no complete tag exists. This metric has objective weight0.
        score = 0.2
        if _ANSWER.search(content) is not None:
            score += 0.4
        if content.strip().startswith("<answer>"):
            score += 0.2
        if content.strip().endswith("</answer>"):
            score += 0.2
        scores.append(score)
    return sum(scores) / len(scores)
