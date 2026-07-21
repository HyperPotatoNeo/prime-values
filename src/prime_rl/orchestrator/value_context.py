from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class TokenPrefix:
    """Tokens inserted at one stable position in every branch of a rollout."""

    token_ids: tuple[int, ...]
    insert_at: Literal[0, 1]

    def __post_init__(self) -> None:
        if self.insert_at not in (0, 1):
            raise ValueError(f"prefix insertion position must be 0 or 1, got {self.insert_at}")

    def apply(self, token_ids: list[int]) -> list[int]:
        if self.insert_at > len(token_ids):
            raise ValueError(f"prefix insertion position {self.insert_at} exceeds sequence length {len(token_ids)}")
        return token_ids[: self.insert_at] + list(self.token_ids) + token_ids[self.insert_at :]

    def project(self, values: list[T]) -> list[T]:
        end = self.insert_at + len(self.token_ids)
        if end > len(values):
            raise ValueError(f"prefixed span [{self.insert_at}, {end}) exceeds sequence length {len(values)}")
        return values[: self.insert_at] + values[end:]

    def lift(self, values: list[T], *, fill: T) -> list[T]:
        if self.insert_at > len(values):
            raise ValueError(f"prefix insertion position {self.insert_at} exceeds sequence length {len(values)}")
        return values[: self.insert_at] + [fill] * len(self.token_ids) + values[self.insert_at :]
