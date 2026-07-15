from __future__ import annotations

import math
from typing import cast

import torch
from torch import nn

from prime_rl.configs.value import ValueLossConfig
from prime_rl.trainer.model import predict_value
from prime_rl.value.batch import ValueInputMicroBatch
from prime_rl.value.math import align_value_logits, predict_values


def predict_value_microbatches(
    model: nn.Module,
    micro_batches: list[ValueInputMicroBatch],
    *,
    device: torch.device,
    loss: ValueLossConfig,
) -> list[tuple[int, list[float]]]:
    indexed: list[tuple[int, list[float]]] = []
    for micro_batch in micro_batches:
        input_ids = torch.tensor(micro_batch.input_ids, dtype=torch.long, device=device).unsqueeze(0)
        position_ids = torch.tensor(micro_batch.position_ids, dtype=torch.long, device=device).unsqueeze(0)
        logits = predict_value(model, input_ids, position_ids)
        logits = align_value_logits(logits, micro_batch.sequence_lengths)
        values = predict_values(logits, loss).reshape(-1)
        offset = 0
        for sample_index, length in zip(micro_batch.sample_indices, micro_batch.sequence_lengths, strict=True):
            if sample_index >= 0:
                indexed.append((sample_index, values[offset : offset + length].float().cpu().tolist()))
            offset += length
    return indexed


def reassemble_value_outputs(
    indexed: list[tuple[int, list[float]]],
    expected_lengths: list[int],
) -> list[list[float]]:
    results: list[list[float] | None] = [None] * len(expected_lengths)
    for sample_index, values in indexed:
        if sample_index < 0 or sample_index >= len(results):
            raise RuntimeError(f"value evaluator returned invalid sample index {sample_index}")
        if results[sample_index] is not None:
            raise RuntimeError(f"value evaluator returned sample {sample_index} more than once")
        if len(values) != expected_lengths[sample_index]:
            raise RuntimeError(
                f"value evaluator sample {sample_index} has {len(values)} values; "
                f"expected {expected_lengths[sample_index]}"
            )
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"value evaluator sample {sample_index} contains non-finite values")
        results[sample_index] = values
    if any(result is None for result in results):
        raise RuntimeError("value evaluator failed to materialize every requested sequence")
    return cast(list[list[float]], results)
