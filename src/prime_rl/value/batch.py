from __future__ import annotations

from dataclasses import dataclass

from prime_rl.value.types import ValueTrainingSample


@dataclass(frozen=True)
class ValueMicroBatch:
    input_ids: list[int]
    position_ids: list[int]
    mask: list[bool]
    targets: list[float]
    sequence_lengths: list[int]
    sample_indices: list[int]


@dataclass(frozen=True)
class ValueInputMicroBatch:
    input_ids: list[int]
    position_ids: list[int]
    sequence_lengths: list[int]
    sample_indices: list[int]


def _pack_indices(
    lengths: list[int],
    *,
    seq_len: int,
    world_size: int,
    pack_sequences: bool,
) -> list[list[list[int]]]:
    if not lengths:
        raise ValueError("cannot pack an empty value batch")
    if world_size < 1:
        raise ValueError("value packing world_size must be positive")
    for length in lengths:
        if length == 0 or length > seq_len:
            raise ValueError(f"value sample length {length} is outside [1, {seq_len}]")

    ordered = sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True)
    if pack_sequences:
        bins: list[list[int]] = []
        remaining: list[int] = []
        for sample_index in ordered:
            length = lengths[sample_index]
            for index, capacity in enumerate(remaining):
                if length <= capacity:
                    bins[index].append(sample_index)
                    remaining[index] -= length
                    break
            else:
                bins.append([sample_index])
                remaining.append(seq_len - length)
    else:
        # Models without varlen attention need one sequence per forward.
        bins = [[sample_index] for sample_index in ordered]

    while len(bins) % world_size:
        bins.append([])

    grid: list[list[list[int]]] = [[] for _ in range(world_size)]
    for index, packed in enumerate(bins):
        grid[index % world_size].append(packed)
    return grid


def _materialize_samples(
    sample_indices: list[int],
    samples: list[ValueTrainingSample],
    pad_token_id: int,
) -> ValueMicroBatch:
    if not sample_indices:
        return ValueMicroBatch([pad_token_id], [0], [False], [0.0], [1], [-1])

    input_ids: list[int] = []
    position_ids: list[int] = []
    mask: list[bool] = []
    targets: list[float] = []
    sequence_lengths: list[int] = []
    materialized_indices: list[int] = []
    for sample_index in sample_indices:
        sample = samples[sample_index]
        length = len(sample.token_ids)
        input_ids.extend(sample.token_ids)
        position_ids.extend(range(length))
        mask.extend(sample.mask)
        targets.extend(sample.targets)
        sequence_lengths.append(length)
        materialized_indices.append(sample_index)
    return ValueMicroBatch(input_ids, position_ids, mask, targets, sequence_lengths, materialized_indices)


def _materialize_inputs(
    sample_indices: list[int],
    token_ids: list[list[int]],
    pad_token_id: int,
) -> ValueInputMicroBatch:
    if not sample_indices:
        return ValueInputMicroBatch([pad_token_id], [0], [1], [-1])

    input_ids: list[int] = []
    position_ids: list[int] = []
    sequence_lengths: list[int] = []
    for sample_index in sample_indices:
        tokens = token_ids[sample_index]
        input_ids.extend(tokens)
        position_ids.extend(range(len(tokens)))
        sequence_lengths.append(len(tokens))
    return ValueInputMicroBatch(input_ids, position_ids, sequence_lengths, sample_indices)


def pack_value_samples(
    samples: list[ValueTrainingSample],
    *,
    seq_len: int,
    world_size: int = 1,
    pad_token_id: int = 0,
    pack_sequences: bool = True,
) -> list[list[ValueMicroBatch]]:
    """Build an equal microbatch count per rank, optionally using FFD packing."""
    for sample in samples:
        if len(sample.mask) != len(sample.token_ids) or len(sample.targets) != len(sample.token_ids):
            raise ValueError("value sample token, mask, and target streams must align")
    grid = _pack_indices(
        [len(sample.token_ids) for sample in samples],
        seq_len=seq_len,
        world_size=world_size,
        pack_sequences=pack_sequences,
    )
    return [[_materialize_samples(indices, samples, pad_token_id) for indices in rank] for rank in grid]


def pack_value_inputs(
    token_ids: list[list[int]],
    *,
    seq_len: int,
    world_size: int = 1,
    pad_token_id: int = 0,
    pack_sequences: bool = True,
) -> list[list[ValueInputMicroBatch]]:
    """Pack inference inputs without manufacturing training masks or targets."""
    grid = _pack_indices(
        [len(tokens) for tokens in token_ids],
        seq_len=seq_len,
        world_size=world_size,
        pack_sequences=pack_sequences,
    )
    return [[_materialize_inputs(indices, token_ids, pad_token_id) for indices in rank] for rank in grid]
