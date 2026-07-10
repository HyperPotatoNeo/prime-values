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


def _materialize(samples: list[tuple[int, ValueTrainingSample]]) -> ValueMicroBatch:
    input_ids: list[int] = []
    position_ids: list[int] = []
    mask: list[bool] = []
    targets: list[float] = []
    sequence_lengths: list[int] = []
    sample_indices: list[int] = []
    for sample_index, sample in samples:
        length = len(sample.token_ids)
        if len(sample.mask) != length or len(sample.targets) != length:
            raise ValueError("value sample token, mask, and target streams must align")
        input_ids.extend(sample.token_ids)
        position_ids.extend(range(length))
        mask.extend(sample.mask)
        targets.extend(sample.targets)
        sequence_lengths.append(length)
        sample_indices.append(sample_index)
    return ValueMicroBatch(input_ids, position_ids, mask, targets, sequence_lengths, sample_indices)


def pack_value_samples(
    samples: list[ValueTrainingSample],
    *,
    seq_len: int,
    world_size: int = 1,
    pad_token_id: int = 0,
    pack_sequences: bool = True,
) -> list[list[ValueMicroBatch]]:
    """Build an equal microbatch count per rank, optionally using FFD packing."""
    if not samples:
        raise ValueError("cannot pack an empty value batch")
    ordered = sorted(enumerate(samples), key=lambda item: len(item[1].token_ids), reverse=True)
    for _, sample in ordered:
        length = len(sample.token_ids)
        if length == 0 or length > seq_len:
            raise ValueError(f"value sample length {length} is outside [1, {seq_len}]")

    if pack_sequences:
        bins: list[list[tuple[int, ValueTrainingSample]]] = []
        remaining: list[int] = []
        for sample_index, sample in ordered:
            length = len(sample.token_ids)
            for index, capacity in enumerate(remaining):
                if length <= capacity:
                    bins[index].append((sample_index, sample))
                    remaining[index] -= length
                    break
            else:
                bins.append([(sample_index, sample)])
                remaining.append(seq_len - length)
    else:
        # HF and non-varlen attention do not infer document boundaries from
        # reset position IDs. Keep one sequence per forward so causal attention
        # cannot cross samples.
        bins = [[item] for item in ordered]

    while len(bins) % world_size:
        bins.append([(-1, ValueTrainingSample(token_ids=[pad_token_id], mask=[False], targets=[0.0]))])

    grid: list[list[ValueMicroBatch]] = [[] for _ in range(world_size)]
    for index, packed in enumerate(bins):
        grid[index % world_size].append(_materialize(packed))
    return grid
