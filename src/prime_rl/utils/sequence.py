from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate

import torch


@dataclass(frozen=True, eq=False)
class PackedSequenceMetadata:
    lengths: tuple[int, ...]
    offsets: tuple[int, ...]
    max_seqlen: int
    cu_seqlens: torch.Tensor

    @property
    def total_tokens(self) -> int:
        return self.offsets[-1]

    def validate_shape(self, batch_size: int, sequence_length: int, device: torch.device) -> None:
        if batch_size != 1 or sequence_length != self.total_tokens:
            raise ValueError(
                f"Packing metadata requires shape (1, {self.total_tokens}), got ({batch_size}, {sequence_length})"
            )
        if self.cu_seqlens.device != device:
            raise ValueError(f"Packing metadata device {self.cu_seqlens.device} does not match inputs {device}")

    def for_cp(self, cp_rank: int, cp_world_size: int) -> PackedSequenceMetadata:
        if type(cp_world_size) is not int or cp_world_size < 1:
            raise ValueError("Context parallel world size must be a positive integer")
        if type(cp_rank) is not int or not 0 <= cp_rank < cp_world_size:
            raise ValueError("Context parallel rank is outside the world size")
        if self.total_tokens % cp_world_size:
            raise ValueError("Packed token count must be divisible by context parallel world size")
        if cp_world_size == 1:
            return self
        chunk = self.total_tokens // cp_world_size
        start, end = cp_rank * chunk, (cp_rank + 1) * chunk
        lengths = [min(right, end) - max(left, start) for left, right in zip(self.offsets, self.offsets[1:])]
        return prepare_packed_sequence_metadata(
            [length for length in lengths if length > 0], total_tokens=chunk, device=self.cu_seqlens.device
        )


def prepare_packed_sequence_metadata(
    lengths: Sequence[int], *, total_tokens: int, device: torch.device | str
) -> PackedSequenceMetadata:
    """Prepare varlen boundaries from CPU packing lengths without reading device tensors."""
    if isinstance(lengths, torch.Tensor):
        raise TypeError("Packing lengths must be CPU integer metadata, not a tensor")
    lengths = tuple(lengths)
    if not lengths or any(type(length) is not int or length <= 0 for length in lengths):
        raise ValueError("Packing lengths must be nonempty positive integers")
    if type(total_tokens) is not int or total_tokens <= 0 or sum(lengths) != total_tokens:
        raise ValueError("Packing lengths must sum to the positive input token count")
    if total_tokens > torch.iinfo(torch.int32).max:
        raise ValueError("Packed token count exceeds int32 cumulative-offset capacity")
    offsets = tuple(accumulate(lengths, initial=0))
    return PackedSequenceMetadata(
        lengths=lengths,
        offsets=offsets,
        max_seqlen=max(lengths),
        cu_seqlens=torch.tensor(offsets, dtype=torch.int32, device="cpu").to(device, non_blocking=True),
    )


def get_cu_seqlens_from_position_ids(position_ids: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Build local-relative cumulative sequence lengths from packed position ids.

    ``position_ids`` may be a full packed sequence or a context-parallel local shard.
    In the shard case, the first local token can be a continuation with a non-zero
    position id, so cumulative offsets must be anchored to the local tensor rather
    than to the absolute position id value.
    """
    flat_position_ids = position_ids.view(-1)
    total_tokens = flat_position_ids.numel()
    assert total_tokens > 0, "Cannot build cu_seqlens for an empty position_ids tensor"

    zero_starts = (flat_position_ids == 0).nonzero(as_tuple=True)[0]
    starts = torch.cat(
        [torch.zeros(1, dtype=zero_starts.dtype, device=zero_starts.device), zero_starts]
    ).unique_consecutive()

    ends = torch.cat(
        [
            starts[1:],
            torch.tensor([total_tokens], dtype=starts.dtype, device=starts.device),
        ]
    )
    seqlens = ends - starts

    cu_seqlens = torch.empty(seqlens.numel() + 1, dtype=torch.int32, device=position_ids.device)
    cu_seqlens[0] = 0
    cu_seqlens[1:] = seqlens.cumsum(dim=0, dtype=torch.int32)

    return cu_seqlens, seqlens.max().item()


def get_cu_seqlens_from_seq_lens(seq_lens: torch.Tensor, total_tokens: int | None = None) -> tuple[torch.Tensor, int]:
    if seq_lens.ndim != 1:
        raise ValueError(f"seq_lens must be 1D, got shape={tuple(seq_lens.shape)}")
    if seq_lens.numel() == 0:
        raise ValueError("seq_lens must not be empty")
    if bool((seq_lens <= 0).any().item()):
        raise ValueError(f"seq_lens must be positive, got {seq_lens.tolist()}")
    if total_tokens is not None and int(seq_lens.sum().item()) != total_tokens:
        raise ValueError(f"seq_lens sum must equal sequence length: {seq_lens.tolist()} vs {total_tokens}")

    seq_lens = seq_lens.to(dtype=torch.int32)
    cu_seqlens = torch.empty(seq_lens.numel() + 1, dtype=torch.int32, device=seq_lens.device)
    cu_seqlens[0] = 0
    cu_seqlens[1:] = seq_lens.cumsum(dim=0, dtype=torch.int32)
    return cu_seqlens, int(seq_lens.max().item())
