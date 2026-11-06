from itertools import groupby

import pytest
import torch

from prime_rl.utils.sequence import prepare_packed_sequence_metadata


def test_cpu_boundaries_and_cp_fragments_match_independent_token_labels():
    # Enumerate all partitions of small token streams. The oracle labels tokens
    # by document, slices tokens and run-length encodes; it never intersects
    # cumulative intervals like the implementation.
    for total in range(1, 10):
        for cuts in range(1 << (total - 1)):
            labels = [0]
            for index in range(total - 1):
                labels.append(labels[-1] + bool(cuts & (1 << index)))
            lengths = [len(list(run)) for _, run in groupby(labels)]
            metadata = prepare_packed_sequence_metadata(lengths, total_tokens=total, device="cpu")
            starts = [index for index in range(total) if index == 0 or labels[index] != labels[index - 1]]
            assert metadata.offsets == tuple([*starts, total])
            assert metadata.max_seqlen == max(lengths)
            assert metadata.total_tokens == total
            assert metadata.cu_seqlens.dtype == torch.int32
            assert metadata.cu_seqlens.tolist() == [*starts, total]
            for world_size in range(1, total + 1):
                if total % world_size:
                    continue
                shard_size = total // world_size
                for rank in range(world_size):
                    shard = labels[rank * shard_size : (rank + 1) * shard_size]
                    expected = tuple(len(list(run)) for _, run in groupby(shard))
                    local = metadata.for_cp(rank, world_size)
                    assert local.lengths == expected
                    assert local.total_tokens == shard_size
                    assert local.max_seqlen == max(expected)
                    assert local.offsets[0] == 0
                    assert local.cu_seqlens.tolist() == list(local.offsets)


def test_metadata_owns_lengths_and_preserves_padding_and_dummy_segments():
    lengths = [5, 3, 4]
    metadata = prepare_packed_sequence_metadata(lengths, total_tokens=12, device="cpu")
    lengths[:] = [12]
    assert metadata.lengths == (5, 3, 4)
    assert metadata.offsets == (0, 5, 8, 12)
    # The second shard starts at position 1 of the second document.
    assert metadata.for_cp(1, 2).lengths == (2, 4)
    dummy = prepare_packed_sequence_metadata([1], total_tokens=1, device="cpu")
    assert dummy.offsets == (0, 1)
    assert dummy.max_seqlen == 1


@pytest.mark.parametrize(
    ("lengths", "total"),
    [([], 0), ([0], 0), ([-1, 3], 2), ([True], 1), ([1.0], 1), ([2, 3], 6), ([2**31], 2**31)],
)
def test_invalid_cpu_metadata_is_rejected_before_device_allocation(lengths, total, monkeypatch):
    def forbidden_allocation(*args, **kwargs):
        raise AssertionError("invalid host metadata reached tensor allocation")

    monkeypatch.setattr(torch, "tensor", forbidden_allocation)
    with pytest.raises((TypeError, ValueError, OverflowError)):
        prepare_packed_sequence_metadata(lengths, total_tokens=total, device="cpu")


def test_invalid_shards_and_tensor_shapes_fail_closed():
    metadata = prepare_packed_sequence_metadata([5, 3, 4], total_tokens=12, device="cpu")
    for rank, world_size in [(-1, 2), (2, 2), (0, 0), (0, 5), (False, 2), (0, 2.0)]:
        with pytest.raises((TypeError, ValueError)):
            metadata.for_cp(rank, world_size)
    metadata.validate_shape(1, 12, torch.device("cpu"))
    for batch_size, sequence_length, device in [(2, 12, "cpu"), (1, 11, "cpu"), (1, 12, "meta")]:
        with pytest.raises(ValueError):
            metadata.validate_shape(batch_size, sequence_length, torch.device(device))
