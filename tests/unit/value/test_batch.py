import pytest

from prime_rl.value.batch import pack_value_inputs, pack_value_samples
from prime_rl.value.types import ValueTrainingSample


def _sample(length: int, start: int = 0) -> ValueTrainingSample:
    return ValueTrainingSample(
        token_ids=list(range(start, start + length)),
        mask=[True] * length,
        targets=[1.0] * length,
    )


def test_value_packing_preserves_boundaries_and_balances_microbatch_count():
    grid = pack_value_samples([_sample(4), _sample(3, 10), _sample(2, 20)], seq_len=5, world_size=2)

    assert len(grid[0]) == len(grid[1])
    assert sorted(index for rank in grid for batch in rank for index in batch.sample_indices if index >= 0) == [0, 1, 2]
    for rank in grid:
        for batch in rank:
            assert sum(batch.sequence_lengths) <= 5
            assert len(batch.input_ids) == len(batch.mask) == len(batch.targets)


def test_value_packing_rejects_oversized_samples():
    with pytest.raises(ValueError, match="outside"):
        pack_value_samples([_sample(6)], seq_len=5)


def test_value_packing_can_keep_attention_unsafe_models_unpacked():
    grid = pack_value_samples(
        [_sample(2), _sample(2, 10)],
        seq_len=4,
        pack_sequences=False,
    )

    assert len(grid[0]) == 2
    assert [batch.sequence_lengths for batch in grid[0]] == [[2], [2]]


@pytest.mark.parametrize("pack_sequences", [True, False])
def test_inference_packing_matches_training_layout_including_dummy_ranks(pack_sequences):
    samples = [_sample(4), _sample(3, 10), _sample(2, 20)]
    token_ids = [sample.token_ids for sample in samples]

    training = pack_value_samples(
        samples,
        seq_len=5,
        world_size=4,
        pad_token_id=99,
        pack_sequences=pack_sequences,
    )
    inference = pack_value_inputs(
        token_ids,
        seq_len=5,
        world_size=4,
        pad_token_id=99,
        pack_sequences=pack_sequences,
    )

    assert len(training) == len(inference) == 4
    assert [len(rank) for rank in training] == [len(rank) for rank in inference]
    for training_rank, inference_rank in zip(training, inference, strict=True):
        for training_batch, inference_batch in zip(training_rank, inference_rank, strict=True):
            assert inference_batch.input_ids == training_batch.input_ids
            assert inference_batch.position_ids == training_batch.position_ids
            assert inference_batch.sequence_lengths == training_batch.sequence_lengths
            assert inference_batch.sample_indices == training_batch.sample_indices
    assert any(batch.sample_indices == [-1] for rank in inference for batch in rank)
