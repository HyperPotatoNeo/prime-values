from dataclasses import FrozenInstanceError

import pytest

from prime_rl.transport import TrainingSample
from prime_rl.value.trajectory import snapshot_value_trajectory


def test_value_snapshot_copies_native_and_unique_action_masks():
    sample = TrainingSample(
        token_ids=[0, 1, 2],
        mask=[False, False, True],
        logprobs=[0.0, -0.1, -0.1],
        temperatures=[],
        env_name="test",
    )
    native_mask = [False, True, True]
    trajectory = snapshot_value_trajectory([sample], sampled_masks=[native_mask], layout=None, reward=1.0)
    sample.token_ids.clear()
    sample.mask.clear()
    native_mask.clear()
    assert trajectory.samples[0].token_ids == (0, 1, 2)
    assert trajectory.samples[0].mask == (False, False, True)
    assert trajectory.samples[0].sampled_mask == (False, True, True)
    with pytest.raises(FrozenInstanceError):
        trajectory.samples[0].mask = ()


@pytest.mark.parametrize("native_mask", [[False], [False, False]])
def test_value_snapshot_rejects_misaligned_or_non_sampled_trainable_tokens(native_mask):
    sample = TrainingSample(
        token_ids=[0, 1], mask=[False, True], logprobs=[0.0, -0.1], temperatures=[], env_name="test"
    )
    with pytest.raises(ValueError):
        snapshot_value_trajectory([sample], sampled_masks=[native_mask], layout=None, reward=1.0)
