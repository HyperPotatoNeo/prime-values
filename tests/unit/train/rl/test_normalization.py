import torch

from prime_rl.trainer.rl.normalization import count_loss_tokens


def test_component_counts_ignore_weight_magnitude_and_keep_observation_losses():
    batch = {
        "loss_mask": torch.tensor([True, True, False, True]),
        "rl_weights": torch.tensor([0.5, 1.25, 20.0, 0.0]),
        "ce_weights": torch.tensor([0.0, 0.5, 2.0, 0.0]),
        "ref_kl_weights": torch.tensor([0.0, 0.0, 0.25, 0.0]),
    }
    assert count_loss_tokens([batch]) == (2, 2, 1)
    partitions = [{key: value[part] for key, value in batch.items()} for part in (slice(0, 1), slice(1, 4))]
    assert count_loss_tokens(partitions) == (2, 2, 1)


def test_implicit_rl_weights_and_empty_components():
    batch = {
        "loss_mask": torch.tensor([True, False, True]),
        "rl_weights": None,
        "ce_weights": None,
        "ref_kl_weights": None,
    }
    assert count_loss_tokens([batch]) == (2, 0, 0)
    assert count_loss_tokens([]) == (0, 0, 0)
