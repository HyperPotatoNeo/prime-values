"""Numerical contracts across real critic attention, packing and serving paths.

The reference forwards each sequence separately and reads state t-1 directly;
it does not call the production packer, alignment or loss helpers.
"""

import copy

import pytest
import torch
from transformers.models.llama.configuration_llama import LlamaConfig

from prime_rl.configs.value import ClassificationValueLossConfig, MSEValueLossConfig
from prime_rl.trainer.model import ValueFunctionModel, predict_value
from prime_rl.trainer.models.llama import LlamaForCausalLM
from prime_rl.utils.utils import default_dtype
from prime_rl.value.batch import pack_value_inputs, pack_value_samples
from prime_rl.value.inference import predict_value_microbatches, reassemble_value_outputs
from prime_rl.value.math import align_value_logits, compute_value_loss
from prime_rl.value.types import ValueTrainingSample

pytestmark = [pytest.mark.gpu]


def _model(output_size):
    torch.manual_seed(812)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "flash_attention_2"
    # Dedicated evaluators use a BF16 serving copy. Autocast alone leaves the
    # embedding/rotary path in FP32, which is not a valid FlashAttention input.
    # The separate runtime probe covers FP32 optimizer/reduction state in FSDP.
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = ValueFunctionModel(LlamaForCausalLM._from_config(config), output_size)
    assert model.supports_packed_sequences
    return model


def _samples():
    # Unequal lengths force reordering. Internal context tokens remain causal
    # inputs while contributing no critic loss.
    return [
        ValueTrainingSample([1, 9, 4, 8, 2], [False, True, False, True, True], [0, -0.5, 0, 0.25, 0.75]),
        ValueTrainingSample([3, 7, 6], [False, False, True], [0, 0, -0.25]),
        ValueTrainingSample([5], [False], [0]),
        ValueTrainingSample(
            [2, 11, 13, 4, 8, 6, 9], [False, True, True, False, True, False, True], [0, -0.75, 0.5, 0, 0.25, 0, 1.0]
        ),
    ]


def _forward(model, tokens):
    ids = torch.tensor([tokens], device="cuda")
    positions = torch.arange(len(tokens), device="cuda").unsqueeze(0)
    return predict_value(model, ids, positions)


def _reference_loss(model, samples, output_size):
    terms = []
    predictions = []
    for sample in samples:
        raw = _forward(model, sample.token_ids)
        # Boundary logits are zero; a categorical zero-logit state has the
        # uniform support expectation (zero for this symmetric support).
        aligned = torch.cat([raw[:, :1] * 0, raw[:, :-1]], dim=1)
        if output_size == 1:
            predictions.append(aligned[0, :, 0])
        else:
            support = torch.tensor([-1.0, 0.0, 1.0], device="cuda")
            predictions.append((aligned[0].float().softmax(-1) * support).sum(-1))
        for action, trained in enumerate(sample.mask):
            if not trained:
                continue
            target = sample.targets[action]
            state = raw[0, action - 1].float()
            if output_size == 1:
                terms.append((state[0] - target).square())
            else:
                # Independent piecewise barycentric coordinates on [-1,0,1].
                probabilities = torch.tensor([max(-target, 0), 1 - abs(target), max(target, 0)], device="cuda")
                terms.append(-(probabilities * state.log_softmax(-1)).sum())
    return torch.stack(terms).mean(), predictions


@pytest.mark.parametrize("output_size", [1, 3], ids=["mse", "categorical"])
def test_critic_packing_serving_and_gradients_match_independent_sequences(output_size):
    model = _model(output_size)
    reference = copy.deepcopy(model)
    samples = _samples()
    loss_config = (
        MSEValueLossConfig()
        if output_size == 1
        else ClassificationValueLossConfig(num_bins=3, reward_range=(-1.0, 1.0))
    )
    scale = sum(sum(sample.mask) for sample in samples)
    # Two bins over three simulated ranks force a dummy microbatch, which must
    # participate in backward without changing the global objective.
    grid = pack_value_samples(samples, seq_len=8, world_size=3)
    assert any(batch.sample_indices == [-1] for rank in grid for batch in rank)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        expected_loss, expected_predictions = _reference_loss(reference, samples, output_size)
        expected_loss.backward()
        total = torch.zeros((), device="cuda")
        for rank in grid:
            for batch in rank:
                ids = torch.tensor([batch.input_ids], device="cuda")
                positions = torch.tensor([batch.position_ids], device="cuda")
                raw = predict_value(model, ids, positions)
                aligned = align_value_logits(raw, batch.sequence_lengths)
                loss, _ = compute_value_loss(
                    aligned,
                    torch.tensor([batch.targets], device="cuda"),
                    torch.tensor([batch.mask], device="cuda"),
                    loss_config,
                    scale,
                )
                loss.backward()
                total += loss.detach()
        with torch.no_grad():
            inputs = [sample.token_ids for sample in samples]
            packed = pack_value_inputs(inputs, seq_len=8, world_size=3)
            indexed = [
                result
                for rank in packed
                for result in predict_value_microbatches(model, rank, device=torch.device("cuda"), loss=loss_config)
            ]
            served = reassemble_value_outputs(indexed[::-1], [len(tokens) for tokens in inputs])

    torch.testing.assert_close(total, expected_loss, rtol=2e-3, atol=2e-4)
    for actual, expected in zip(served, expected_predictions, strict=True):
        torch.testing.assert_close(torch.tensor(actual, device="cuda"), expected, rtol=2e-3, atol=2e-3)
    actual_parameters = dict(model.named_parameters())
    for name, parameter in reference.named_parameters():
        actual = actual_parameters[name]
        assert parameter.grad is not None and actual.grad is not None, name
        assert torch.isfinite(actual.grad).all(), name
        # BF16 matmul reduction grouping differs between packed and separate
        # forwards. Compare each parameter's gradient direction and magnitude.
        delta = (actual.grad - parameter.grad).float().norm()
        bound = 0.02 * parameter.grad.float().norm() + 1e-5
        assert delta <= bound, f"{name}: gradient L2 error {delta.item()} exceeds {bound.item()}"


def test_critic_cannot_see_current_action_or_neighboring_packed_sequence():
    model = _model(1).eval()
    first = [1, 2, 3, 4, 5]
    second = [6, 7, 8, 9]

    def evaluate(sequences):
        grid = pack_value_inputs(sequences, seq_len=16)
        indexed = predict_value_microbatches(model, grid[0], device=torch.device("cuda"), loss=MSEValueLossConfig())
        return reassemble_value_outputs(indexed, list(map(len, sequences)))

    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        original = evaluate([first, second])
        neighbor_changed = evaluate([[21, 22, 23, 24, 25], second])
        action_changed = evaluate([first, [6, 7, 38, 39]])
        # Include a positive control: changing the preceding state must be
        # visible, so a constant-output model cannot pass the isolation checks.
        context_changed = evaluate([first, [36, 37, 8, 9]])

    torch.testing.assert_close(torch.tensor(original[1]), torch.tensor(neighbor_changed[1]), rtol=0, atol=1e-5)
    torch.testing.assert_close(torch.tensor(original[1][:3]), torch.tensor(action_changed[1][:3]), rtol=0, atol=1e-5)
    assert abs(original[1][2] - context_changed[1][2]) > 1e-3
