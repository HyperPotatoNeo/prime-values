"""Run unchanged against frozen and candidate sources before comparing receipts.

Uses real Qwen/FA2 layers and complete gradients; no model or attention mocks.
Deterministic FA2 backward makes same-layout old/new comparisons reproducible.
BF16 model/gradient storage matches the existing serving parity fixture; this
does not validate FSDP mixed-precision optimizer/reduction behavior.
"""

import argparse
import copy
import functools
import hashlib
import inspect
import json
from pathlib import Path

import torch
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

from prime_rl.configs.trainer import DefaultLossConfig
from prime_rl.configs.value import ClassificationValueLossConfig
from prime_rl.trainer.model import ValueFunctionModel, forward, predict_value
from prime_rl.trainer.models.layers.attn import FlashAttention
from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head
from prime_rl.trainer.models.qwen3 import Qwen3ForCausalLM
from prime_rl.trainer.rl.loss import compute_loss, setup_rl_loss_fn
from prime_rl.utils.utils import default_dtype
from prime_rl.value.batch import pack_value_inputs, pack_value_samples
from prime_rl.value.inference import predict_value_microbatches, reassemble_value_outputs
from prime_rl.value.math import align_value_logits, compute_value_loss
from prime_rl.value.types import ValueTrainingSample


def make_model(critic=False):
    torch.manual_seed(812)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=192,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=64,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "flash_attention_2"
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = Qwen3ForCausalLM._from_config(config)
        if critic:
            model = ValueFunctionModel(model, 3)
        else:
            inject_prime_lm_head(model, chunk_size=None)
    for module in model.modules():
        if isinstance(module, FlashAttention):
            module._flash_attn_call = functools.partial(module._flash_attn_call, deterministic=True)
    return model


def samples():
    # Two initial private/context tokens and an internal context token are not
    # supervised, but remain causal inputs. Unequal lengths force reassembly.
    return [
        ValueTrainingSample([31, 32, 1, 9, 4], [False, False, True, False, True], [0, 0, -0.5, 0, 0.75]),
        ValueTrainingSample([33, 34, 3], [False, False, True], [0, 0, -0.25]),
        ValueTrainingSample([5], [False], [0]),
        ValueTrainingSample(
            [35, 36, 2, 11, 13, 4, 8], [False, False, True, False, True, False, True], [0, 0, -0.75, 0, 0.5, 0, 1]
        ),
    ]


def metadata_kwargs(lengths, mode):
    if mode == "position":
        return {}
    from prime_rl.utils.sequence import prepare_packed_sequence_metadata

    return {"packing_metadata": prepare_packed_sequence_metadata(lengths, total_tokens=sum(lengths), device="cuda")}


def tensor_inputs(tokens, lengths):
    return (
        torch.tensor([tokens], device="cuda"),
        torch.tensor([[position for length in lengths for position in range(length)]], device="cuda"),
    )


def actor_output(model, tokens, lengths, mode):
    ids, positions = tensor_inputs(tokens, lengths)
    return forward(model, ids, positions, **metadata_kwargs(lengths, mode))["logits"]


def actor_loss(logits, tokens, masks, lengths, scale):
    # State t-1 predicts token t; all segment-start/padding masks are false.
    previous = torch.cat([logits[:, :1] * 0, logits[:, :-1]], dim=1)
    token_ids = torch.tensor([tokens], device="cuda")
    selected = previous.float().log_softmax(-1).gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)[0]
    mask = torch.tensor(masks, device="cuda", dtype=torch.bool)
    advantages = torch.tensor([0.4 if token % 2 else -0.7 for token in tokens], device="cuda")
    loss, _ = compute_loss(
        trainer_logprobs=list(selected.split(lengths)),
        inference_logprobs=list(torch.full_like(selected, -4.25).split(lengths)),
        ref_logprobs=None,
        advantages=list(advantages.split(lengths)),
        loss_mask=list(mask.split(lengths)),
        rl_weights=None,
        ce_weights=None,
        ref_kl_weights=None,
        rl_loss_fn=setup_rl_loss_fn(DefaultLossConfig()),
        rl_scale=scale,
        ce_scale=1,
        ref_kl_scale=1,
    )
    return loss


def compare_gradients_and_step(model, reference, result, prefix):
    actual = dict(model.named_parameters())
    worst_relative = 0.0
    originals = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    for name, expected in reference.named_parameters():
        parameter = actual[name]
        assert parameter.grad is not None and expected.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        delta = (parameter.grad.float() - expected.grad.float()).norm()
        norm = expected.grad.float().norm()
        # Fixed before candidate execution; must pass the frozen source first.
        assert delta <= 0.02 * norm + 1e-5, (name, float(delta), float(norm))
        worst_relative = max(worst_relative, float(delta / (norm + 1e-5)))
        result[f"{prefix}/gradient/{name}"] = parameter.grad.detach().cpu().clone()
    # A real optimizer update, including every model parameter.
    torch.optim.SGD(model.parameters(), lr=0.05).step()
    torch.optim.SGD(reference.parameters(), lr=0.05).step()
    changed = 0
    for name, expected in reference.named_parameters():
        parameter = actual[name]
        torch.testing.assert_close(parameter, expected, rtol=0.02, atol=2e-4)
        changed += int(not torch.equal(parameter, originals[name]))
        result[f"{prefix}/updated/{name}"] = parameter.detach().cpu().clone()
        result[f"{prefix}/update/{name}"] = (parameter.float() - originals[name].float()).detach().cpu()
    assert changed > 0, "optimizer comparison must contain a nonzero update"
    return {"worst_gradient_relative_l2": worst_relative, "changed_parameters": changed}


def actor_case(mode, result):
    model = make_model()
    reference = copy.deepcopy(model)
    data = samples()
    real_lengths = [len(sample.token_ids) for sample in data]
    lengths = [*real_lengths, 4]
    tokens = [token for sample in data for token in sample.token_ids] + [0] * 4
    masks = [mask for sample in data for mask in sample.mask] + [False] * 4
    scale = sum(masks)
    raw = actor_output(model, tokens, lengths, mode)
    separate = [actor_output(reference, sample.token_ids, [len(sample.token_ids)], "position") for sample in data]
    expected = torch.cat(separate, dim=1)
    torch.testing.assert_close(raw[:, : sum(real_lengths)], expected, rtol=2e-3, atol=2e-3)
    loss = actor_loss(raw, tokens, masks, lengths, scale)
    expected_loss = sum(
        actor_loss(output, sample.token_ids, sample.mask, [len(sample.token_ids)], scale)
        for output, sample in zip(separate, data, strict=True)
    )
    torch.testing.assert_close(loss, expected_loss, rtol=2e-3, atol=2e-4)
    loss.backward()
    expected_loss.backward()
    result["actor/logits"] = raw.detach().cpu()
    result["actor/loss"] = loss.detach().cpu()
    # Isolation has a positive control: the perturbed document must change.
    with torch.no_grad():
        changed_tokens = [token + 10 for token in data[0].token_ids] + tokens[real_lengths[0] :]
        changed = actor_output(model, changed_tokens, lengths, mode)
        torch.testing.assert_close(changed[:, real_lengths[0] :], raw[:, real_lengths[0] :], rtol=0, atol=0)
        assert not torch.equal(changed[:, : real_lengths[0]], raw[:, : real_lengths[0]])
    return compare_gradients_and_step(model, reference, result, "actor")


def critic_case(mode, result):
    model = make_model(critic=True)
    reference = copy.deepcopy(model)
    data = samples()
    loss_config = ClassificationValueLossConfig(num_bins=3, reward_range=(-1, 1))
    scale = sum(sum(sample.mask) for sample in data)
    expected_terms, expected_values = [], []
    support = torch.tensor([-1.0, 0.0, 1.0], device="cuda")
    for sample in data:
        ids, positions = tensor_inputs(sample.token_ids, [len(sample.token_ids)])
        raw = predict_value(reference, ids, positions)
        aligned = torch.cat([raw[:, :1] * 0, raw[:, :-1]], dim=1)
        expected_values.append((aligned[0].float().softmax(-1) * support).sum(-1))
        for index, trained in enumerate(sample.mask):
            if trained:
                target = sample.targets[index]
                probabilities = torch.tensor([max(-target, 0), 1 - abs(target), max(target, 0)], device="cuda")
                expected_terms.append(-(probabilities * raw[0, index - 1].float().log_softmax(-1)).sum())
    expected_loss = torch.stack(expected_terms).mean()
    expected_loss.backward()
    grid = pack_value_samples(data, seq_len=8, world_size=3)
    assert any(batch.sample_indices == [-1] for rank in grid for batch in rank)
    total = torch.zeros((), device="cuda")
    for index, batch in enumerate(batch for rank in grid for batch in rank):
        ids, positions = tensor_inputs(batch.input_ids, batch.sequence_lengths)
        raw = predict_value(model, ids, positions, **metadata_kwargs(batch.sequence_lengths, mode))
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
        result[f"critic/aligned/{index}"] = aligned.detach().cpu()
    torch.testing.assert_close(total, expected_loss, rtol=2e-3, atol=2e-4)
    result["critic/loss"] = total.cpu()
    with torch.no_grad():
        tokens = [sample.token_ids for sample in data]
        packed = pack_value_inputs(tokens, seq_len=8, world_size=3)
        indexed = [
            output
            for rank in packed
            for output in predict_value_microbatches(model, rank, device=torch.device("cuda"), loss=loss_config)
        ]
        assert len(indexed) == len(data), "dummy rank must not be published"
        served = reassemble_value_outputs(indexed[::-1], list(map(len, tokens)))
        for index, (actual, expected) in enumerate(zip(served, expected_values, strict=True)):
            actual_tensor = torch.tensor(actual, device="cuda")
            torch.testing.assert_close(actual_tensor, expected, rtol=2e-3, atol=2e-3)
            result[f"critic/served/{index}"] = actual_tensor.cpu()
        changed_tokens = [[token + 10 for token in tokens[0]], *tokens[1:]]
        changed_grid = pack_value_inputs(changed_tokens, seq_len=8, world_size=3)
        changed_indexed = [
            output
            for rank in changed_grid
            for output in predict_value_microbatches(model, rank, device=torch.device("cuda"), loss=loss_config)
        ]
        changed_values = reassemble_value_outputs(changed_indexed, list(map(len, changed_tokens)))
        assert changed_values[1:] == served[1:], "critic values leaked across packed documents"
        assert changed_values[0] != served[0], "critic isolation requires a nonconstant positive control"
    return compare_gradients_and_step(model, reference, result, "critic")


def routed_moe_case(mode, result):
    from prime_rl.trainer.models.layers.moe import MoE
    from prime_rl.trainer.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(910)
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=128,
        head_dim=32,
        intermediate_size=192,
        moe_intermediate_size=96,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=64,
        use_grouped_mm=False,
        norm_topk_prob=True,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "flash_attention_2"
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = Qwen3MoeForCausalLM._from_config(config)
        inject_prime_lm_head(model, chunk_size=None)
    # HF's generic initialization does not initialize custom torch.empty
    # expert parameters. Use the real MoE initialization contract explicitly.
    torch.manual_seed(911)
    for module in model.modules():
        if isinstance(module, MoE):
            module.init_weights(config.initializer_range, buffer_device=torch.device("cuda"))
            for weight in (module.experts.w1, module.experts.w2, module.experts.w3):
                assert torch.isfinite(weight).all() and weight.float().std() > 0
                assert not torch.equal(weight[0], weight[1])
    for name, parameter in model.named_parameters():
        assert torch.isfinite(parameter).all(), name
        result[f"moe/initial/{name}"] = parameter.detach().cpu().clone()
    for module in model.modules():
        if isinstance(module, FlashAttention):
            module._flash_attn_call = functools.partial(module._flash_attn_call, deterministic=True)
    tokens, lengths = [1, 2, 3, 4, 5, 6, 7, 8], [5, 3]
    ids, positions = tensor_inputs(tokens, lengths)
    # Distinct experts per token, varying by token and layer; no routing mocks.
    first = torch.arange(16, device="cuda").reshape(1, 8, 2, 1) % 4
    routes = torch.cat([first, (first + 1) % 4], dim=-1)
    reference = copy.deepcopy(model)
    expected = forward(reference, ids, positions, routed_experts=routes)["logits"]
    actual = forward(model, ids, positions, routed_experts=routes, **metadata_kwargs(lengths, mode))["logits"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    weights = torch.arange(actual.numel(), device="cuda").reshape_as(actual).float().sin()
    (actual.float() * weights).mean().backward()
    (expected.float() * weights).mean().backward()
    result["moe/routed_logits"] = actual.detach().cpu()
    actual_parameters = dict(model.named_parameters())
    checked = 0
    for name, expected_parameter in reference.named_parameters():
        parameter = actual_parameters[name]
        assert (parameter.grad is None) == (expected_parameter.grad is None), name
        if parameter.grad is not None:
            torch.testing.assert_close(parameter.grad, expected_parameter.grad, rtol=0, atol=0)
            result[f"moe/gradient/{name}"] = parameter.grad.detach().cpu()
            checked += 1
    assert checked > 0
    with torch.no_grad():
        changed = forward(model, ids, positions, routed_experts=(routes + 1) % 4, **metadata_kwargs(lengths, mode))[
            "logits"
        ]
        assert not torch.equal(changed, actual), "routed-expert argument must affect the actual expert computation"
    return {
        "gradient_parameters": checked,
        "routing_positive_control": True,
        "expert_initialization": "seed 911, real MoE.init_weights, complete finite initial state recorded",
    }


def hf_fallback_case(mode, result):
    from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

    torch.manual_seed(913)
    config = Qwen3Config(
        vocab_size=64,
        hidden_size=128,
        intermediate_size=192,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=64,
        attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "sdpa"
    with torch.device("cuda"), default_dtype(torch.bfloat16):
        model = ValueFunctionModel(HFQwen3ForCausalLM._from_config(config), 3)
    assert not model.supports_packed_sequences
    tokens = [[1, 2, 3], [4, 5, 6, 7, 8]]
    grid = pack_value_inputs(tokens, seq_len=16, pack_sequences=model.supports_packed_sequences)
    assert len(grid[0]) == 2
    with torch.no_grad():
        indexed = predict_value_microbatches(
            model, grid[0], device=torch.device("cuda"), loss=ClassificationValueLossConfig(num_bins=3)
        )
    for index, values in indexed:
        result[f"hf_fallback/served/{index}"] = torch.tensor(values)
    if mode == "explicit":
        from prime_rl.trainer.model import model_supports_packed_sequence_metadata

        assert not model_supports_packed_sequence_metadata(model)
        ids, positions = tensor_inputs(tokens[0], [len(tokens[0])])
        try:
            predict_value(model, ids, positions, **metadata_kwargs([len(tokens[0])], mode))
        except ValueError as error:
            assert "does not support explicit packing metadata" in str(error)
        else:
            raise AssertionError("unsupported HF backbone must reject explicit packing metadata")
    return {"separate_sequences": len(grid[0]), "implicit_packing": False}


def save_result(output, mode, tensors, metrics, reference=None):
    output.mkdir(parents=True, exist_ok=True)
    comparison = None
    if reference:
        frozen = torch.load(reference, map_location="cpu", weights_only=True)
        assert frozen.keys() == tensors.keys()
        unequal = [name for name in tensors if not torch.equal(tensors[name], frozen[name])]
        comparison = {"tensors": len(tensors), "unequal": unequal, "exact": not unequal}
        assert not unequal, f"identical-layout frozen/candidate tensor differences: {unequal}"
    torch.save(tensors, output / "tensors.pt")
    root = Path(inspect.getfile(ValueFunctionModel)).resolve().parents[2]
    paths = [
        "prime_rl/utils/sequence.py",
        "prime_rl/utils/cp.py",
        "prime_rl/trainer/model.py",
        "prime_rl/trainer/models/qwen3/modeling_qwen3.py",
        "prime_rl/trainer/models/qwen3_moe/modeling_qwen3_moe.py",
        "prime_rl/trainer/models/layers/attn.py",
        "prime_rl/trainer/models/layers/ulysses_attn.py",
        "prime_rl/trainer/rl/loss.py",
        "prime_rl/value/batch.py",
        "prime_rl/value/inference.py",
        "prime_rl/value/math.py",
    ]
    hashes = {str(root / path): hashlib.sha256((root / path).read_bytes()).hexdigest() for path in paths}
    receipt = {
        "mode": mode,
        "source_hashes": hashes,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "precision_scope": "BF16 model and gradient storage, SGD step; production FSDP optimizer/reduction behavior is outside this probe",
        "metrics": metrics,
        "comparison": comparison,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["position", "explicit"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    assert torch.cuda.is_available(), "this is a real GPU oracle, not a CPU fallback"
    tensors = {}
    metrics = {
        "actor": actor_case(args.mode, tensors),
        "critic": critic_case(args.mode, tensors),
        "routed_moe": routed_moe_case(args.mode, tensors),
        "hf_fallback": hf_fallback_case(args.mode, tensors),
    }
    save_result(args.output, args.mode, tensors, metrics, args.reference)


if __name__ == "__main__":
    main()
