"""Literal-token oracles for observation-only policy version diagnostics."""

import copy
import inspect
import os
from pathlib import Path
from types import SimpleNamespace

import msgspec
import numpy as np
import pytest
import torch
import torch.distributed as dist
import verifiers.v1 as vf
from verifiers.v1.configs.agent import WireAgentConfig
from verifiers.v1.types import WeightVersionSpan

from prime_rl.configs.orchestrator import SingleTurnTrainingProjectionConfig
from prime_rl.configs.trainer import DefaultLossConfig
from prime_rl.orchestrator.episodes import from_episode
from prime_rl.orchestrator.trajectories import materialize_training_data
from prime_rl.trainer.batch import prepare_batch
from prime_rl.trainer.rl.data import DataLoader
from prime_rl.trainer.rl.loss import compute_loss, setup_rl_loss_fn
from prime_rl.trainer.rl.metrics import (
    accumulate_policy_provenance,
    accumulate_token_metrics,
    compute_policy_provenance,
    policy_version_reference,
    prepare_token_env_ids,
)
from prime_rl.trainer.utils import TensorSummaries
from prime_rl.transport.provenance import encode_policy_versions
from prime_rl.transport.types import TrainingSample


@pytest.fixture(scope="module", autouse=True)
def actual_cpu_collectives(tmp_path_factory):
    if expected_root := os.environ.get("PRIME_RL_TEST_SOURCE_ROOT"):
        source = Path(inspect.getfile(inspect.unwrap(accumulate_policy_provenance))).resolve()
        assert source.is_relative_to(Path(expected_root).resolve())
    rendezvous = tmp_path_factory.mktemp("provenance") / "gloo"
    dist.init_process_group("gloo", init_method=rendezvous.as_uri(), rank=0, world_size=1)
    yield
    dist.destroy_process_group()


def summarize(labels, mask, reference):
    summary = TensorSummaries("cpu")
    accumulate_policy_provenance(
        summary,
        token_policy_versions=None if labels is None else torch.tensor([labels], dtype=torch.int64),
        eligible_mask=torch.tensor([mask]),
        reference_version=reference,
    )
    return compute_policy_provenance(summary)


def test_literal_first_resume_unknown_future_and_integer_boundaries():
    assert policy_version_reference(step=1, max_concurrent_runs=1, has_lora=False) == 0
    reference = policy_version_reference(step=8, max_concurrent_runs=1, has_lora=False)
    assert reference == 7
    result = summarize([7, 5, -1, 8, 2**63 - 1, 0], [True, True, True, True, True, False], reference)
    assert result == {
        "observed_policy_version/eligible_count": 5,
        "observed_policy_version/known_count": 4,
        "observed_policy_version/known_fraction": 4 / 5,
        "observed_policy_version/unknown_count": 1,
        "observed_policy_version/unknown_fraction": 1 / 5,
        "observed_policy_version/future_count": 2,
        "observed_policy_version/future_fraction": 2 / 5,
        "sampled_weight_age/count": 2,
        "sampled_weight_age/mean": 1.0,
        "sampled_weight_age/max": 2.0,
    }
    assert summarize([0], [True], 0)["sampled_weight_age/mean"] == 0
    # Float64 classification before integer subtraction would lose these ages.
    for reference in [2**53 + 1, 2**63 - 1]:
        result = summarize([reference, reference - 1, -1], [True] * 3, reference)
        assert result["sampled_weight_age/mean"] == 0.5
        assert result["sampled_weight_age/max"] == 1
        assert result["observed_policy_version/future_count"] == 0
    for labels in [None, [-1, -1]]:
        result = summarize(labels, [True, True], 0)
        assert result["observed_policy_version/known_fraction"] == 0
        assert result["observed_policy_version/unknown_count"] == 2
        assert result["sampled_weight_age/count"] == 0
        assert not any(key.endswith(("/mean", "/max")) for key in result)
    result = summarize([0], [False], 0)
    assert result["observed_policy_version/eligible_count"] == 0
    assert not any(key.endswith("fraction") for key in result)
    for runs, lora in [(2, False), (1, True)]:
        assert policy_version_reference(step=8, max_concurrent_runs=runs, has_lora=lora) is None
        result = summarize([0, 9, -1], [True] * 3, None)
        assert result["observed_policy_version/known_count"] == 2
        assert not any("future" in key or "age" in key for key in result)


def node(parent, ids, mask, spans=None):
    return vf.MessageNode(
        parent=parent,
        message=vf.AssistantMessage(content="answer") if any(mask) else vf.UserMessage(content="question"),
        sampled=any(mask),
        token_ids=ids,
        mask=mask,
        logprobs=[-0.125] * sum(mask),
        weight_versions=None if spans is None else [WeightVersionSpan(version=v, start=a, end=b) for v, a, b in spans],
    )


def episode(nodes):
    trace = vf.WireTrace(
        task=vf.TraceTask(type="fixture", data=vf.WireTaskData(idx=1, prompt="question")),
        agent=vf.AgentInfo(config=WireAgentConfig(model="fixture", runtime={"type": "subprocess"})),
        nodes=nodes,
        calls=[vf.ModelCall(node=i, finish_reason="stop") for i, n in enumerate(nodes) if n.sampled],
        ok=True,
        is_completed=True,
    )
    return vf.WireEpisode(task=trace.task, env=vf.EnvInfo(id="fixture"), traces=[trace], ok=True)


def test_real_materializer_projection_packing_and_cpu_decoder_cohort_is_observation_only():
    shared = from_episode(
        episode(
            [
                node(None, [1], [False]),
                node(0, [2, 3], [True, True], [("7", 0, 1), ("8", 1, 2)]),
                node(1, [4], [False]),
                node(2, [5, 6], [True, True], [("9", 0, 2)]),
                node(1, [7], [False]),
                node(4, [8], [True], [("10", 0, 1)]),
            ]
        )
    )
    projection = from_episode(
        episode(
            [
                node(None, [11, 12], [False, False]),
                node(0, [13, 14, 15, 16, 17], [False, True, False, True, True], [("7", 0, 2), ("99", 2, 3)]),
            ]
        ),
        training_projection=SingleTurnTrainingProjectionConfig(max_seq_len=5),
    )
    samples = materialize_training_data(shared, capture_policy_versions=True).samples
    projected = materialize_training_data(projection, capture_policy_versions=True).samples
    samples += projected + materialize_training_data(projection, capture_policy_versions=False).samples
    for ids, version, rl, ce, ref in [([101, 102], 10, 0, 1, 0), ([103, 104], 9, 0, 0, 1), ([105, 106], 11, 1, 0, 0)]:
        samples.append(
            TrainingSample(
                ids,
                [False, True],
                [0, -0.125],
                [1, 1],
                "component",
                ref_logprobs=[0, -0.25],
                rl_weights=[0, rl],
                ce_weights=[0, ce],
                ref_kl_weights=[0, ref],
                token_policy_versions=encode_policy_versions(np.array([-1, version], dtype=np.int64)),
            )
        )
    for sample in samples:
        sample.advantages = [0.25] * len(sample.token_ids)
        sample.temperatures = [1.0] * len(sample.token_ids)
        sample.env_name = "cohort"
    packed = prepare_batch(
        samples,
        seq_len=16,
        num_train_workers=2,
        idxs=[0] * len(samples),
        num_loras=1,
        bin_cost=sum,
        pad_to_multiple_of=8,
    )
    # Exercise a genuine packer-created dummy separately without double-counting its real sample.
    dummy_groups = prepare_batch(
        projected, seq_len=16, num_train_workers=2, idxs=[0], num_loras=1, bin_cost=sum, pad_to_multiple_of=8
    )
    dummies = [micro for group in dummy_groups for micro in group if not any(micro.loss_mask)]
    assert dummies
    rows = [micro for group in packed for micro in group] + dummies
    original = msgspec.msgpack.encode(rows)
    summary = TensorSummaries("cpu")
    holder = SimpleNamespace(multi_run_manager=SimpleNamespace(max_runs=1))
    table = []
    for row in rows:
        batch = DataLoader._micro_batch_to_tensor(holder, row)
        names, ids = prepare_token_env_ids(batch["env_names"], torch.device("cpu"))
        _, mask = accumulate_token_metrics(
            TensorSummaries("cpu"),
            TensorSummaries("cpu"),
            entropy=torch.zeros_like(batch["inference_logprobs"]),
            trainer_logprobs=batch["inference_logprobs"],
            inference_logprobs=batch["inference_logprobs"],
            loss_mask=batch["loss_mask"],
            rl_weights=batch["rl_weights"],
            ref_kl_weights=batch["ref_kl_weights"],
            env_names=names,
            env_ids=ids,
        )
        labels = batch["token_policy_versions"]
        table += [
            (token, int(labels[0, i]) if labels is not None else -1)
            for i, token in enumerate(row.input_ids)
            if bool(mask[0, i])
        ]
        accumulate_policy_provenance(summary, token_policy_versions=labels, eligible_mask=mask, reference_version=10)
    assert sorted(table) == [(2, 7), (3, 8), (5, 9), (6, 9), (8, 10), (14, -1), (14, 7), (104, 9), (106, 11)]
    result = compute_policy_provenance(summary)
    assert result["observed_policy_version/eligible_count"] == 9
    assert result["observed_policy_version/known_count"] == 8
    assert result["observed_policy_version/unknown_count"] == 1
    assert result["observed_policy_version/future_count"] == 1
    assert result["sampled_weight_age/count"] == 7
    assert result["sampled_weight_age/mean"] == 11 / 7
    assert result["sampled_weight_age/max"] == 3
    assert msgspec.msgpack.encode(rows) == original


def test_observation_preserves_actual_policy_loss_full_gradients_and_adam_update():
    torch.manual_seed(61)
    model = torch.nn.Linear(3, 5, dtype=torch.float64)
    reference = copy.deepcopy(model)
    inputs = torch.arange(18, dtype=torch.float64).reshape(6, 3).sin()
    mask = torch.tensor([[False, True, True, False, True, True]])
    labels = torch.tensor([[0, 1, 2, 3, 4, 0]])
    saved = []
    for current, instrument in [(reference, False), (model, True)]:
        logits = current(inputs).unsqueeze(0)
        logprobs = logits.log_softmax(-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        loss, _ = compute_loss(
            trainer_logprobs=[logprobs[0]],
            inference_logprobs=[logprobs[0].detach() + 0.01],
            ref_logprobs=None,
            advantages=[torch.tensor([0, 0.5, -0.2, 0, -0.7, 0.3])],
            loss_mask=[mask[0]],
            rl_weights=None,
            ce_weights=None,
            ref_kl_weights=None,
            rl_loss_fn=setup_rl_loss_fn(DefaultLossConfig()),
            rl_scale=4,
            ce_scale=1,
            ref_kl_scale=1,
        )
        if instrument:
            summary = TensorSummaries("cpu")
            accumulate_policy_provenance(
                summary,
                token_policy_versions=torch.tensor([[-1, 0, 1, -1, 2**63 - 1, -1]]),
                eligible_mask=mask,
                reference_version=1,
            )
            assert all(not value.requires_grad for value in summary.summaries.values())
            assert compute_policy_provenance(summary)["sampled_weight_age/count"] == 2
        loss.backward()
        gradients = [p.grad.clone() for p in current.parameters()]
        assert all(torch.count_nonzero(g) for g in gradients)
        optimizer = torch.optim.AdamW(current.parameters(), lr=0.01)
        optimizer.step()
        saved.append(
            (
                loss.detach().clone(),
                gradients,
                [p.detach().clone() for p in current.parameters()],
                [
                    [v.clone() for v in state.values() if isinstance(v, torch.Tensor)]
                    for state in optimizer.state.values()
                ],
            )
        )
    for left, right in zip(saved[0][:1], saved[1][:1]):
        assert torch.equal(left, right)
    for group in [1, 2]:
        for left, right in zip(saved[0][group], saved[1][group], strict=True):
            assert torch.equal(left, right)
    for left, right in zip(saved[0][3], saved[1][3], strict=True):
        for a, b in zip(left, right, strict=True):
            assert torch.equal(a, b)
