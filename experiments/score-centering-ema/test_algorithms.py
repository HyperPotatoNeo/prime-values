import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from prime_rl.inference.vllm.score_centering import encode_sampler_head
from prime_rl.orchestrator.sampler_head import branch_sampler_head
from prime_rl.trainer.batch import prepare_batch
from prime_rl.trainer.models.layers.lm_head import FusedOutputLinear
from prime_rl.trainer.rl.ema import EMAReference
from prime_rl.trainer.rl.score_centering import centering_term, importance_weight, reverse_kl_under_sampler
from prime_rl.transport.types import TrainingSample


@pytest.mark.parametrize("cap", [None, 2.0, 1.0])
def test_centered_expected_gradient_matches_full_vocabulary(cap):
    z = torch.tensor([-0.7, 0.3, 1.2, -0.1], dtype=torch.float64, requires_grad=True)
    logp = z.log_softmax(-1)
    q = torch.tensor([0.41, 0.32, 0.18, 0.09], dtype=z.dtype)
    correction, _, _ = centering_term(logp, q.log(), cap, 1e-9)
    w = importance_weight(logp.detach() - q.log(), cap)
    loss = -(q * w * logp).sum() + correction
    torch.testing.assert_close(torch.autograd.grad(loss, z)[0], torch.zeros_like(z), atol=1e-9, rtol=0)


@pytest.mark.parametrize("cap", [None, 2.0])
def test_head_tail_gradient_matches_explicit_modeled_distribution(cap):
    z = torch.tensor([-0.7, 0.3, 1.2, -0.1], dtype=torch.float64, requires_grad=True)
    lp = z.log_softmax(-1)
    head = torch.tensor([0, 2])
    qh = torch.tensor([0.4, 0.35], dtype=z.dtype)
    correction, _, _ = centering_term(lp[head], qh.log(), cap, 1e-10)
    q = lp.detach().exp() * (0.25 / (1 - lp.detach()[head].exp().sum()))
    q[head] = qh
    explicit = (q * importance_weight(lp.detach() - q.log(), cap) * lp).sum()
    actual = torch.autograd.grad(correction, z, retain_graph=True)[0]
    expected = torch.autograd.grad(explicit, z)[0]
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_on_policy_centering_gradient_zero():
    z = torch.randn(11, dtype=torch.float64, requires_grad=True)
    lp = z.log_softmax(-1)
    ids = lp.topk(4).indices
    term, _, _ = centering_term(lp[ids], lp[ids].detach(), 2.0, 1e-9)
    torch.testing.assert_close(torch.autograd.grad(term, z)[0], torch.zeros_like(z))


def test_ema_kl_value_and_gradient_off_policy():
    z = torch.tensor([0.7, -0.1, -1.2], dtype=torch.float64, requires_grad=True)
    p = z.log_softmax(-1)
    q = torch.tensor([0.1, 0.6, 0.3], dtype=z.dtype)
    e = torch.tensor([0.3, 0.3, 0.4], dtype=z.dtype)
    estimated = (q * reverse_kl_under_sampler(p, e.log(), q.log())).sum()
    exact = (p.exp() * (p - e.log())).sum()
    torch.testing.assert_close(estimated, exact)
    torch.testing.assert_close(
        torch.autograd.grad(estimated, z, retain_graph=True)[0], torch.autograd.grad(exact, z)[0]
    )


@pytest.mark.parametrize("candidates", [0, 4])
@pytest.mark.parametrize("vocab", [37, 8201])
def test_chunked_head_values_and_gradients(candidates, vocab):
    torch.manual_seed(42)
    head = FusedOutputLinear(8, vocab, 3)
    x = torch.randn(1, 7, 8, requires_grad=True)
    labels = torch.randint(vocab, (1, 7))
    ids = torch.randint(vocab, (1, 7, candidates)) if candidates else None
    temp = torch.rand(1, 7) + 0.5
    out = head(x, labels, temp, ids)
    all_ids = torch.cat([labels.unsqueeze(-1), ids], -1) if ids is not None else labels.unsqueeze(-1)
    expected = (x @ head.weight.T / temp.unsqueeze(-1)).log_softmax(-1).gather(-1, all_ids)
    actual = (
        torch.cat([out["logprobs"].unsqueeze(-1), out["topk_logprobs"]], -1)
        if ids is not None
        else out["logprobs"].unsqueeze(-1)
    )
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
    g = torch.randn_like(actual)
    a = torch.autograd.grad((actual * g).sum(), (x, head.weight), retain_graph=True)
    b = torch.autograd.grad((expected * g).sum(), (x, head.weight))
    for aa, bb in zip(a, b):
        torch.testing.assert_close(aa, bb, atol=3e-6, rtol=2e-5)


def test_ema_publications_and_resume(tmp_path):
    policy = torch.nn.Linear(3, 2)
    ema = EMAReference(copy.deepcopy(policy), policy, 0.95)
    expected = {n: p.detach().bfloat16().float().clone() for n, p in policy.named_parameters()}
    for step in range(3):
        with torch.no_grad():
            for n, p in policy.named_parameters():
                p.add_(0.13)
                expected[n].mul_(0.95).add_(p.bfloat16().float(), alpha=0.05)
        ema.update(policy, step)
    for n, p in ema.model.named_parameters():
        torch.testing.assert_close(p, expected[n], atol=0, rtol=0)
        assert not p.requires_grad
    state = copy.deepcopy(ema.state_dict())
    restored = EMAReference(copy.deepcopy(policy), policy, 0.95)
    restored.load_state_dict(state)
    restored.update(policy, 3)
    ema.update(policy, 3)
    for p, q in zip(restored.model.parameters(), ema.model.parameters()):
        torch.testing.assert_close(p, q, atol=0, rtol=0)
    assert restored.updates == 4 and restored.last_publication == 3
    with pytest.raises(ValueError):
        restored.update(policy, 3)


def test_sampler_head_rank_excludes_extra_sample_and_packs():
    rows = [
        {
            9: SimpleNamespace(rank=4, logprob=-4.0),
            2: SimpleNamespace(rank=2, logprob=-1.5),
            7: SimpleNamespace(rank=1, logprob=-0.5),
        }
        for _ in range(2)
    ]
    payload = encode_sampler_head(SimpleNamespace(logprobs=rows, token_ids=[9, 7]), 2)
    node = SimpleNamespace(token_ids=[1, 9, 7], mask=[False, True, True], sampler_head=payload)
    ids, logps = branch_sampler_head(SimpleNamespace(nodes=[node]))
    np.testing.assert_array_equal(np.frombuffer(ids.data, dtype=np.int32).reshape(3, 2), [[0, 0], [7, 2], [7, 2]])
    sample = TrainingSample(
        [1, 9, 7],
        [False, True, True],
        [0.0, -4.0, -0.5],
        [1.0] * 3,
        "test",
        advantages=[0.0, 1.0, 1.0],
        sampler_topk_ids=ids,
        sampler_topk_logprobs=logps,
    )
    batches = prepare_batch([sample, sample], 8, 1, [0, 0], 1, sum, pad_to_multiple_of=8)
    mb = batches[0][0]
    assert mb.sampler_topk_ids.shape == [8, 2]
    np.testing.assert_array_equal(
        np.frombuffer(mb.sampler_topk_ids.data, dtype=np.int32).reshape(8, 2),
        [[0, 0], [7, 2], [7, 2], [0, 0], [7, 2], [7, 2], [0, 0], [0, 0]],
    )


def test_injected_model_candidate_forward():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    from prime_rl.trainer.models.layers.lm_head import inject_prime_lm_head

    cfg = Qwen3Config(
        vocab_size=37,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
    )
    model = Qwen3ForCausalLM(cfg).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    ids = torch.tensor([[[2, 6], [3, 7], [4, 8], [5, 9]]])
    dense = model(tokens).logits.log_softmax(-1)
    inject_prime_lm_head(model, chunk_size=2)
    output = model(tokens, labels=tokens, temperature=torch.ones_like(tokens), candidate_ids=ids)
    torch.testing.assert_close(output["topk_logprobs"], dense.gather(-1, ids), atol=1e-5, rtol=1e-5)
    output["topk_logprobs"].sum().backward()
    assert torch.isfinite(model.lm_head.weight.grad).all()


def test_broadcast_retention_requires_acknowledgement(tmp_path):
    from prime_rl.utils.broadcast_retention import acknowledge, clean_acknowledged

    for i in range(12):
        (tmp_path / f"step_{i}").mkdir()
    clean_acknowledged(tmp_path, 11)
    assert len(list(tmp_path.glob("step_*"))) == 12
    acknowledge(tmp_path, 4)
    clean_acknowledged(tmp_path, 11)
    assert sorted(int(p.name[5:]) for p in tmp_path.glob("step_*")) == list(range(4, 12))
    clean_acknowledged(tmp_path, 2)
    assert (tmp_path / "step_4").exists()


def test_ema_kl_extreme_policy_probability_is_finite():
    p = torch.tensor([-1000.0], requires_grad=True)
    result = reverse_kl_under_sampler(p, torch.tensor([-0.5]), torch.tensor([-1.0]))
    assert torch.isfinite(result).all()
    assert torch.isfinite(torch.autograd.grad(result.sum(), p)[0]).all()


@pytest.mark.parametrize("arm", ["base", "sc", "ema"])
def test_research_loss_mask_weight_and_independent_gradient(arm):
    from prime_rl.configs.trainer import ResearchLossConfig
    from prime_rl.trainer.rl.loss import LossInputs, research_loss_fn

    z = torch.tensor([[0.2, -0.8, 0.5], [-0.4, 0.9, 0.1]], dtype=torch.float64, requires_grad=True)
    lp = z.log_softmax(-1)
    q = torch.tensor([[0.2, 0.5, 0.3], [0.4, 0.4, 0.2]], dtype=z.dtype)
    e = torch.tensor([[0.3, 0.3, 0.4], [0.2, 0.5, 0.3]], dtype=z.dtype)
    a = torch.tensor([0.7, 0.0], dtype=z.dtype)
    weights = torch.tensor([0.5, 2.0], dtype=z.dtype)
    action = torch.tensor([1, 2])
    p_action = lp[torch.arange(2), action]
    q_action = q[torch.arange(2), action]
    e_action = e[torch.arange(2), action]
    pad = lambda x: torch.cat((x, torch.full_like(x[:1], float("nan"))))
    config = ResearchLossConfig(score_centering=arm == "sc", top_k=3, ema_kl_beta=0.01 if arm == "ema" else 0)
    result = research_loss_fn(
        LossInputs(
            trainer_logprobs=pad(p_action),
            inference_logprobs=pad(q_action.log()),
            ref_logprobs=pad(e_action.log()),
            advantages=pad(a),
            loss_mask=torch.tensor([True, True, False]),
            loss_weights=pad(weights),
            trainer_topk_logprobs=pad(lp),
            sampler_topk_logprobs=pad(q.log()),
        ),
        config,
    ).loss
    w = (p_action.detach().exp() / q_action).clamp_max(2)
    expected = -a * w * p_action
    if arm == "sc":
        # Full vocabulary score expectation; no head/tail implementation reuse.
        mass = torch.minimum(lp.detach().exp(), 2 * q)
        expected = expected + a * (mass * lp).sum(-1)
    if arm == "ema":
        expected = expected + 0.01 * (p_action.exp() / q_action) * (
            e_action / p_action.exp() - (e_action.log() - p_action) - 1
        )
    expected = (expected * weights).sum()
    # SC surrogates may differ by a constant with zero softmax gradient.
    if arm != "sc":
        torch.testing.assert_close(result, expected)
    torch.testing.assert_close(
        torch.autograd.grad(result, z, retain_graph=True)[0],
        torch.autograd.grad(expected, z)[0],
        atol=1e-10,
        rtol=1e-10,
    )


def test_rollout_retention_preserves_future_and_trajectory_bytes(tmp_path):
    import gzip

    from prime_rl.utils.rollout_retention import retire_consumed_rollouts

    run = tmp_path / "run_default"
    for root in (tmp_path / "rollouts", run / "rollouts"):
        for step in range(1, 8):
            d = root / f"step_{step}"
            d.mkdir(parents=True)
            (d / "batch.bin").write_bytes(b"tensor payload")
            (d / "train_rollouts.jsonl").write_bytes(b'{"reward":1}\n')
    retire_consumed_rollouts(tmp_path, run, 4, 2)
    for root in (tmp_path / "rollouts", run / "rollouts"):
        for step in range(1, 8):
            d = root / f"step_{step}"
            if step < 3:
                assert not (d / "batch.bin").exists()
                assert gzip.decompress((d / "train_rollouts.jsonl.gz").read_bytes()) == b'{"reward":1}\n'
            else:
                assert (d / "batch.bin").exists()
                assert (d / "train_rollouts.jsonl").exists()
    retire_consumed_rollouts(tmp_path, run, 2, 2)
    assert (run / "rollouts/step_7/batch.bin").exists()
