import asyncio
from unittest.mock import MagicMock

import pytest
import verifiers.v1 as vf

from prime_rl.configs.algorithm import (
    AdaptiveTetherConfig,
    GRPOAlgoConfig,
    LinearLengthPenaltyConfig,
    LinearMixBaselineConfig,
    MaxRLAlgoConfig,
    TetherBaselineConfig,
)
from prime_rl.configs.value import ValueFunctionConfig
from prime_rl.orchestrator.algo.advantage import (
    compute_gae,
    group_advantages,
    group_baselines,
    linear_mix_advantages,
)
from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.algo.max_rl import MaxRLAlgorithm
from prime_rl.orchestrator.trajectories import trace_to_samples
from prime_rl.orchestrator.types import Rollout


def test_gae_monte_carlo_terminal_reward_ignores_context_tokens():
    advantages, returns = compute_gae(
        reward=1.0,
        values=[9.0, 0.2, 8.0, 0.6],
        mask=[False, True, False, True],
        gamma=1.0,
        gae_lambda=1.0,
        value_target_lambda=1.0,
    )

    assert advantages == pytest.approx([0.0, 0.8, 0.0, 0.4])
    assert returns == pytest.approx([0.0, 1.0, 0.0, 1.0])


def test_gae_nontrivial_gamma_and_lambda():
    advantages, returns = compute_gae(
        reward=1.0,
        values=[9.0, 0.2, 8.0, 0.4],
        mask=[False, True, False, True],
        gamma=0.9,
        gae_lambda=0.5,
        value_target_lambda=0.5,
    )

    assert advantages == pytest.approx([0.0, 0.43, 0.0, 0.6])
    assert returns == pytest.approx([0.0, 0.63, 0.0, 1.0])


def test_policy_gae_and_value_target_use_independent_lambdas():
    advantages, returns = compute_gae(
        reward=1.0,
        values=[9.0, 0.2, 8.0, 0.4],
        mask=[False, True, False, True],
        gamma=1.0,
        gae_lambda=0.0,
        value_target_lambda=1.0,
    )

    assert advantages == pytest.approx([0.0, 0.2, 0.0, 0.6])
    assert returns == pytest.approx([0.0, 1.0, 0.0, 1.0])


def test_leave_one_out_group_advantage_excludes_own_reward():
    assert group_advantages([0.0, 1.0, 1.0], "leave_one_out") == pytest.approx([-1.0, 0.5, 0.5])
    assert group_baselines([0.0, 1.0, 1.0], "leave_one_out") == pytest.approx([1.0, 0.5, 0.5])


def test_leave_one_out_requires_siblings():
    with pytest.raises(ValueError, match="group_size"):
        group_advantages([1.0], "leave_one_out")


def test_linear_mix_uses_static_unbounded_coefficient_on_actions():
    output = linear_mix_advantages(
        group_advantage=1.0,
        value_advantages=[0.0, 0.0, 0.0, -1.0],
        mask=[False, True, False, True],
        config=LinearMixBaselineConfig(rho=2.0),
    )

    assert output == pytest.approx([0.0, -1.0, 0.0, -3.0])


def _build_rollout(
    reward: float,
    *,
    sampled_lengths: list[int],
    obs_lengths: list[int] | None = None,
    env_name: str = "test",
    metrics: dict | None = None,
) -> Rollout:
    """Build a ``Rollout`` (a ``vf.Trace``) as an alternating message graph.

    ``sampled_lengths`` gives the token count of each model turn (a sampled
    ``AssistantMessage`` node); ``obs_lengths`` (one shorter, if given) gives the
    token count of the non-sampled observation node injected *after* each turn
    (tool output / user feedback). ``samples`` is built via the real
    ``trace_to_samples`` so the rollout matches what ``score_group`` sees.
    """
    obs_lengths = obs_lengths or []
    nodes: list[vf.MessageNode] = []
    parent: int | None = None
    next_token = 0

    def _take(n: int) -> list[int]:
        nonlocal next_token
        ids = list(range(next_token, next_token + n))
        next_token += n
        return ids

    # Leading user prompt (never trainable).
    prompt_ids = _take(1)
    nodes.append(
        vf.MessageNode(
            message=vf.UserMessage(content="q"),
            token_ids=prompt_ids,
            mask=[False] * len(prompt_ids),
            logprobs=[0.0] * len(prompt_ids),
            sampled=False,
            parent=parent,
        )
    )
    parent = len(nodes) - 1

    for i, n_sampled in enumerate(sampled_lengths):
        ids = _take(n_sampled)
        nodes.append(
            vf.MessageNode(
                message=vf.AssistantMessage(content="a"),
                token_ids=ids,
                mask=[True] * n_sampled,
                logprobs=[-0.1] * n_sampled,
                sampled=True,
                parent=parent,
            )
        )
        parent = len(nodes) - 1
        if i < len(obs_lengths):
            obs_ids = _take(obs_lengths[i])
            nodes.append(
                vf.MessageNode(
                    message=vf.ToolMessage(content="t", tool_call_id="x"),
                    token_ids=obs_ids,
                    mask=[False] * obs_lengths[i],
                    logprobs=[0.0] * obs_lengths[i],
                    sampled=False,
                    parent=parent,
                )
            )
            parent = len(nodes) - 1

    rollout = Rollout[vf.Task](
        task=vf.Task(idx=0, prompt=None),
        nodes=nodes,
        rewards={"reward": reward},
        metrics=metrics or {},
    )
    rollout.env_name = env_name
    rollout.samples = trace_to_samples(rollout, env_name=env_name)
    return rollout


def _build_branched_rollout(reward: float) -> Rollout:
    """Build two leaf branches that share one sampled assistant prefix."""
    nodes = [
        vf.MessageNode(
            message=vf.UserMessage(content="q"),
            token_ids=[0],
            mask=[False],
            logprobs=[0.0],
            sampled=False,
            parent=None,
        ),
        vf.MessageNode(
            message=vf.AssistantMessage(content="shared"),
            token_ids=[1],
            mask=[True],
            logprobs=[-0.1],
            sampled=True,
            parent=0,
        ),
        vf.MessageNode(
            message=vf.AssistantMessage(content="left"),
            token_ids=[2],
            mask=[True],
            logprobs=[-0.1],
            sampled=True,
            parent=1,
        ),
        vf.MessageNode(
            message=vf.AssistantMessage(content="right"),
            token_ids=[3],
            mask=[True],
            logprobs=[-0.1],
            sampled=True,
            parent=1,
        ),
    ]
    rollout = Rollout[vf.Task](
        task=vf.Task(idx=0, prompt=None),
        nodes=nodes,
        rewards={"reward": reward},
        metrics={},
    )
    rollout.env_name = "test"
    rollout.samples = trace_to_samples(rollout, env_name="test")
    return rollout


def _make_rollout(
    reward: float,
    completion_len: int = 1,
    num_turns: int = 1,
    env_name: str = "test",
    metrics: dict | None = None,
) -> Rollout:
    """Build a ``Rollout`` carrying ``completion_len`` model-sampled tokens split
    across ``num_turns`` sampled turns. Always carries at least one trainable
    token so credit broadcasts somewhere."""
    num_turns = max(num_turns, 1)
    per_turn, rem = divmod(max(completion_len, 1), num_turns)
    sampled_lengths = [per_turn + (rem if i == 0 else 0) for i in range(num_turns)]
    sampled_lengths = [max(n, 1) for n in sampled_lengths]
    return _build_rollout(reward, sampled_lengths=sampled_lengths, env_name=env_name, metrics=metrics)


def _make_group(rewards, completion_lengths=None, num_turns=None) -> list[Rollout]:
    """Build one group of ``Rollout``\\ s from 1D arrays of rewards/lengths/turns —
    exactly what ``score_group`` sees."""
    rollouts = []
    for i, reward in enumerate(rewards):
        cl = int(completion_lengths[i]) if completion_lengths is not None else 1
        nt = int(num_turns[i]) if num_turns is not None else 1
        rollouts.append(_make_rollout(float(reward), cl, nt))
    return rollouts


def _scalar(rollout: Rollout) -> float:
    """The per-rollout advantage scalar an algorithm assigned — broadcast over
    the rollout's trainable (mask-True) tokens, so any trainable position holds it."""
    mask = [m for sample in rollout.samples for m in sample.mask]
    return rollout.advantages[mask.index(True)]


def _grpo(group: list[Rollout], length_penalty=None) -> list[float]:
    """Drive ``GRPOAlgorithm.score_group`` and read back each per-rollout scalar."""
    algo = GRPOAlgorithm(GRPOAlgoConfig(length_penalty=length_penalty), policy_pool=None)
    asyncio.run(algo.score_group(group))
    return [_scalar(rollout) for rollout in group]


def _max_rl(group: list[Rollout]) -> list[float]:
    """Drive ``MaxRLAlgorithm.score_group`` and read back each per-rollout scalar."""
    algo = MaxRLAlgorithm(MaxRLAlgoConfig(), policy_pool=None)
    asyncio.run(algo.score_group(group))
    return [_scalar(rollout) for rollout in group]


# --------------------------------------------------------------------------
# GRPO / MaxRL: group-relative credit, assigned in score_group.
# --------------------------------------------------------------------------


def test_grpo_plain_mean():
    advs = _grpo(_make_group(rewards=[1.0, 0.5, 0.8], completion_lengths=[10, 12, 8]))
    assert len(advs) == 3
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)


def test_grpo_singleton_group_is_zero():
    # A group of size 1 has reward == mean, so its advantage is 0.
    assert _grpo([_build_rollout(0.7, sampled_lengths=[2])]) == pytest.approx([0.0], abs=1e-6)


def test_max_rl_mean_normalized():
    # mean 0.25: the success gets (1 - 0.25)/0.25 = 3, failures (0 - 0.25)/0.25 = -1
    assert _max_rl(_make_group(rewards=[1.0, 0.0, 0.0, 0.0])) == pytest.approx([3.0, -1.0, -1.0, -1.0])
    # no-success groups carry no signal (the paper's K=0 convention) ...
    assert _max_rl(_make_group(rewards=[0.0, 0.0])) == pytest.approx([0.0, 0.0])
    # ... and all-success groups center to zero like GRPO
    assert _max_rl(_make_group(rewards=[1.0, 1.0])) == pytest.approx([0.0, 0.0])


# --------------------------------------------------------------------------
# GRPO linear length penalty: pass_rate-scaled penalty before the baseline.
# --------------------------------------------------------------------------


def test_linear_equal_lengths_reduce_to_plain_grpo():
    """Equal completion length and turns → every rollout takes the same penalty
    fraction, so subtracting it leaves the centered advantages unchanged."""
    penalized = _grpo(
        _make_group(rewards=[1.0, 0.0, 1.0], completion_lengths=[10, 10, 10], num_turns=[2, 2, 2]),
        length_penalty=LinearLengthPenaltyConfig(),
    )
    plain = _grpo(_make_group(rewards=[1.0, 0.0, 1.0], completion_lengths=[10, 10, 10], num_turns=[2, 2, 2]))
    assert penalized == pytest.approx(plain, abs=1e-6)


def test_linear_completion_term_penalizes_longer():
    """With only the completion term, longer completions get a larger penalty and a
    lower advantage; advantages stay zero-mean."""
    cfg = LinearLengthPenaltyConfig(num_output_tokens_weight=0.25, num_input_tokens_weight=0.0, num_turns_weight=0.0)
    advs = _grpo(_make_group(rewards=[1.0, 1.0, 1.0], completion_lengths=[10, 20, 30]), length_penalty=cfg)
    assert advs[0] > advs[1] > advs[2]
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)


def test_linear_context_term_penalizes_more_context():
    """The context term penalizes non-completion (prompt / tool-response) tokens: at
    equal completion length, more context tokens yields a lower advantage."""
    cfg = LinearLengthPenaltyConfig(num_output_tokens_weight=0.0, num_input_tokens_weight=0.25, num_turns_weight=0.0)
    group = [
        _build_rollout(1.0, sampled_lengths=[10], obs_lengths=[]),
        _build_rollout(1.0, sampled_lengths=[10], obs_lengths=[100]),
    ]
    asyncio.run(GRPOAlgorithm(GRPOAlgoConfig(length_penalty=cfg), policy_pool=None).score_group(group))
    advs = [_scalar(rollout) for rollout in group]
    assert advs[0] > advs[1]
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)


def test_linear_turns_term_penalizes_more_turns():
    """The turns term penalizes higher turn counts at equal token lengths."""
    cfg = LinearLengthPenaltyConfig(num_output_tokens_weight=0.0, num_input_tokens_weight=0.0, num_turns_weight=0.25)
    advs = _grpo(
        _make_group(rewards=[1.0, 1.0], completion_lengths=[100, 100], num_turns=[1, 4]),
        length_penalty=cfg,
    )
    assert advs[0] > advs[1]
    assert sum(advs) == pytest.approx(0.0, abs=1e-6)


def test_adaptive_tether_applies_each_fit_only_to_later_groups():
    baseline = TetherBaselineConfig(adaptive=AdaptiveTetherConfig(batch_size=2, ridge=1e-6, ema_decay=0.0))
    algo = GRPOAlgorithm(
        GRPOAlgoConfig(baseline=baseline),
        policy_pool=None,
        value_evaluator=MagicMock(),
        value_config=ValueFunctionConfig(model={"seq_len": 8}, batch_size=2),
    )

    def group_with_perfect_start_values() -> list[Rollout]:
        group = _make_group([1.0, 0.0])
        for rollout, start_value in zip(group, [1.0, 0.0], strict=True):
            rollout.value_predictions = [
                [start_value if trainable else 0.0 for trainable in sample.mask] for sample in rollout.samples
            ]
        return group

    first = group_with_perfect_start_values()
    asyncio.run(algo.score_group(first))
    # The first group is scored by the zero/zero snapshot (pure LOO), even
    # though its moments produce alpha ~= 1 immediately afterward.
    assert [_scalar(rollout) for rollout in first] == pytest.approx([1.0, -1.0])
    assert algo.adaptive_tether is not None
    assert algo.adaptive_tether.alpha == pytest.approx(1.0, rel=2e-6)

    second = group_with_perfect_start_values()
    asyncio.run(algo.score_group(second))
    assert [_scalar(rollout) for rollout in second] == pytest.approx([0.0, 0.0], abs=2e-6)


@pytest.mark.parametrize(
    "baseline",
    [
        TetherBaselineConfig(alpha=1.0, rho=0.0),
        TetherBaselineConfig(
            adaptive=AdaptiveTetherConfig(
                batch_size=4,
                initial_alpha=1.0,
                initial_rho=0.0,
            )
        ),
    ],
)
def test_static_and_adaptive_initial_tether_coefficients_score_end_to_end(baseline):
    algo = GRPOAlgorithm(
        GRPOAlgoConfig(baseline=baseline),
        policy_pool=None,
        value_evaluator=MagicMock(),
        value_config=ValueFunctionConfig(model={"seq_len": 8}, batch_size=4),
    )
    group = _make_group([1.0, 0.0])
    for rollout, start_value in zip(group, [1.0, 0.0], strict=True):
        rollout.value_predictions = [
            [start_value if trainable else 0.0 for trainable in sample.mask] for sample in rollout.samples
        ]

    asyncio.run(algo.score_group(group))

    assert [_scalar(rollout) for rollout in group] == pytest.approx([0.0, 0.0])


@pytest.mark.parametrize(
    ("position", "shared_advantage"),
    [
        (None, 0.3),
        ({"bin_size": 1, "max_action_tokens": 2}, 0.5),
    ],
)
def test_tether_shared_prefix_keeps_native_start_value_and_action_depth(position, shared_advantage):
    baseline = TetherBaselineConfig(alpha=1.0, rho=0.0, position=position)
    algo = GRPOAlgorithm(
        GRPOAlgoConfig(baseline=baseline),
        policy_pool=None,
        value_evaluator=MagicMock(),
        value_config=ValueFunctionConfig(model={"seq_len": 8}, batch_size=2),
    )
    branched = _build_branched_rollout(1.0)
    sibling = _make_rollout(0.5)
    branched.value_predictions = [[0.0, 0.7, 0.8], [0.0, 0.7, 0.9]]
    sibling.value_predictions = [[0.0, 0.5]]

    asyncio.run(algo.score_group([branched, sibling]))

    assert [sample.mask for sample in branched.samples] == [[False, True, True], [False, False, True]]
    assert branched.advantages == pytest.approx([0.0, shared_advantage, 0.3, 0.0, 0.0, 0.3])


def test_adaptive_tether_shared_prefix_counts_each_gradient_row_once():
    baseline = TetherBaselineConfig(
        position={"bin_size": 1, "max_action_tokens": 2},
        adaptive=AdaptiveTetherConfig(batch_size=4, min_bin_rollouts=1),
    )
    algo = GRPOAlgorithm(
        GRPOAlgoConfig(baseline=baseline),
        policy_pool=None,
        value_evaluator=MagicMock(),
        value_config=ValueFunctionConfig(model={"seq_len": 8}, batch_size=4),
    )
    branched = _build_branched_rollout(1.0)
    sibling = _make_rollout(0.5)
    branched.value_predictions = [[0.0, 0.7, 0.8], [0.0, 0.7, 0.9]]
    sibling.value_predictions = [[0.0, 0.5]]

    asyncio.run(algo.score_group([branched, sibling]))

    assert algo.adaptive_tether is not None
    assert [stats.weight for stats in algo.adaptive_tether.pending_bins] == [2, 2]
    assert algo.adaptive_tether.pending_contributors == [2, 1]
    child_stats = algo.adaptive_tether.pending_bins[1]
    assert child_stats.alpha_alpha == pytest.approx(0.08)
    assert child_stats.alpha_rho == pytest.approx(0.06)
    assert child_stats.rho_rho == pytest.approx(0.05)
    assert child_stats.alpha_target == pytest.approx(0.2)
    assert child_stats.rho_target == pytest.approx(0.15)


def test_tether_rejects_permuted_branch_samples():
    baseline = TetherBaselineConfig(alpha=1.0, rho=0.0)
    algo = GRPOAlgorithm(
        GRPOAlgoConfig(baseline=baseline),
        policy_pool=None,
        value_evaluator=MagicMock(),
        value_config=ValueFunctionConfig(model={"seq_len": 8}, batch_size=2),
    )
    branched = _build_branched_rollout(1.0)
    branched.samples.reverse()
    branched.value_predictions = [[0.0, 0.7, 0.9], [0.0, 0.7, 0.8]]
    sibling = _make_rollout(0.5)
    sibling.value_predictions = [[0.0, 0.5]]

    with pytest.raises(ValueError, match="token streams are misaligned"):
        asyncio.run(algo.score_group([branched, sibling]))


def test_adaptive_tether_excludes_tokens_beyond_actor_context_from_regression():
    baseline = TetherBaselineConfig(adaptive=AdaptiveTetherConfig(batch_size=4))
    algo = GRPOAlgorithm(
        GRPOAlgoConfig(baseline=baseline),
        policy_pool=None,
        value_evaluator=MagicMock(),
        value_config=ValueFunctionConfig(model={"seq_len": 8}, batch_size=4),
        policy_seq_len=2,
    )
    group = _make_group([1.0, 0.0], completion_lengths=[2, 2])
    for rollout, start_value in zip(group, [1.0, 0.0], strict=True):
        # prompt + two actions, but the actor-valid prefix ends after the
        # first action. The final value must not become a regression row.
        rollout.value_predictions = [[0.0, start_value, 100.0]]

    asyncio.run(algo.score_group(group))

    assert algo.adaptive_tether is not None
    assert algo.adaptive_tether.pending_rollouts == 2
    assert algo.adaptive_tether.pending_bins[0].weight == 2
    assert [rollout.advantages[-1] for rollout in group] == [0.0, 0.0]


# --------------------------------------------------------------------------
# assign_advantages: scalar broadcast over the rollout's trainable tokens.
# --------------------------------------------------------------------------


def test_assign_advantages_broadcasts_scalar():
    """A scalar broadcasts uniformly over the rollout's trainable (mask-True) tokens."""
    rollout = _build_rollout(0.0, sampled_lengths=[2])
    # one user prompt token (masked) + 2 sampled tokens (trainable)
    rollout.assign_advantages(0.7)
    assert rollout.advantages == [0.0, 0.7, 0.7]


def test_assign_advantages_zeros_non_trainable():
    """Non-trainable (mask=False) positions stay 0.0 under scalar broadcast."""
    # prompt(1, masked) + sampled(1) + obs(1, masked): mask is [F, T, F]
    rollout = _build_rollout(0.0, sampled_lengths=[1], obs_lengths=[1])
    rollout.assign_advantages(0.7)
    assert rollout.advantages == [0.0, 0.7, 0.0]


def test_assign_advantages_rejects_misaligned():
    rollout = _build_rollout(0.0, sampled_lengths=[2])
    # full length is 3 (prompt + 2 sampled); a 1-element list must be rejected
    with pytest.raises(ValueError, match="align"):
        rollout.assign_advantages([0.5])
