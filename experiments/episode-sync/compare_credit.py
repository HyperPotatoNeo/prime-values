"""Compare actual finalized value/TETHER credit and gradients in isolated processes.

Each source root must contain src/, packages/prime-rl-configs/, and its pinned
deps/{verifiers,renderers,pydantic-config}. A git archive of the frozen baseline
plus archives of its gitlinks is sufficient. The CPU Python environment supplies
third-party packages; production imports are checked against the requested root.
"""

import argparse
import asyncio
import hashlib
import importlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

CASES = [
    dict(name="value_monte_carlo", shape="linear", gamma=1.0, gae=1.0, target=1.0),
    dict(name="value_independent_lambdas", shape="turns", gamma=0.9, gae=0.35, target=0.8),
    dict(name="value_shared_prefix", shape="branches", gamma=0.9, gae=0.35, target=0.8),
    dict(name="value_privileged_prefix", shape="turns", gamma=0.95, gae=0.6, target=0.2, prefix=True),
    dict(name="value_categorical", shape="turns", gamma=0.9, gae=0.35, target=0.8, categorical=True),
    dict(name="tether_static_branches", shape="branches", gamma=0.9, gae=0.6, target=0.3, tether="static"),
    dict(name="tether_global_resume", shape="turns", gamma=0.9, gae=0.7, target=0.4, tether="adaptive"),
    dict(name="tether_position_resume", shape="branches", gamma=0.9, gae=0.7, target=0.4, tether="position"),
]


def fixture_rollout(case, member, round_index):
    """Only the input schema differs across the frozen and Episode boundaries."""
    import verifiers.v1 as vf

    from prime_rl.orchestrator.types import Rollout

    offset = member + round_index

    def node(parent, tokens, sampled):
        return vf.MessageNode(
            parent=parent,
            message=vf.AssistantMessage(content="answer") if sampled else vf.UserMessage(content="context"),
            sampled=sampled,
            token_ids=[token + offset for token in tokens],
            mask=[sampled] * len(tokens),
            logprobs=[-1.7 - 0.03 * token for token in tokens] if sampled else [],
        )

    nodes = [node(None, [0, 1], False), node(0, [2, 3], True)]
    if case["shape"] != "linear":
        nodes.extend([node(1, [4], False), node(2, [5, 6], True)])
    if case["shape"] == "branches":
        # Leaf order differs from generation order; both leaves share actions.
        nodes.extend([node(1, [7], False), node(4, [8], True), node(3, [9], False), node(6, [10, 11], True)])
    reward = [0.9, 0.15][(member + round_index) % 2]
    idx = round_index * 2 + member
    if hasattr(vf, "TaskData"):
        from prime_rl.orchestrator.episodes import from_episode

        trace = vf.WireTrace(
            id=f"trace-{idx}",
            task=vf.TraceTask(type="credit_fixture", data=vf.WireTaskData(idx=idx, prompt="fixture")),
            nodes=nodes,
            rewards={"reward": reward},
            ok=True,
        )
        episode = vf.WireEpisode(id=f"episode-{idx}", env="parity", traces=[trace], ok=True)
        rollout = from_episode(episode, task_idx=idx)
    else:
        rollout = Rollout(task=vf.Task(idx=idx, prompt="fixture"), nodes=nodes, rewards={"reward": reward})
    rollout.env_name = "parity"
    rollout.group_id = uuid.UUID(int=round_index + 1)
    rollout.policy_version = 3
    return rollout


class FixedEvaluator:
    """Deterministic service boundary; production code owns all credit math."""

    def __init__(self):
        self.requests = []
        self.version = 7

    async def evaluate(self, token_ids):
        self.requests.append([list(tokens) for tokens in token_ids])
        values = [
            [0.0] + [0.03 * (token % 11) + 0.01 * index for index, token in enumerate(tokens[1:], 1)]
            for tokens in token_ids
        ]
        return SimpleNamespace(values=values, version=self.version)


def loss_and_gradients(rollout, value_config):
    import torch

    from prime_rl.configs.trainer import DefaultLossConfig
    from prime_rl.trainer.rl.loss import compute_loss, setup_rl_loss_fn
    from prime_rl.value.math import align_value_logits, compute_value_loss, value_head_output_size

    samples = rollout.samples
    actor_logits = [
        (torch.arange(len(sample.token_ids) * 19, dtype=torch.float64).reshape(-1, 19).sin() * 0.4).requires_grad_()
        for sample in samples
    ]
    logprobs = [
        logits.log_softmax(-1).gather(-1, torch.tensor(sample.token_ids).remainder(19).unsqueeze(-1)).squeeze(-1)
        for logits, sample in zip(actor_logits, samples, strict=True)
    ]
    masks = [torch.tensor(sample.mask) for sample in samples]
    actor_loss, _ = compute_loss(
        trainer_logprobs=logprobs,
        inference_logprobs=[torch.tensor(sample.logprobs, dtype=torch.float64) for sample in samples],
        ref_logprobs=None,
        advantages=[torch.tensor(sample.advantages, dtype=torch.float64) for sample in samples],
        loss_mask=masks,
        rl_weights=None,
        ce_weights=None,
        ref_kl_weights=None,
        rl_loss_fn=setup_rl_loss_fn(DefaultLossConfig()),
        rl_scale=max(sum(int(mask.sum()) for mask in masks), 1),
        ce_scale=1,
        ref_kl_scale=1,
    )
    actor_loss.backward()
    # Include actual prefix lifting and packed state-to-action alignment before
    # the production critic loss, so context and segment boundaries matter.
    lengths, targets, critic_masks = [], [], []
    for sample, returns in zip(samples, rollout.value_returns, strict=True):
        if rollout.value_prefix is None:
            length = min(len(sample.token_ids), value_config.model.seq_len)
            targets.extend(returns[:length])
            critic_masks.extend(sample.mask[:length])
        else:
            length = len(rollout.value_prefix.apply(list(sample.token_ids)))
            targets.extend(rollout.value_prefix.lift(returns, fill=0.0))
            critic_masks.extend(rollout.value_prefix.lift(list(sample.mask), fill=False))
        lengths.append(length)
    outputs = value_head_output_size(value_config.loss)
    critic_logits = (
        torch.arange(sum(lengths) * outputs, dtype=torch.float64).reshape(1, -1, outputs).cos() * 0.3
    ).requires_grad_()
    critic_loss, _ = compute_value_loss(
        align_value_logits(critic_logits, lengths),
        torch.tensor([targets], dtype=torch.float32),
        torch.tensor([critic_masks]),
        value_config.loss,
        scale=max(sum(critic_masks), 1),
    )
    critic_loss.backward()
    return {
        "actor_loss": actor_loss.item(),
        "actor_full_logit_gradients": [tensor.grad.tolist() for tensor in actor_logits],
        "critic_loss": critic_loss.item(),
        "critic_full_logit_gradients": critic_logits.grad.tolist(),
    }


async def run_case(case):
    from prime_rl.configs.algorithm import GRPOAlgoConfig
    from prime_rl.configs.value import ValueFunctionConfig
    from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
    from prime_rl.orchestrator.trajectories import materialize_training_data
    from prime_rl.orchestrator.value_context import TokenPrefix

    baseline = {"type": "value"}
    tether = case.get("tether")
    if tether == "static":
        baseline = {"type": "tether", "adaptive": None, "rho": 0.3}
    elif tether:
        adaptive = {"batch_size": 3, "ridge": 0.05, "ema_decay": 0.6}
        if tether == "position":
            adaptive["position"] = {"bin_size": 2, "max_action_tokens": 8, "min_bin_rollouts": 1}
        baseline = {"type": "tether", "adaptive": adaptive}
    config = ValueFunctionConfig(
        model={"name": "fixture", "seq_len": 32},
        gamma=case["gamma"],
        gae_lambda=case["gae"],
        value_target_lambda=case["target"],
        warmup_updates=2,
        loss={"type": "classification", "num_bins": 9, "reward_range": [-4.0, 4.0]}
        if case.get("categorical")
        else {"type": "mse"},
    )
    evaluator = FixedEvaluator()

    def new_algorithm():
        return GRPOAlgorithm(
            GRPOAlgoConfig(baseline=baseline, branch_semantics="sequential"),
            SimpleNamespace(),
            value_evaluator=evaluator,
            value_config=config,
            policy_seq_len=32,
        )

    algorithm = new_algorithm()
    result = {"rounds": []}
    for round_index in range(4 if tether in {"adaptive", "position"} else 1):
        if round_index == 2:
            state = json.loads(json.dumps(algorithm.state_dict(), allow_nan=False))
            assert state["tether"]["updates"] == 1 and state["tether"]["pending_rollouts"] == 1, state
            result["restored_partial_state"] = state
            algorithm = new_algorithm()
            algorithm.load_state_dict(state)
            assert algorithm.state_dict() == state
        evaluator.version = 7 + round_index
        group = [fixture_rollout(case, member, round_index) for member in range(2)]
        for rollout in group:
            materialization = materialize_training_data(rollout, env_name=rollout.env_name)
            rollout.samples = materialization.samples
            rollout.training_layout = materialization.layout
            if case.get("prefix"):
                rollout.value_prefix = TokenPrefix((15, 16), insert_at=1)
            await algorithm.finalize_rollout(rollout)
        await algorithm.finalize_group(group)
        result["rounds"].append(
            {
                "rollouts": [
                    {
                        "samples": [
                            {
                                "token_ids": list(s.token_ids),
                                "mask": list(s.mask),
                                "logprobs": list(s.logprobs),
                                "advantages": list(s.advantages),
                            }
                            for s in r.samples
                        ],
                        "values": r.value_predictions,
                        "value_advantages": r.value_advantages,
                        "targets": r.value_returns,
                        "value_version": r.value_version,
                        **loss_and_gradients(r, config),
                    }
                    for r in group
                ],
                "algorithm_state": algorithm.state_dict(),
            }
        )
    result["evaluator_requests"] = evaluator.requests
    return result


def worker(root, output):
    import renderers
    import torch
    import verifiers.v1 as vf

    modules = [
        "prime_rl.orchestrator.algo.base",
        "prime_rl.orchestrator.algo.grpo",
        "prime_rl.orchestrator.algo.tether",
        "prime_rl.orchestrator.trajectories",
        "prime_rl.trainer.rl.loss",
        "prime_rl.value.math",
        "prime_rl.configs.value",
    ]
    # These modules are new in the ownership seam and absent in the frozen tree.
    for name in ("prime_rl.value.credit", "prime_rl.value.trajectory", "prime_rl.orchestrator.episodes"):
        if (root / "src" / (name.replace(".", "/") + ".py")).exists():
            modules.append(name)
    paths = [Path(importlib.import_module(name).__file__).resolve() for name in modules]
    paths.extend([Path(vf.__file__).resolve(), Path(renderers.__file__).resolve()])
    assert all(path.is_relative_to(root) for path in paths), paths
    metadata = {
        "source_root": str(root),
        "torch": torch.__version__,
        "python": sys.version,
        "import_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }
    results = {case["name"]: asyncio.run(run_case(case)) for case in CASES}
    output.write_text(json.dumps({"metadata": metadata, "cases": results}, indent=2, allow_nan=False) + "\n")


def first_difference(left, right, path="cases"):
    if type(left) is not type(right):
        return path + ": different types"
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return path + ": different keys"
        for key in left:
            if difference := first_difference(left[key], right[key], f"{path}.{key}"):
                return difference
    elif isinstance(left, list):
        if len(left) != len(right):
            return path + ": different lengths"
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            if difference := first_difference(a, b, f"{path}[{index}]"):
                return difference
    elif left != right:
        return f"{path}: {left!r} != {right!r}"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--worker-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.worker_root:
        worker(args.worker_root.resolve(), args.output)
        return
    if args.baseline_root is None or args.candidate_root is None:
        parser.error("both --baseline-root and --candidate-root are required")
    args.output.mkdir(parents=True, exist_ok=False)
    for label, source in (("baseline", args.baseline_root), ("candidate", args.candidate_root)):
        root = source.resolve()
        import_paths = [
            root / "src",
            root / "packages/prime-rl-configs/src",
            root / "deps/verifiers",
            root / "deps/renderers",
            root / "deps/pydantic-config/src",
        ]
        environment = {**os.environ, "PYTHONPATH": os.pathsep.join(map(str, import_paths))}
        with (args.output / f"{label}.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-root",
                    str(root),
                    "--output",
                    str((args.output / f"{label}.json").resolve()),
                ],
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=120,
            )
    baseline = json.loads((args.output / "baseline.json").read_text())
    candidate = json.loads((args.output / "candidate.json").read_text())
    difference = first_difference(baseline["cases"], candidate["cases"])
    summary = {
        "cases": len(CASES),
        "exact_match": difference is None,
        "first_difference": difference,
        "comparison": "finalized credit, requests, controller state, actor/critic loss and full logit gradients",
    }
    (args.output / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    assert difference is None, difference
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
