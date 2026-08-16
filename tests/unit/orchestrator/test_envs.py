from types import SimpleNamespace

import pytest

from prime_rl.orchestrator.envs import TrainEnvs


def test_group_value_context_rejects_only_affected_group_scored_environments():
    affected = [
        SimpleNamespace(
            name=name,
            requires_group_scoring=True,
            algorithm=SimpleNamespace(value_evaluator=object()),
        )
        for name in ("grpo-env", "echo-env")
    ]
    unaffected = SimpleNamespace(
        name="max-rl-env",
        requires_group_scoring=True,
        algorithm=SimpleNamespace(value_evaluator=None),
    )
    train_envs = TrainEnvs.__new__(TrainEnvs)
    train_envs._envs = {env.name: env for env in [*affected, unaffected]}
    group_context = SimpleNamespace(privileged_context="group_leave_one_out")

    with pytest.raises(ValueError, match="grpo-env, echo-env"):
        train_envs.validate_group_value_context(group_context)

    train_envs._envs = {unaffected.name: unaffected}
    train_envs.validate_group_value_context(group_context)
    train_envs.validate_group_value_context(SimpleNamespace(privileged_context="task"))
    train_envs.validate_group_value_context(None)
