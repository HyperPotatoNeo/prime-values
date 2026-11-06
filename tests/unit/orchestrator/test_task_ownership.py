"""Selection and ownership contracts; server execution is covered by producer smoke."""

import asyncio
import json
import uuid

import pytest
import verifiers.v1 as vf
from pydantic import BaseModel, Field

from prime_rl.configs.algorithm import GRPOAlgoConfig
from prime_rl.configs.orchestrator import TrainEnvConfig
from prime_rl.orchestrator.dispatcher import RolloutDispatcher
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import Policy


class FixtureData(vf.TaskData):
    value_function_prompt: str = "private solution"
    metadata: dict = Field(default_factory=dict)


class FixtureTask(vf.Task[FixtureData]):
    pass


class ExcludedData(vf.TaskData):
    answer: str = Field("secret", exclude=True)


class ExcludedTask(vf.Task[ExcludedData]):
    pass


class NestedSecret(BaseModel):
    answer: str = Field("default", exclude=True)


class NestedData(vf.TaskData):
    secret: NestedSecret = Field(default_factory=NestedSecret)


class NestedTask(vf.Task[NestedData]):
    pass


def _envs(*, legacy=False):
    # Use a real native config and replace only its loaded dataset boundary.
    # Neither the dataset nor its execution runtime is needed for selection.
    configs = [
        TrainEnvConfig(
            name="native",
            env={"taskset": {"id": "harbor"}},
            group_size=3,
            ratio=1.0,
            algo=GRPOAlgoConfig(),
        )
    ]
    if legacy:
        configs.append(TrainEnvConfig(name="legacy", id="reverse-text", group_size=2, ratio=2.0, algo=GRPOAlgoConfig()))
    envs = TrainEnvs(configs, policy_pool=None)
    envs.get("native")._task_class = FixtureTask
    if legacy:
        envs.get("legacy").num_tasks = 3
        envs.get("legacy").requires_group_scoring = True
    return envs


def _task(index, position=0):
    return FixtureTask(
        FixtureData(
            idx=index,
            prompt=[vf.UserMessage(content=f"question-{position}")],
            metadata={"position": position, "nested": ["original"]},
        ),
        vf.TaskConfig(),
    )


def _dispatcher(envs, source):
    return RolloutDispatcher(
        train_envs=envs,
        eval_envs=None,
        train_source=source,
        eval_source=None,
        policy_pool=None,
        policy=Policy(version=8, model_name="fixture"),
        max_inflight_rollouts=2,
        tasks_per_minute=None,
        max_off_policy_steps=1,
    )


def test_finite_native_selection_preserves_frozen_seeded_position_sequence():
    envs = _envs(legacy=True)
    native = envs.get("native")
    semantic_ids = [31, 7, 88, 2, 19]
    tasks = [_task(index, position) for position, index in enumerate(semantic_ids)]
    native.tasks = iter(tasks)
    native.num_tasks = len(tasks)
    source = TrainSource(envs, seed=19)
    # Recorded from the index-based TrainSource at 4b8ca528. Native
    # numbers are dataset positions, deliberately distinct from semantic idx.
    expected = [
        None,
        ("native", 1),
        ("native", 3),
        ("native", 2),
        ("native", 4),
        ("native", 0),
        ("native", 4),
        None,
        ("native", 0),
        ("native", 3),
        ("native", 2),
        ("legacy", 0),
        None,
        None,
        ("legacy", 2),
        ("legacy", 1),
        ("native", 1),
        ("legacy", 0),
        ("legacy", 1),
        None,
        ("legacy", 2),
        None,
        ("legacy", 2),
        ("legacy", 0),
        None,
        ("native", 1),
        ("native", 3),
        ("legacy", 1),
        None,
        ("legacy", 2),
        ("legacy", 1),
        None,
        ("legacy", 0),
        None,
        ("legacy", 1),
        ("legacy", 2),
    ]
    actual = []
    for permits in [0, 1, 2, 4, 1, 4, 2, 0, 4, 1, 2, 4] * 3:
        example = source.next_example(permits)
        if example is None:
            actual.append(None)
        elif example["env_name"] == "native":
            task = example["task"]
            position = task.data.metadata["position"]
            assert example["task_idx"] == semantic_ids[position]
            assert task is tasks[position]
            actual.append(("native", position))
        else:
            assert "task" not in example
            actual.append(("legacy", example["task_idx"]))
    assert actual == expected


def test_group_owns_one_complete_payload_before_source_nested_data_changes():
    async def run():
        second_started = asyncio.Event()
        hold_second = asyncio.Event()
        requests = []

        class ClientPool:
            async def select_train_client(self, load):
                return vf.EvalClientConfig()

        class EpisodeClient:
            async def run(self, *, task_idx, task_data, client, model, sampling):
                assert task_idx is None  # Native addressing never sends both.
                requests.append(json.loads(json.dumps(task_data)))
                if len(requests) == 2:
                    second_started.set()
                    await hold_second.wait()
                return vf.WireEpisode(
                    ok=True,
                    traces=[
                        vf.WireTrace(
                            ok=True,
                            task=vf.TraceTask(type="fixture", data=vf.WireTaskData.model_validate(task_data)),
                            nodes=[
                                vf.MessageNode(
                                    sampled=True,
                                    message=vf.AssistantMessage(content="answer"),
                                    token_ids=[2],
                                    mask=[True],
                                    logprobs=[-0.5],
                                )
                            ],
                            calls=[vf.ModelCall(node=0, finish_reason="stop")],
                        )
                    ],
                )

        envs = _envs()
        task = _task(31)
        native = envs.get("native")
        native.tasks, native.num_tasks = iter([task]), 1
        native.sampler.pool = ClientPool()
        native._env_client = EpisodeClient()
        dispatcher = _dispatcher(envs, TrainSource(envs, seed=1))
        group = dispatcher.next_fresh_group("train", envs)
        assert group is not None and group.task_idx == 31 and group.target_rollouts == 3
        group_id = uuid.UUID(int=1)
        dispatcher.groups[group_id] = group
        expected = task.data.model_dump(mode="json")
        assert await dispatcher.schedule_group_rollout(group_id, group)
        (first,) = dispatcher.inflight
        await first
        await dispatcher.handle_completed_rollout(first)

        task.data.prompt[0].content = "changed after first sibling"
        task.data.metadata["nested"].append("changed")
        assert await dispatcher.schedule_group_rollout(group_id, group)
        await asyncio.wait_for(second_started.wait(), timeout=2)
        assert requests == [expected, expected]
        assert group.task_data["value_function_prompt"] == "private solution"
        assert group.task_data["metadata"]["nested"] == ["original"]
        assert group.policy_version_at_start == 8

        # One completed, one in flight, one never scheduled. Cancellation must
        # settle the same group without pulling a replacement task or double emit.
        assert await dispatcher.drop_group(group_id) == 2
        arrivals = [dispatcher.out_q.get_nowait() for _ in range(3)]
        assert dispatcher.out_q.empty() and not dispatcher.groups and not dispatcher.inflight
        assert dispatcher.available_permits == dispatcher.max_inflight
        assert [rollout.has_error for rollout in arrivals] == [False, True, True]
        assert [rollout.error.type for rollout in arrivals[1:]] == ["Cancelled", "Cancelled"]
        assert all(rollout.group_id == group_id and rollout.task.data.idx == 31 for rollout in arrivals)
        assert all(rollout.policy_version == 8 for rollout in arrivals)
        assert len(requests) == 2 and dispatcher.train_source.cursors["native"] == 1

    asyncio.run(run())


def test_native_task_validation_rejects_unreconstructable_behavior_and_data():
    env = _envs().get("native")
    task = _task(4)
    assert env.validate_task(task) is task

    class SpecializedTask(FixtureTask):
        pass

    with pytest.raises(ValueError, match="worker reconstructs"):
        env.validate_task(SpecializedTask(task.data, task.config))
    different_config = vf.TaskConfig(judges=[vf.ReferenceJudgeConfig(name="changed-score")])
    with pytest.raises(ValueError, match="per-row task config"):
        env.validate_task(FixtureTask(task.data, different_config))
    with pytest.raises(ValueError, match="integer task.data.idx"):
        env.validate_task(_task(None))
    env._task_class = ExcludedTask
    with pytest.raises(ValueError, match="excluded fields"):
        env.validate_task(ExcludedTask(ExcludedData(idx=4), vf.TaskConfig()))
    env._task_class = NestedTask
    with pytest.raises(ValueError, match="changes when reconstructed"):
        env.validate_task(NestedTask(NestedData(idx=4, secret=NestedSecret(answer="actual answer")), vf.TaskConfig()))


def test_infinite_source_is_lazy_and_dispatch_gates_do_not_consume_tasks():
    async def run():
        envs = _envs()
        env = envs.get("native")
        pulled = []

        def generate():
            position = 0
            while True:
                pulled.append(position)
                yield env.validate_task(_task(100 + position, position))
                position += 1

        env.tasks, env.num_tasks = generate(), None
        source = TrainSource(envs, seed=19)
        dispatcher = _dispatcher(envs, source)
        assert pulled == []
        for _ in range(20):
            assert source.next_example(available_permits=0) is None
        dispatcher.dispatch_allowed.clear()
        await dispatcher.fill_inflight()
        dispatcher.dispatch_allowed.set()
        dispatcher.inflight_permits = dispatcher.max_inflight
        await dispatcher.fill_inflight()
        assert pulled == []
        dispatcher.inflight_permits = 0
        first = dispatcher.next_fresh_group("train", envs)
        second = dispatcher.next_fresh_group("train", envs)
        assert pulled == [0, 1]  # One pull for each three-sibling group.
        assert first.target_rollouts == second.target_rollouts == 3
        assert first.task_idx == 100 and second.task_idx == 101
        assert source.examples["native"] is None
        assert len(source.iters) == 1

    asyncio.run(run())
