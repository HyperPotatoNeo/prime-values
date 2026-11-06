"""Episode-to-credit outcomes through the real environment collection and sink."""

import asyncio
import uuid

import pytest
import verifiers.v1 as vf
from verifiers.v1.configs.agent import WireAgentConfig

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.envs import TrainEnvs
from prime_rl.orchestrator.episodes import from_episode
from prime_rl.orchestrator.filters import GibberishFilter
from prime_rl.orchestrator.train_sink import TrainSink
from prime_rl.value.types import ValueEvaluationResponse


class FixedEvaluator:
    def __init__(self):
        self.version = 7
        self.requests = []

    async def evaluate(self, token_ids):
        self.requests.append([list(tokens) for tokens in token_ids])
        return ValueEvaluationResponse(values=[[0.25] * len(tokens) for tokens in token_ids], version=self.version)


class RecordingPublisher:
    def __init__(self, events):
        self.rollouts = []
        self.events = events

    def publish(self, rollout):
        self.events.append("publish")
        self.rollouts.append(rollout)


class ByteRenderer:
    """Deterministic renderer boundary that preserves the complete prompt bytes."""

    def __init__(self):
        self.prompts = []

    def render_ids(self, messages, *, add_generation_prompt):
        assert not add_generation_prompt and len(messages) == 1
        prompt = messages[0]["content"]
        self.prompts.append(prompt)
        return list(prompt.encode("utf-8"))


class Tokenizer:
    bos_token_id = None
    all_special_tokens = ("<control>",)


class ObservedGibberishFilter:
    name = "gibberish"
    enforce = True

    def __init__(self, events):
        self.events = events
        self.filter = GibberishFilter(self.name, token_id_threshold=100, logprob_threshold=-2.0, enforce=True)

    def check(self, rollout):
        self.events.append("filter")
        return self.filter.check(rollout)


def _sink(*, group_size=2, privileged=False, filter_stage=None):
    config = OrchestratorConfig(
        model={"name": "Qwen/Qwen3-4B"},
        batch_size=group_size,
        group_size=group_size,
        max_inflight_rollouts=4,
        train={
            "env": [
                {
                    "env": {"taskset": {"id": "reverse-text"}},
                    "name": "sink",
                    "algo": {"type": "grpo", "baseline": {"type": "value"}, "branch_semantics": "sequential"},
                }
            ]
        },
        value_function={
            "model": {"name": "fixture", "seq_len": 8192},
            "gamma": 1.0,
            "gae_lambda": 0.5,
            "value_target_lambda": 1.0,
            "privileged_context": "group_leave_one_out" if privileged else "task",
        },
    )
    evaluator = FixedEvaluator()
    events = []
    publisher = RecordingPublisher(events)
    renderer = ByteRenderer()
    envs = TrainEnvs(
        config.train.env,
        policy_pool=None,
        value_evaluator=evaluator,
        value_config=config.value_function,
    )
    sink = TrainSink(
        config,
        tokenizer=Tokenizer(),
        renderer=renderer,
        train_envs=envs,
        mm_token_type_ids_mapping=None,
        batch_size=config.batch_size,
        token_batch_size=None,
        pre_filters=[ObservedGibberishFilter(events)] if filter_stage == "pre" else [],
        post_filters=[ObservedGibberishFilter(events)] if filter_stage == "post" else [],
        value_publisher=publisher,
    )
    return sink, evaluator, publisher, renderer, events


def _arrival(member, *, cancelled=False):
    trace = vf.WireTrace(
        id=f"trace-{member}",
        agent=vf.AgentInfo(config=WireAgentConfig(model="fixture")),
        task=vf.TraceTask(type="fixture", data=vf.WireTaskData(idx=3, prompt="shared question")),
        nodes=[
            vf.MessageNode(message=vf.UserMessage(content="shared question"), token_ids=[1], mask=[False]),
            vf.MessageNode(
                parent=0,
                message=vf.AssistantMessage(content=f"answer-{member}"),
                token_ids=[200, 201] if member == 0 else [20, 21],
                sampled=True,
                mask=[True, True],
                logprobs=[-3.0, -3.0],
            ),
        ],
        calls=[vf.ModelCall(node=1, finish_reason="stop")],
        rewards={"reward": vf.Reward(score=1.0 if member == 0 else 0.5)},
        ok=True,
    )
    episode = vf.WireEpisode(
        id=f"episode-{member}",
        env=vf.EnvInfo(id="fixture", name="sink"),
        task=trace.task,
        ok=not cancelled,
        traces=[] if cancelled else [trace],
        errors=[vf.Error(type="Cancelled", message="dispatch cancelled")] if cancelled else [],
    )
    rollout = from_episode(episode, task_idx=3)
    rollout.env_name = "sink"
    rollout.group_id = uuid.UUID(int=1)
    rollout.policy_version = 2 + member
    return rollout


@pytest.mark.parametrize("filter_stage", [None, "pre", "post"])
def test_arrival_values_keep_versions_and_publish_before_policy_filters(filter_stage):
    async def run():
        sink, evaluator, publisher, _, events = _sink(filter_stage=filter_stage)
        first, second = _arrival(0), _arrival(1)
        assert await sink.add(first) is None
        await sink.scoring_tasks[id(first)]
        assert len(evaluator.requests) == 1 and first.value_version == 7
        assert publisher.rollouts == []

        evaluator.version = 11
        batch = await sink.add(second)
        assert len(evaluator.requests) == 2  # No refresh when the group closes.
        assert [first.value_version, second.value_version] == [7, 11]
        assert first.value_advantages[0] == pytest.approx([0.0, 0.375, 0.75])
        assert second.value_advantages[0] == pytest.approx([0.0, 0.125, 0.25])
        assert [r.policy_version for r in publisher.rollouts] == [2, 3]
        assert [r.value_version for r in publisher.rollouts] == [7, 11]
        assert publisher.rollouts[0].samples[0].targets == [0.0, 1.0, 1.0]
        assert publisher.rollouts[1].samples[0].targets == [0.0, 0.5, 0.5]
        assert events == (["publish", "publish", "filter", "filter"] if filter_stage else ["publish", "publish"])
        if filter_stage == "pre":
            assert batch is None and sink.pending_batch == [second]
            for member in range(2):
                refill = _arrival(member)
                refill.group_id = uuid.UUID(int=2)
                batch = await sink.add(refill)
            assert len(evaluator.requests) == 4 and len(publisher.rollouts) == 4
            assert events == ["publish", "publish", "filter", "filter"] * 2
        assert batch is not None
        assert len(batch.samples) == (1 if filter_stage == "post" else 2)
        assert batch.shipped_value_version_min == (11 if filter_stage else 7)
        assert second.samples[0].advantages == pytest.approx([0.0, 0.125, 0.25])
        first.nodes.clear()
        first.samples.clear()
        assert publisher.rollouts[0].samples[0].token_ids == [1, 200, 201]
        await sink.stop()

    asyncio.run(run())


@pytest.mark.parametrize("cancelled", [False, True])
def test_privileged_context_waits_for_complete_group_and_never_uses_cancelled_peers(cancelled):
    async def run():
        sink, evaluator, publisher, renderer, _ = _sink(group_size=3, privileged=True)
        arrivals = [_arrival(0), _arrival(1), _arrival(2, cancelled=cancelled)]
        for rollout in arrivals[:2]:
            assert await sink.add(rollout) is None
            await sink.scoring_tasks[id(rollout)]
        assert evaluator.requests == [] and publisher.rollouts == [] and renderer.prompts == []
        evaluator.version = 13
        batch = await sink.add(arrivals[2])
        assert not sink.pending_groups and not sink.scoring_tasks
        if cancelled:
            assert batch is None and not sink.pending_batch
            assert evaluator.requests == [] and publisher.rollouts == [] and renderer.prompts == []
            assert all(rollout.value_prefix is None for rollout in arrivals)
        else:
            assert batch is not None and len(batch.samples) == 3
            assert len(evaluator.requests) == 1 and len(evaluator.requests[0]) == 3
            assert batch.shipped_value_version_min == 13
            for member, (rollout, prompt, published) in enumerate(
                zip(arrivals, renderer.prompts, publisher.rollouts, strict=True)
            ):
                assert f"answer-{member}" not in prompt
                assert all(f"answer-{peer}" in prompt for peer in range(3) if peer != member)
                prefix = list(prompt.encode("utf-8"))
                assert evaluator.requests[0][member] == prefix + rollout.samples[0].token_ids
                assert published.samples[0].token_ids == evaluator.requests[0][member]
                assert published.samples[0].mask == [False] * (len(prefix) + 1) + [True, True]
                assert published.samples[0].targets[-2:] == [rollout.reward, rollout.reward]
                assert rollout.samples[0].token_ids[0] == 1  # Policy input contains no critic prefix.
        await sink.stop()

    asyncio.run(run())
