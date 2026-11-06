import pytest
import verifiers.v1 as vf
from verifiers.v1.configs.agent import WireAgentConfig

from prime_rl.orchestrator.episodes import UnsupportedEpisodeError, from_episode
from prime_rl.orchestrator.records import public_rollout_record
from prime_rl.orchestrator.trajectories import materialize_training_data
from prime_rl.value.trajectory import snapshot_value_trajectory


def _trace(*, ok=True, trainable=True, errors=()):
    return vf.WireTrace(
        task=vf.TraceTask(
            type="fixture",
            data=vf.WireTaskData(idx=3, prompt="question", value_function_prompt="private answer"),
        ),
        nodes=[
            vf.MessageNode(message=vf.UserMessage(content="question"), token_ids=[1], mask=[False]),
            vf.MessageNode(
                parent=0,
                message=vf.AssistantMessage(content="answer"),
                token_ids=[2, 3],
                sampled=True,
                mask=[True, True],
                logprobs=[-0.2, -0.3],
            ),
        ],
        rewards={"score": vf.Reward(score=0.75)},
        agent=vf.AgentInfo(config=WireAgentConfig(model="fixture"), trainable=trainable),
        ok=ok,
        errors=list(errors),
        info={"episode": "environment-owned"},
    )


def _episode(*, traces=(), task_idx=3, **kwargs):
    task = traces[0].task if traces else vf.TraceTask(type="fixture", data=vf.WireTaskData(idx=task_idx))
    return vf.WireEpisode(task=task, env=vf.EnvInfo(id="fixture"), traces=list(traces), **kwargs)


def test_episode_projection_keeps_task_actions_and_owned_value_inputs():
    episode = _episode(id="episode-1", ok=True, traces=[_trace()])
    rollout = from_episode(episode, task_idx=3)
    rollout.policy_version = 8
    materialization = materialize_training_data(rollout, capture_sampled_masks=True)
    trajectory = snapshot_value_trajectory(
        materialization.samples,
        sampled_masks=materialization.sampled_masks,
        layout=materialization.layout,
        reward=rollout.reward,
        policy_version=rollout.policy_version,
    )
    assert rollout.task.data.value_function_prompt == "private answer"
    assert materialization.samples[0].logprobs == [0.0, -0.2, -0.3]
    public = public_rollout_record(rollout)
    assert "value_function_prompt" not in public["task"]["data"]
    assert public["info"]["episode"] == "environment-owned"
    assert public["info"]["prime_rl_episode"]["id"] == "episode-1"
    rollout.nodes.clear()
    materialization.samples[0].mask.clear()
    assert trajectory.samples[0].token_ids == (1, 2, 3)
    assert trajectory.samples[0].mask == (False, True, True)
    assert trajectory.reward == 0.75
    assert trajectory.policy_version == 8


@pytest.mark.parametrize("second", [dict(trainable=False), dict(ok=False), dict()])
def test_episode_rejects_original_multiple_traces_before_eligibility(second):
    episode = _episode(ok=True, traces=[_trace(), _trace(**second)])
    with pytest.raises(UnsupportedEpisodeError, match="2 original traces"):
        from_episode(episode)


def test_successful_retry_keeps_error_history_without_becoming_failure():
    error = vf.Error(type="Retryable", message="first attempt failed")
    trace = _trace(errors=[error])
    episode = _episode(ok=True, errors=[error], traces=[trace])
    rollout = from_episode(episode)
    assert not rollout.has_error
    assert rollout.errors == [error]
    assert rollout.info["prime_rl_episode"]["errors"][0]["type"] == "Retryable"
    assert "prime_rl_episode" not in trace.info


def test_failed_episode_cannot_train_its_successful_trace():
    trace = _trace()
    episode = _episode(ok=False, traces=[trace])
    rollout = from_episode(episode)
    assert rollout.has_error
    assert rollout.error.type == "EpisodeFailed"
    assert trace.ok and not trace.errors


def test_empty_episode_keeps_dispatch_identity_and_failure_provenance():
    error = vf.Error(type="SetupFailed", message="environment failed before producing a trace")
    rollout = from_episode(_episode(id="failed", errors=[error], task_idx=17), task_idx=17)
    assert rollout.has_error and rollout.error.type == "SetupFailed"
    assert rollout.task.data.idx == 17
    assert rollout.info["prime_rl_episode"]["id"] == "failed"
    assert not rollout.nodes and not rollout.samples
