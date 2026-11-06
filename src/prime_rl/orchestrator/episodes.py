"""Project a complete upstream episode onto the supported single-trace credit unit."""

from __future__ import annotations

import verifiers.v1 as vf

from prime_rl.orchestrator.types import Rollout


class UnsupportedEpisodeError(ValueError):
    """The episode requires credit semantics this runtime does not define."""


def from_episode(episode: vf.WireEpisode, *, task_idx: int = -1) -> Rollout:
    # Selection precedes eligibility: a frozen or failed sibling still makes
    # this a multi-trace episode, with undefined cross-agent value credit.
    if len(episode.traces) > 1:
        raise UnsupportedEpisodeError(
            f"episode {episode.id!r} contains {len(episode.traces)} original traces; "
            "this runtime requires one trace per episode"
        )
    if episode.traces:
        trace = episode.traces[0]
        rollout = Rollout[vf.WireTaskData].model_construct(**dict(trace))
        rollout.errors = list(trace.errors)
        rollout.info = dict(trace.info)
        if not episode.ok:
            rollout.ok = False
            rollout.errors.append(
                episode.error or vf.Error(type="EpisodeFailed", message="The environment episode did not complete")
            )
        if not rollout.ok and not rollout.errors:
            rollout.errors.append(vf.Error(type="TraceFailed", message="The trace did not complete"))
    else:
        rollout = Rollout[vf.WireTaskData](
            task=vf.TraceTask(type="DispatchTask", data=vf.WireTaskData(idx=task_idx)),
            ok=False,
            errors=list(episode.errors) or [vf.Error(type="EmptyEpisode", message="The environment returned no trace")],
        )
    rollout.info["prime_rl_episode"] = {
        "id": episode.id,
        "env": episode.env,
        "ok": episode.ok,
        "errors": [error.model_dump(mode="json", exclude_none=True) for error in episode.errors],
    }
    return rollout
