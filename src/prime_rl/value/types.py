import msgspec


class ValueTrainingSample(msgspec.Struct, array_like=True, gc=False):
    token_ids: list[int]
    mask: list[bool]
    targets: list[float]


class ValueTrainingRollout(msgspec.Struct, array_like=True, gc=False):
    samples: list[ValueTrainingSample]
    rollout_id: int
    policy_version: int
    value_version: int


class ValueTrainingBatch(msgspec.Struct, array_like=True, gc=False):
    samples: list[ValueTrainingSample]
    num_rollouts: int
    rollout_id_min: int
    rollout_id_max: int
    policy_version_min: int
    policy_version_max: int
    value_version_min: int
    value_version_max: int
    replay_attempt_min: int
    replay_attempt_max: int
    replay_attempt_mean: float


class ValueEvaluationRequest(msgspec.Struct, gc=False):
    token_ids: list[list[int]]


class ValueEvaluationResponse(msgspec.Struct, gc=False):
    values: list[list[float]]
    version: int


class ValueVersionResponse(msgspec.Struct, gc=False):
    version: int
