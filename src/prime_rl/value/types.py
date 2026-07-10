import msgspec


class ValueTrainingSample(msgspec.Struct, array_like=True, gc=False):
    token_ids: list[int]
    mask: list[bool]
    targets: list[float]


class ValueTrainingBatch(msgspec.Struct, array_like=True, gc=False):
    samples: list[ValueTrainingSample]
    batch_id: int
    num_rollouts: int
    policy_version_min: int
    policy_version_max: int
    value_version_min: int
    value_version_max: int


class ValueEvaluationRequest(msgspec.Struct, gc=False):
    token_ids: list[list[int]]


class ValueEvaluationResponse(msgspec.Struct, gc=False):
    values: list[list[float]]
    version: int


class ValueVersionResponse(msgspec.Struct, gc=False):
    version: int
