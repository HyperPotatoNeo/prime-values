import pytest
from pydantic import TypeAdapter, ValidationError

from prime_rl.configs.algorithm import AlgoConfig
from prime_rl.configs.rl import RLConfig

ALGORITHM = TypeAdapter(AlgoConfig)


def _grpo_baseline(baseline: dict):
    return ALGORITHM.validate_python({"type": "grpo", "baseline": baseline}).baseline


def test_adaptive_baseline_defaults_and_static_escape():
    baseline = _grpo_baseline({"type": "tether"})
    assert baseline.group == "leave_one_out"
    assert baseline.rho == 0.5
    assert baseline.adaptive is not None
    assert baseline.adaptive.initial_rho == 0.0
    assert baseline.adaptive.ema_decay == 0.95
    assert baseline.adaptive.ridge == 1e-6
    assert baseline.adaptive.position is None

    static = _grpo_baseline({"type": "tether", "adaptive": "None"})
    assert static.adaptive is None and static.rho == 0.5


def test_tether_position_config_is_adaptive_only_and_resolves_fixed_bins():
    positioned = _grpo_baseline(
        {
            "type": "tether",
            "adaptive": {
                "position": {
                    "bin_size": 128,
                    "max_action_tokens": 512,
                    "min_bin_rollouts": 3,
                }
            },
        }
    )
    assert positioned.adaptive is not None and positioned.adaptive.position is not None
    assert positioned.adaptive.position.resolve(policy_seq_len=1024, batch_size=8) == (4, 128, 3)

    with pytest.raises(ValidationError, match="position"):
        _grpo_baseline(
            {
                "type": "tether",
                "adaptive": "None",
                "position": {"bin_size": 128},
            }
        )


@pytest.mark.parametrize(
    ("position", "match"),
    [
        ({"bin_size": 1024, "max_action_tokens": 1024}, "at least two bins"),
        ({"bin_size": 1, "max_action_tokens": 129}, "maximum is 128"),
        ({"bin_size": 128, "max_action_tokens": 1025}, "cannot exceed policy sequence length"),
        (
            {"bin_size": 128, "max_action_tokens": 512, "min_bin_rollouts": 9},
            "cannot exceed adaptive batch_size",
        ),
    ],
)
def test_tether_position_config_rejects_invalid_resolved_contract(position, match):
    baseline = _grpo_baseline({"type": "tether", "adaptive": {"position": position}})
    assert baseline.adaptive is not None and baseline.adaptive.position is not None
    with pytest.raises(ValueError, match=match):
        baseline.adaptive.position.resolve(policy_seq_len=1024, batch_size=8)


def test_tether_position_default_support_is_one_eighth_of_adaptive_batch():
    baseline = _grpo_baseline(
        {
            "type": "tether",
            "adaptive": {"position": {"bin_size": 1024, "max_action_tokens": 2048}},
        }
    )
    assert baseline.adaptive is not None and baseline.adaptive.position is not None
    assert baseline.adaptive.position.resolve(policy_seq_len=2048, batch_size=256) == (2, 1024, 32)


def test_adaptive_tether_requires_loo_but_static_mode_allows_mean():
    with pytest.raises(ValidationError, match="requires group='leave_one_out'"):
        _grpo_baseline({"type": "tether", "group": "mean"})
    assert _grpo_baseline({"type": "tether", "group": "mean", "adaptive": "None"}).group == "mean"


def test_tether_requires_value_function():
    with pytest.raises(ValidationError, match="value-backed baselines require"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {
                    "group_size": 2,
                    "algo": {"type": "grpo", "baseline": {"type": "tether"}},
                },
            }
        )


def test_adaptive_tether_inherits_value_batch_and_enables_warmup():
    config = RLConfig.model_validate(
        {
            "trainer": {},
            "orchestrator": {
                "group_size": 2,
                "algo": {"type": "grpo", "baseline": {"type": "tether"}},
            },
            "value_function": {"batch_size": 7},
            "deployment": {"type": "single_node", "gpus_per_node": 4},
        }
    )
    baseline = config.orchestrator.algo.baseline
    assert baseline.adaptive_config is not None
    assert baseline.adaptive_config.batch_size == 7
    assert config.value_function is not None and config.value_function.warmup_updates == 1


def test_positioned_tether_contract_resolves_after_adaptive_batch_inheritance():
    raw = {
        "trainer": {},
        "orchestrator": {
            "seq_len": 512,
            "group_size": 2,
            "algo": {
                "type": "grpo",
                "baseline": {
                    "type": "tether",
                    "adaptive": {
                        "position": {
                            "bin_size": 128,
                            "max_action_tokens": 512,
                            "min_bin_rollouts": 7,
                        }
                    },
                },
            },
        },
        "value_function": {"batch_size": 7},
        "deployment": {"type": "single_node", "gpus_per_node": 4},
    }
    config = RLConfig.model_validate(raw)
    baseline = config.orchestrator.algo.baseline
    assert baseline.adaptive_config is not None and baseline.adaptive_config.batch_size == 7

    raw["orchestrator"]["algo"]["baseline"]["adaptive"]["position"]["min_bin_rollouts"] = 8
    with pytest.raises(ValidationError, match="cannot exceed adaptive batch_size"):
        RLConfig.model_validate(raw)


def test_tether_rejects_length_penalty():
    with pytest.raises(ValidationError, match="cannot be combined"):
        ALGORITHM.validate_python(
            {
                "type": "grpo",
                "baseline": {"type": "tether"},
                "length_penalty": {},
            }
        )


def test_tether_requires_group_size_two():
    with pytest.raises(ValidationError, match="leave_one_out baseline requires group_size"):
        RLConfig.model_validate(
            {
                "trainer": {},
                "orchestrator": {
                    "group_size": 1,
                    "algo": {"type": "grpo", "baseline": {"type": "tether"}},
                },
                "value_function": {},
                "deployment": {"type": "single_node", "gpus_per_node": 4},
            }
        )
