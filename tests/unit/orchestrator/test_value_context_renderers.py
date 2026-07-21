from types import SimpleNamespace

import pytest
from renderers import create_renderer
from renderers.base import load_tokenizer
from renderers.configs import AutoRendererConfig

from prime_rl.orchestrator.train_sink import TrainSink


@pytest.mark.slow
@pytest.mark.parametrize(
    ("model_name", "expected_insert_at"),
    [
        pytest.param("Qwen/Qwen3-4B-Instruct-2507", 0, id="qwen3-no-bos"),
        pytest.param("NousResearch/Hermes-3-Llama-3.1-8B", 1, id="llama3-bos"),
    ],
)
def test_real_renderer_prefix_preserves_complete_multiturn_policy_sequence(
    model_name: str,
    expected_insert_at: int,
):
    tokenizer = load_tokenizer(model_name)
    renderer = create_renderer(tokenizer, AutoRendererConfig())
    policy_messages = [
        {"role": "system", "content": "Solve the task one step at a time."},
        {"role": "user", "content": "Start with the first step."},
        {"role": "assistant", "content": "First step."},
        {"role": "user", "content": "Continue."},
        {"role": "assistant", "content": "Second step."},
    ]
    privileged_message = {"role": "system", "content": "Privileged reference answer."}
    original = renderer.render_ids(policy_messages, add_generation_prompt=False)
    sample = SimpleNamespace(token_ids=list(original))
    rollout = SimpleNamespace(
        env_name="test-environment",
        task=SimpleNamespace(idx=0, value_function_prompt=privileged_message["content"]),
    )
    sink = TrainSink.__new__(TrainSink)
    sink.renderer = renderer
    sink.tokenizer = tokenizer
    sink._value_seq_len = 1_000_000

    prefix = sink._build_value_prefix(rollout, [sample], enabled=True)

    assert prefix is not None
    assert prefix.insert_at == expected_insert_at
    augmented = prefix.apply(sample.token_ids)
    assert augmented == renderer.render_ids([privileged_message, *policy_messages], add_generation_prompt=False)
    assert prefix.project(augmented) == original
    assert sample.token_ids == original
    assert len(augmented) == len(original) + len(prefix.token_ids)
