"""Record configured sampling, model overrides, and unchanged request builders."""

import ast
import hashlib
import json
import re
from pathlib import Path

from prime_rl.configs.rl import RLConfig
from prime_rl.orchestrator.envs import Env
from prime_rl.orchestrator.sampler import Sampler

root = Path("/data/task-ownership-001")
prior = Path("/data/episode-sync-001")
builders = [
    (
        "packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py",
        "TrainSamplingConfig",
        "to_sampling_args",
    ),
    (
        "packages/prime-rl-configs/src/prime_rl/configs/orchestrator.py",
        "OrchestratorConfig",
        "resolve_env_config",
    ),
    ("src/prime_rl/orchestrator/envs.py", "Env", "_sampling"),
    ("src/prime_rl/orchestrator/sampler.py", "Sampler", "sampling_args"),
    ("deps/verifiers/verifiers/v1/types.py", "SamplingConfig", None),
]


def fingerprint(source, name, method):
    path = source / name
    tree = ast.parse(path.read_text())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == method[0])
    if method[1] is not None:
        node = next(node for node in node.body if isinstance(node, ast.FunctionDef) and node.name == method[1])
    return {
        "path": str(path),
        "line": node.lineno,
        "ast_sha256": hashlib.sha256(ast.dump(node).encode()).hexdigest(),
    }


source_checks = []
for name, cls, method in builders:
    old = fingerprint(prior / "source", name, (cls, method))
    new = fingerprint(root / "source", name, (cls, method))
    assert old["ast_sha256"] == new["ast_sha256"], name
    source_checks.append({"historical": old, "current": new})

vllm_source = Path("/app/.venv/lib/python3.12/site-packages/vllm/sampling_params.py")
params = next(
    node
    for node in ast.parse(vllm_source.read_text()).body
    if isinstance(node, ast.ClassDef) and node.name == "SamplingParams"
)
penalties = {}
for node in params.body:
    if (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id in {"presence_penalty", "frequency_penalty", "repetition_penalty"}
    ):
        penalties[node.target.id] = ast.literal_eval(node.value)
assert penalties == {
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repetition_penalty": 1.0,
}, penalties
records = {}
for label, base in [("episode_unit", prior), ("task_data_unit", root)]:
    evidence = base / "output/dedicated/fresh"
    config_path = evidence / "resolved.json"
    config = RLConfig.model_validate_json(config_path.read_text())
    env_config = config.orchestrator.train.env[0]
    assert env_config.algo.sampling.source == "policy"
    env = Env(env_config)
    env.sampling_args = Sampler(env_config.algo.sampling, None).sampling_args(env_config.sampling.to_sampling_args())
    request = env._sampling(None).model_dump(exclude_none=True)
    assert request["temperature"] == request["top_p"] == 1.0
    assert request["extra_body"]["top_k"] == -1 and request["extra_body"]["min_p"] == 0.0
    assert not any(key in request or key in request["extra_body"] for key in penalties)
    log_path = (
        evidence / "logs/inference.log"
        if (evidence / "logs/inference.log").exists()
        else base / "run/logs/inference.log"
    )
    if not log_path.exists():
        log_path = base / "output/dedicated/run/logs/inference.log"
    content = log_path.read_text()
    match = re.search(r"overridden by the model's `generation_config.json`: `([^`]+)`", content)
    assert match, log_path
    model_overrides = ast.literal_eval(match[1])
    assert model_overrides == {"temperature": 0.7, "top_k": 20, "top_p": 0.8}, model_overrides
    records[label] = {
        "resolved_config": str(config_path),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "inference_log": str(log_path),
        "model_generation_config_overrides": model_overrides,
        "constructed_sampling_request": request,
        "effective_distribution": {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            **penalties,
        },
    }
result = {
    "scope": "Configured policy requests for these two Qwen smoke runs; production request construction and source/log audit, not a network wire capture. No new generation or judge call.",
    "request_builders_unchanged": source_checks,
    "vllm_defaults_source": str(vllm_source),
    "vllm_defaults_source_sha256": hashlib.sha256(vllm_source.read_bytes()).hexdigest(),
    "runs": records,
}
(root / "sampling-audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(
    json.dumps(
        {
            "sampling_audit": "passed",
            "runs": list(records),
            "builder_checks": len(source_checks),
        }
    )
)
