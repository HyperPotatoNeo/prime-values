"""CPU-only acceptance of real target config loading and launcher arguments."""

import argparse
import hashlib
import inspect
import json
import tempfile
from copy import copy
from dataclasses import fields
from pathlib import Path

import transformers
import vllm
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from transformers import LlamaConfig, Qwen3Config
from vllm.engine.arg_utils import EngineArgs
from vllm.entrypoints.launchers.cli_args import make_arg_parser
from vllm.entrypoints.scale_out.factories import register_scale_out_api_routers
from vllm.entrypoints.scale_out.token_in_token_out.api_router import generate
from vllm.transformers_utils.config import get_config, patch_rope_parameters
from vllm.utils.argparse_utils import FlexibleArgumentParser

import prime_rl.configs.inference as inference_module
import prime_rl.configs.trainer as trainer_module
from prime_rl.configs.inference import InferenceConfig
from prime_rl.configs.trainer import ModelConfig
from prime_rl.configs.value import ValueFunctionConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--skip-engine-args", action="store_true", help="CPU-only host without a vLLM device; rerun full probe on GPU"
    )
    args = parser.parse_args()
    assert not args.output.exists(), "Use a fresh receipt path"
    root = args.source_root.resolve()
    for module in (inference_module, trainer_module):
        assert Path(module.__file__).resolve().is_relative_to(root)

    default = InferenceConfig(model={"name": "Qwen/Qwen3-4B"}).to_vllm()
    assert default.logprobs_mode == "processed_logprobs"
    assert default.enable_scale_out is True
    assert not hasattr(default, "kv_transfer_config")
    # Exercise the real upstream route gate without inventing an engine/model.
    for enabled, expected_status in ((False, 403), (True, 422)):
        route_args = copy(default)
        route_args.enable_scale_out = enabled
        app = FastAPI()
        app.state.args = route_args
        register_scale_out_api_routers(app, ("generate",))
        routes = [route for route in app.routes if getattr(route, "path", None) == "/inference/v1/generate"]
        assert len(routes) == 1
        assert (routes[0].endpoint is generate) == enabled
        response = TestClient(app).post("/inference/v1/generate", json={})
        assert response.status_code == expected_status, response.text
    assert not hasattr(default, "hf_overrides") and not hasattr(default, "rope_scaling")
    server_arguments = {
        "host",
        "port",
        "liveness_timeout_seconds",
        "chat_template",
        "tool_call_parser",
        "api_server_count",
        "enable_auto_tool_choice",
        "enable_scale_out",
    }
    assert not (set(vars(default)) - {field.name for field in fields(EngineArgs)} - server_arguments)
    if not args.skip_engine_args:
        parsed = make_arg_parser(FlexibleArgumentParser()).parse_args([], namespace=default)
        assert parsed.enable_scale_out is True and parsed.tokens_only is False
        engine = EngineArgs.from_cli_args(parsed)
        assert engine.logprobs_mode == "processed_logprobs"
        assert engine.return_sampling_mask is False

    results = []
    with tempfile.TemporaryDirectory() as tmp:
        for config_cls in (Qwen3Config, LlamaConfig):
            base = dict(rope_theta=1000000.0, partial_rotary_factor=0.5, max_position_embeddings=65536)
            directory = Path(tmp) / config_cls.__name__
            config_cls(**base).save_pretrained(directory)
            for legacy_alias in (False, True):
                scaling = {
                    "type" if legacy_alias else "rope_type": "yarn",
                    "factor": 2.0,
                    "original_max_position_embeddings": 32768,
                }
                if legacy_alias:
                    scaling["rope_theta"] = 500000.0
                    scaling["partial_rotary_factor"] = 0.25
                original_scaling = json.loads(json.dumps(scaling))
                namespace = InferenceConfig(
                    model={"name": str(directory), "rope_scaling": json.dumps(scaling) if legacy_alias else scaling}
                ).to_vllm()
                override = namespace.hf_overrides
                # Exercise vLLM's dummy architecture probe and actual HF loader.
                actual = get_config(str(directory), trust_remote_code=False, hf_overrides_fn=override)
                assert scaling == original_scaling and override.scaling == original_scaling
                expected = config_cls(**base, rope_scaling=scaling)
                patch_rope_parameters(expected)
                assert actual.rope_parameters == expected.rope_parameters
                results.append(
                    {
                        "config": config_cls.__name__,
                        "json_legacy_alias": legacy_alias,
                        "parameters": actual.rope_parameters,
                    }
                )
            nested = config_cls(**base)
            nested.rope_parameters = {"full_attention": {"rope_theta": 1000000.0, "rope_type": "default"}}
            try:
                override(nested)
            except ValueError as error:
                assert "flat text" in str(error)
            else:
                raise AssertionError("Nested RoPE must fail explicitly")
        invalid = InferenceConfig(
            model={"name": str(directory), "rope_scaling": {"type": "yarn", "rope_type": "default", "factor": 2.0}}
        ).to_vllm()
        try:
            get_config(str(directory), trust_remote_code=False, hf_overrides_fn=invalid.hf_overrides)
        except ValueError as error:
            assert "conflict" in str(error).lower()
        else:
            raise AssertionError("Native RoPE validation must reject conflicting type aliases")

    for name in ("fa4", "flash_attention_4"):
        for constructor in (
            lambda: ModelConfig(attn=name, impl="custom"),
            lambda: ValueFunctionConfig(model={"attn": name, "impl": "custom"}),
        ):
            try:
                constructor()
            except ValidationError as error:
                assert "TVM-FFI" in str(error)
            else:
                raise AssertionError("Unavailable FA4 must fail before runtime")
    for name in ("flash_attention_2", "flash_attention_3"):
        assert ModelConfig(attn=name, impl="custom").attn == name

    paths = [
        Path(inference_module.__file__),
        Path(trainer_module.__file__),
        Path(inspect.getfile(get_config)),
        Path(inspect.getfile(register_scale_out_api_routers)),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "status": "passed",
                "transformers": transformers.__version__,
                "vllm": vllm.__version__,
                "source_root": str(root),
                "engine_args_checked": not args.skip_engine_args,
                "actual_scale_out_route_gate_checked": True,
                "sources": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                "rope_cases": results,
                "scope": "Real config loading and launcher argument validation only; no model weights or GPU execution.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
