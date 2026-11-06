"""Bounded actual Qwen/RG pipeline checks; never launches the research sweep."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from prime_rl.configs.rl import RLConfig


def config(arm, out, steps, resume=None):
    sampling = {
        "temperature": 1.0,
        "max_completion_tokens": None,
        "extra_body": {
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
        },
    }
    if arm == "sc":
        sampling["extra_body"]["extra_args"] = {"score_centering_top_k": 128}
    d = {
        "model": {"name": "Qwen/Qwen3-4B-Instruct-2507"},
        "output_dir": str(out),
        "seq_len": 8192,
        "max_steps": steps,
        "log": {"level": "debug"},
        "ckpt": {"interval": 1, "keep_last": 2},
        "trainer": {
            "enable_token_export": True,
            "weight_broadcast": {"type": "filesystem", "acknowledged_cleanup": True},
            "model": {"impl": "custom", "attn": "flash_attention_2", "fused_lm_head_token_chunk_size": 1024},
            "optim": {"lr": 1e-6},
            "loss": {
                "type": "research",
                "score_centering": arm == "sc",
                "top_k": 128,
                "importance_cap": 2.0,
                "ema_kl_beta": 0.01 if arm == "ema" else 0.0,
                "ema_decay": 0.95,
            },
        },
        "inference": {
            "model": {"max_model_len": 8192, "enforce_eager": True},
            "gpu_memory_utilization": 0.5,
            "vllm_extra": {"max_logprobs": 128, "generation_config": "vllm"},
        },
        "orchestrator": {
            "checkpoint_weights_dir": str(out / "weights"),
            "batch_size": 16,
            "post_batch_filters": [
                {"type": t, "enforce": False} for t in ("gibberish", "repetition", "zero_advantage")
            ],
            "group_size": 8,
            "max_inflight_rollouts": 16,
            "algo": {"type": "grpo"},
            "train": {
                "sampling": sampling,
                "env": [
                    {
                        "id": "rg-mix-env",
                        "args": {
                            "dataset_path": "/data/rg-dataset",
                            "num_train_examples": 10000,
                            "num_eval_examples": 256,
                            "seed": 42,
                        },
                    }
                ],
            },
        },
        "deployment": {"type": "single_node", "gpus_per_node": 8, "num_train_gpus": 2, "num_infer_gpus": 1},
    }
    if resume is not None:
        d["ckpt"]["resume_step"] = resume
    c = RLConfig.model_validate(d)
    assert c.value_function is None
    assert c.trainer.model.optimization_dtype == c.trainer.model.reduce_dtype == "float32"
    return c


def launch(arm, steps, resume):
    root = Path("/data/siddarth-sc-ema/experiments/score-centering-ema/results") / (arm + "-8k-v2")
    root.mkdir(parents=True, exist_ok=True)
    out = Path("/tmp/sc-ema-validation") / (arm + "-8k-v2")
    c = config(arm, out, steps, resume)
    phase = "fresh" if resume is None else "resume"
    path = root / f"{phase}.json"
    path.write_text(c.model_dump_json(indent=2))
    start = time.time()
    with (root / f"{phase}.log").open("w") as log:
        proc = subprocess.Popen(
            ["uv", "run", "--no-sync", "rl", "@", str(path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            rc = proc.wait(timeout=1200)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            raise
    receipt = {
        "arm": arm,
        "phase": phase,
        "exit_code": rc,
        "seconds": time.time() - start,
        "output": str(out),
        "requested_steps": steps,
        "resume": resume,
    }
    (root / f"{phase}-receipt.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt), flush=True)
    if rc:
        raise RuntimeError(f"{arm}/{phase} failed; inspect {root}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("arm", choices=["base", "sc", "ema"])
    p.add_argument("--resume", type=int)
    p.add_argument("--steps", type=int, default=2)
    a = p.parse_args()
    launch(a.arm, a.steps, a.resume)
