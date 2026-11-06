"""Compare execution-only settings on the same deterministic full-length fake batches."""

import json
import os
import subprocess
import time
from pathlib import Path

from prepare import configuration

from prime_rl.configs.trainer import FakeDataLoaderConfig, TrainerConfig
from prime_rl.utils.config import cli

root = Path("/data/outputs/siddarth-performance-validation-001/trainer")
root.mkdir(parents=True, exist_ok=True)
for name, offload, checkpoint, replicas in [
    ("baseline", True, True, 1),
    ("no-offload", False, True, 1),
    ("no-recompute", False, False, 1),
    ("four-shards", False, False, 2),
    ("unsharded", False, False, 8),
]:
    _, c = configuration("ema", 101, steps=4)
    t = c.trainer
    t.output_dir = root / name
    t.model.dp_replicate = replicas
    if not offload:
        t.model.ac_offloading = None
    if not checkpoint:
        t.model.ac = None
    t.ckpt = None
    t.wandb = None
    # 128/8 has the same 16 full-length microbatches/rank as 256/16.
    t.data.fake = FakeDataLoaderConfig(batch_size=128)
    path = root / f"{name}.json"
    path.write_text(t.model_dump_json(indent=2))
    assert cli(TrainerConfig, args=["@", str(path)]).model_dump() == t.model_dump()
    env = dict(os.environ, PRIME_RL_METRICS_FILE=str(root / f"{name}-metrics.jsonl"))
    start = time.time()
    with (root / f"{name}.log").open("w") as log:
        result = subprocess.run(
            [
                "uv",
                "run",
                "--no-sync",
                "torchrun",
                "--standalone",
                "--nproc-per-node=8",
                "-m",
                "prime_rl.trainer.rl.train",
                "@",
                str(path),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            timeout=900,
        )
    receipt = {"name": name, "exit_code": result.returncode, "elapsed": time.time() - start}
    (root / f"{name}-receipt.json").write_text(json.dumps(receipt, indent=2))
    print(receipt, flush=True)
    if result.returncode:
        raise RuntimeError(f"Benchmark failed: {name}")
