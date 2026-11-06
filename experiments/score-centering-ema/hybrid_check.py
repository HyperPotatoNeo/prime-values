"""Two bounded synthetic updates on eight GPUs: two replicas, four shards each."""

import json
import os
import signal
import subprocess
import time
from pathlib import Path

from prepare import configuration

_, config = configuration("ema", 101, steps=2)
trainer = config.trainer
trainer.output_dir = Path("/tmp/sc-ema-validation/hybrid")
trainer.model.seq_len = 8192
trainer.ckpt = None
trainer.wandb = None
from prime_rl.configs.trainer import FakeDataLoaderConfig

trainer.data.fake = FakeDataLoaderConfig(batch_size=16)
root = Path(__file__).parent / "results/hybrid"
root.mkdir(parents=True, exist_ok=True)
path = root / "trainer.json"
path.write_text(trainer.model_dump_json(indent=2))
start = time.time()
with (root / "driver.log").open("w") as log:
    process = subprocess.Popen(
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
        start_new_session=True,
    )
    try:
        rc = process.wait(timeout=900)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
receipt = {
    "exit_code": rc,
    "seconds": time.time() - start,
    "gpus": 8,
    "nodes": 1,
    "dp_replicate": 2,
    "dp_shard": 4,
    "steps": 2,
    "synthetic_data": True,
    "seq_len": 8192,
    "note": "Not an inter-node throughput measurement",
}
(root / "receipt.json").write_text(json.dumps(receipt, indent=2))
print(receipt)
if rc:
    raise RuntimeError("Hybrid check failed")
