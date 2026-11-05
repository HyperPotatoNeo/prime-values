"""Run under an external timeout; fatal mode must exit 1 before peers finish.

Example: timeout 45s uv run --no-sync torchrun --standalone --nproc-per-node=2
 tests/integration/distributed_failure_probe.py
Success mode must exit zero and destroy the process group normally.
"""

import argparse
import time
from datetime import timedelta

import torch
import torch.distributed as dist

from prime_rl.trainer.utils import setup_torch_distributed
from prime_rl.trainer.world import get_world
from prime_rl.utils.logger import setup_logger
from prime_rl.utils.utils import clean_exit

_PENDING = None


@clean_exit
def run(success):
    global _PENDING
    setup_logger("info")
    setup_torch_distributed(timeout=timedelta(seconds=120))
    tensor = torch.ones(1, device="cuda")
    dist.all_reduce(tensor)
    assert tensor.item() == get_world().world_size
    if success:
        return
    if get_world().rank == 0:
        _PENDING = dist.all_reduce(tensor, async_op=True)
        raise RuntimeError("intentional rank failure with unmatched collective")
    time.sleep(120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--success", action="store_true")
    run(parser.parse_args().success)
    assert not dist.is_initialized()
