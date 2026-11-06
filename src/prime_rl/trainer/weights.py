import fcntl
import json
import os
import shutil
import tempfile
import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Literal, cast

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors import SafetensorError, safe_open
from safetensors.torch import save_file
from torch import Tensor, nn
from torch.distributed.checkpoint.state_dict import _get_fqns as get_fqns
from torch.distributed.tensor import DTensor
from transformers.utils import (
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)

from prime_rl.trainer.lora import (
    clean_lora_state_dict,
)
from prime_rl.utils.logger import get_logger


def load_state_dict_keys(save_dir: Path) -> list[str]:
    """Load only the key names from safetensor files without reading tensor data."""
    keys: list[str] = []
    for safetensor_path in save_dir.glob("*.safetensors"):
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            keys.extend(f.keys())
    return keys


def load_state_dict(save_dir: Path) -> dict[str, Tensor]:
    """Load a state dict from a local directory with safetensor files."""
    safetensors_paths = list(save_dir.glob("*.safetensors"))
    state_dict = {}
    for safetensor_path in safetensors_paths:
        with safe_open(safetensor_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)
    return state_dict


def save_state_dict(
    state_dict: dict[str, Tensor],
    save_dir: Path,
    save_format: Literal["torch", "safetensors"] = "safetensors",
    save_sharded: bool = True,
    adapter: bool = False,
):
    """Save a state dict to a local directory in safetensors or torch format."""
    logger = get_logger()
    if adapter:
        weights_name = ADAPTER_SAFE_WEIGHTS_NAME if save_format == "safetensors" else ADAPTER_WEIGHTS_NAME
    else:
        weights_name = SAFE_WEIGHTS_NAME if save_format == "safetensors" else WEIGHTS_NAME
    save_dir.mkdir(parents=True, exist_ok=True)
    if save_sharded:
        filename_pattern = weights_name.replace(".bin", "{suffix}.bin").replace(".safetensors", "{suffix}.safetensors")
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict,
            filename_pattern=filename_pattern,
        )
        if state_dict_split.is_sharded:
            filenames = state_dict_split.filename_to_tensors.keys()
            logger.debug(f"Saving sharded weights to {len(filenames)} files: ({', '.join(filenames)})")
        else:
            logger.debug(f"Saving unsharded weights to {weights_name}")

        # Save weights (https://github.com/huggingface/transformers/blob/cd74917ffc3e8f84e4a886052c5ab32b7ac623cc/src/transformers/modeling_utils.py#L4252)
        filename_to_tensors = state_dict_split.filename_to_tensors.items()
        for shard_file, tensors in filename_to_tensors:
            shard = {}
            for tensor in tensors:
                assert isinstance(state_dict[tensor], Tensor)
                shard[tensor] = state_dict[tensor].contiguous()
                # delete reference, see https://github.com/huggingface/transformers/pull/34890
                del state_dict[tensor]
            if save_format == "safetensors":
                save_file(shard, save_dir / shard_file, metadata={"format": "pt"})
            else:
                torch.save(shard, save_dir / shard_file)
        del state_dict

        # Save index (https://github.com/huggingface/transformers/blob/cd74917ffc3e8f84e4a886052c5ab32b7ac623cc/src/transformers/modeling_utils.py#L4301)
        if state_dict_split.is_sharded:
            index = {
                "metadata": {**state_dict_split.metadata},
                "weight_map": state_dict_split.tensor_to_filename,
            }
            save_index_file = SAFE_WEIGHTS_INDEX_NAME if save_format == "safetensors" else WEIGHTS_INDEX_NAME
            save_index_file = save_dir / save_index_file
            # Save the index as well
            with open(save_index_file, "w", encoding="utf-8") as f:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                f.write(content)
    else:
        if save_format == "safetensors":
            save_file(state_dict, save_dir / weights_name, metadata={"format": "pt"})
        else:
            torch.save(state_dict, save_dir / weights_name)


def ensure_converted_weights(
    source: Path,
    target_format: Literal["prime", "hf"],
    convert: Callable[[dict[str, Tensor]], None],
    required_keys: set[str],
) -> Path:
    """Publish a complete derived cache, serializing independent trainer masters."""
    destination = source / target_format

    def complete(path: Path) -> bool:
        try:
            file_keys = {}
            for shard in path.glob("*.safetensors"):
                with safe_open(shard, framework="pt", device="cpu") as handle:
                    file_keys[shard.name] = set(handle.keys())
            keys = set().union(*file_keys.values()) if file_keys else set()
            if not keys or not required_keys <= keys or sum(map(len, file_keys.values())) != len(keys):
                return False
            index_path = path / SAFE_WEIGHTS_INDEX_NAME
            if not index_path.exists():
                return set(file_keys) == {SAFE_WEIGHTS_NAME}
            index = json.loads(index_path.read_text())
            weight_map = index.get("weight_map") if isinstance(index, dict) else None
            return (
                isinstance(weight_map, dict)
                and weight_map.keys() == keys
                and all(
                    isinstance(shard, str) and key in file_keys.get(shard, set()) for key, shard in weight_map.items()
                )
            )
        except (OSError, SafetensorError, json.JSONDecodeError):
            return False

    # A valid cache also works on read-only snapshots. Recheck after locking
    # because another trainer may finish conversion while we wait.
    if complete(destination):
        return destination
    # Only generated children are replaced; a directly selected model directory
    # is never repaired here. The lock also covers policy/value trainer races.
    with (source / f".{target_format}-conversion.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if complete(destination):
            return destination
        if destination.is_symlink():
            raise ValueError(f"Refusing to replace an incomplete symlinked conversion cache: {destination}")
        if destination.exists() and not destination.is_dir():
            raise ValueError(f"Conversion cache is not a directory: {destination}")
        if destination.exists():
            get_logger().warning(f"Rebuilding incomplete converted weight cache {destination}")
        temporary = Path(tempfile.mkdtemp(dir=source, prefix=f".{target_format}-conversion-"))
        obsolete = None
        try:
            state_dict = load_state_dict(source)
            convert(state_dict)
            missing = required_keys - state_dict.keys()
            if missing:
                raise ValueError(f"Converted weights are missing required model keys: {sorted(missing)}")
            save_state_dict(state_dict, temporary)
            if not complete(temporary):
                raise RuntimeError(f"Converted weight cache failed validation: {temporary}")
            if destination.exists():
                obsolete = Path(tempfile.mkdtemp(dir=source, prefix=f".{target_format}-incomplete-"))
                obsolete.rmdir()
                os.replace(destination, obsolete)
            os.replace(temporary, destination)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)
        if obsolete is not None:
            shutil.rmtree(obsolete)
        return destination


def gather_weights_on_master(
    model: nn.Module, is_master: bool, dtype: torch.dtype = torch.bfloat16
) -> dict[str, Tensor]:
    """Gather distributed weights on CPU on master rank."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="torch.distributed")
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed.*")

        cpu_state = {}
        for key, value in model.state_dict().items():
            if isinstance(value, DTensor):
                # only gather after the downcast to dtype as it will be faster
                value = cast(DTensor, value.to(dtype)).full_tensor()

            if is_master:
                key = get_fqns(model, key)
                assert len(key) == 1
                key = next(iter(key))
                # TODO(Sami) Blocking to avoid race condition, should make non-blocking long-term tho
                cpu_state[key] = value.to("cpu", non_blocking=False)
        torch.distributed.barrier()

    # Always clean up the state dict for HF compatibility
    if any(".base_layer." in key or "lora_A" in key or "lora_B" in key for key in cpu_state.keys()):
        cpu_state = clean_lora_state_dict(cpu_state)

    return cpu_state
