from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import torch
from safetensors.torch import save_file

from prime_rl.trainer.weights import ensure_converted_weights, load_state_dict


def _convert(state):
    state["converted.weight"] = state.pop("original.weight")


@pytest.mark.parametrize("partial", ["empty", "missing-key", "truncated"])
def test_incomplete_conversion_is_rebuilt_from_unchanged_source(tmp_path, partial):
    original = torch.arange(4, dtype=torch.float32)
    save_file({"original.weight": original}, tmp_path / "model.safetensors")
    source = (tmp_path / "model.safetensors").read_bytes()
    destination = tmp_path / "prime"
    destination.mkdir()
    if partial == "missing-key":
        save_file({"wrong.weight": original}, destination / "model.safetensors")
    elif partial == "truncated":
        (destination / "model.safetensors").write_bytes(b"truncated")
    assert ensure_converted_weights(tmp_path, "prime", _convert, {"converted.weight"}) == destination
    torch.testing.assert_close(load_state_dict(destination)["converted.weight"], original)
    assert (tmp_path / "model.safetensors").read_bytes() == source
    assert not list(tmp_path.glob(".prime-conversion-*/"))


def test_failed_conversion_does_not_publish_partial_weights(tmp_path):
    save_file({"original.weight": torch.ones(2)}, tmp_path / "model.safetensors")
    destination = tmp_path / "prime"
    destination.mkdir()
    (destination / "partial").write_text("previous interrupted attempt")

    def broken(state):
        state.clear()
        raise RuntimeError("conversion interrupted")

    with pytest.raises(RuntimeError, match="conversion interrupted"):
        ensure_converted_weights(tmp_path, "prime", broken, {"converted.weight"})
    assert (destination / "partial").read_text() == "previous interrupted attempt"
    assert not list(tmp_path.glob(".prime-conversion-*/"))


def test_independent_trainers_share_one_complete_conversion(tmp_path):
    save_file({"original.weight": torch.ones(2)}, tmp_path / "model.safetensors")
    barrier = Barrier(2)
    conversions = []

    def convert(state):
        conversions.append(1)
        _convert(state)

    def prepare():
        barrier.wait(timeout=5)
        return ensure_converted_weights(tmp_path, "prime", convert, {"converted.weight"})

    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.submit(prepare), pool.submit(prepare)
        assert first.result(timeout=10) == second.result(timeout=10)
    assert len(conversions) == 1
    torch.testing.assert_close(load_state_dict(tmp_path / "prime")["converted.weight"], torch.ones(2))


@pytest.mark.parametrize("index", [None, "malformed", "missing-shard", "wrong-key"])
def test_incomplete_shard_index_is_rebuilt(tmp_path, index):
    import json

    save_file({"original.weight": torch.ones(2)}, tmp_path / "model.safetensors")
    destination = tmp_path / "prime"
    destination.mkdir()
    save_file({"converted.weight": torch.ones(2)}, destination / "model-00001-of-00002.safetensors")
    save_file({"extra.weight": torch.ones(2)}, destination / "model-00002-of-00002.safetensors")
    if index is not None:
        mapping = {
            "converted.weight": "missing.safetensors"
            if index == "missing-shard"
            else "model-00001-of-00002.safetensors",
            "wrong.weight" if index == "wrong-key" else "extra.weight": "model-00002-of-00002.safetensors",
        }
        (destination / "model.safetensors.index.json").write_text(
            "broken JSON" if index == "malformed" else json.dumps({"weight_map": mapping})
        )
    ensure_converted_weights(tmp_path, "prime", _convert, {"converted.weight"})
    assert list(destination.glob("*.safetensors")) == [destination / "model.safetensors"]
    assert not (destination / "model.safetensors.index.json").exists()


def test_complete_cache_needs_no_write_or_conversion(tmp_path, monkeypatch):
    from pathlib import Path

    destination = tmp_path / "prime"
    destination.mkdir()
    save_file({"converted.weight": torch.ones(2)}, destination / "model.safetensors")
    original_open = Path.open

    def readonly_open(path, mode="r", *args, **kwargs):
        if any(flag in mode for flag in "wa+"):
            raise PermissionError("read-only snapshot")
        return original_open(path, mode, *args, **kwargs)

    def unexpected_conversion(state):
        pytest.fail("complete cache was converted again")

    monkeypatch.setattr(Path, "open", readonly_open)
    assert ensure_converted_weights(tmp_path, "prime", unexpected_conversion, {"converted.weight"}) == destination
