"""One real trainer update with its normal fake-data source and logging caller.

Run unchanged against the frozen and candidate checkouts in separate one-rank
torchrun processes. This is an integration smoke, not a throughput benchmark.
"""

import argparse
import hashlib
import inspect
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from prime_rl.configs.trainer import TrainerConfig
from prime_rl.trainer.rl.train import train
from prime_rl.trainer.utils import TensorSummaries
from prime_rl.utils.monitor import get_monitor


def _weights(path):
    result = {}
    for shard in sorted(path.glob("*.safetensors")):
        loaded = load_file(str(shard))
        assert not result.keys() & loaded.keys()
        result.update(loaded)
    assert result, path
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    assert not args.output.exists(), "use a fresh output directory"
    sources = {
        "trainer": Path(inspect.getfile(inspect.unwrap(train))).resolve(),
        "summaries": Path(inspect.getfile(TensorSummaries)).resolve(),
        "config": Path(inspect.getfile(TrainerConfig)).resolve(),
    }
    metrics_source = sources["trainer"].with_name("metrics.py")
    if metrics_source.is_file():
        sources["token_metrics"] = metrics_source
    assert all(path.is_relative_to(args.source_root.resolve()) for path in sources.values())
    args.output.mkdir(parents=True)
    seed = args.output / "seed"
    torch.manual_seed(713)
    model = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=120000,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=128,
            tie_word_embeddings=False,
        )
    )
    model.save_pretrained(seed)
    del model
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"<unk>": 0, "<eos>": 1}, unk_token="<unk>")),
        unk_token="<unk>",
        eos_token="<eos>",
    )
    tokenizer.save_pretrained(seed)
    config = TrainerConfig.model_validate(
        {
            "model": {
                "name": str(seed),
                "seq_len": 64,
                "attn": "flash_attention_2",
                "impl": "custom",
                "ep": 1,
                "compile": None,
                "ac": None,
                "ac_offloading": None,
                "optim_cpu_offload": False,
            },
            "tokenizer": {"name": str(seed)},
            "data": {"fake": {"batch_size": 2, "generate_samples": True}},
            "optim": {"type": "adamw", "lr": 1e-3, "weight_decay": 0, "max_norm": None},
            "max_steps": 1,
            "output_dir": str(args.output / "runtime"),
            "ckpt": {"weights_only": True, "weights": {"save_sharded": False}},
            "log": {"level": "INFO"},
            "gc": None,
            "wandb": None,
        }
    )
    assert config.model.optimization_dtype == config.model.reduce_dtype == "float32"
    (args.output / "config.json").write_text(config.model_dump_json(indent=2))
    train(config)
    rows = [row for row in get_monitor().history if "entropy/all/mean" in row]
    assert len(rows) == 1
    metrics = rows[0]
    for name in ("loss/mean", "loss/median", "entropy/all/mean", "entropy/all/std", "mismatch_kl/all/mean"):
        assert math.isfinite(metrics[name]), (name, metrics.get(name))
    # Existing monitor filtering removes per-token medians on both sources.
    assert not any(key.startswith(("entropy/", "mismatch_kl/")) and key.endswith("/median") for key in metrics)
    checkpoint = args.output / "runtime" / "weights" / "step_1"
    assert (checkpoint / "STABLE").is_file()
    initial, final = _weights(seed), _weights(checkpoint)
    assert initial.keys() == final.keys()
    changed = [key for key in initial if not torch.equal(initial[key].to(final[key].dtype), final[key])]
    assert changed, "the actual trainer must perform a nonzero update"
    comparison = None
    if args.reference:
        frozen = json.loads(args.reference.read_text())
        assert frozen["harness_sha256"] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        expected = _weights(Path(frozen["checkpoint"]))
        assert final.keys() == expected.keys()
        assert all(torch.equal(final[key], expected[key]) for key in final), (
            "trainer weights changed with instrumentation"
        )
        assert metrics.keys() == frozen["metrics"].keys()
        for key, value in metrics.items():
            target = frozen["metrics"][key]
            if isinstance(value, float) and math.isnan(value):
                assert math.isnan(target), key
            else:
                assert math.isclose(value, target, rel_tol=1e-12, abs_tol=1e-12), (key, value, target)
        comparison = {"exact_weights": True, "metrics_match": True}
    receipt = {
        "comparison_scope": "One actual trainer update; exact exported BF16 weights, preserved filtered tensor telemetry. Optimizer masters/reductions remain FP32; this export is not a full FP32 optimizer-state comparison.",
        "checkpoint": str(checkpoint.resolve()),
        "metrics": metrics,
        "changed_parameters": changed,
        "source_files": {
            key: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for key, path in sources.items()
        },
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "comparison": comparison,
    }
    (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
