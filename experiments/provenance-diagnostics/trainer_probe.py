"""One actual actor update: full FP32 states and telemetry, same-stack old/new.

The test fixture uses the real FakeDataLoader, maps its token IDs into a tiny
vocabulary and stamps literal observation labels. Candidate reuses those exact
saved CPU batches and model files. No generation or engine-label truth claim.
"""

import argparse
import copy
import hashlib
import importlib
import inspect
import json
import math
import shutil
from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from torch.distributed.tensor import DTensor
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

from prime_rl.configs.trainer import TrainerConfig
from prime_rl.trainer.rl.data import FakeDataLoader
from prime_rl.trainer.utils import TensorSummaries
from prime_rl.utils.monitor import get_monitor

trainer_module = importlib.import_module("prime_rl.trainer.rl.train")


def full(tensor):
    return (tensor.full_tensor() if isinstance(tensor, DTensor) else tensor).detach().cpu().clone()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    sources = {
        "trainer": Path(inspect.getfile(inspect.unwrap(trainer_module.train))).resolve(),
        "summaries": Path(inspect.getfile(TensorSummaries)).resolve(),
        "config": Path(inspect.getfile(TrainerConfig)).resolve(),
        "loader": Path(inspect.getfile(FakeDataLoader)).resolve(),
        "optimizer": Path(inspect.getfile(inspect.unwrap(trainer_module.setup_optimizer))).resolve(),
        "loss": Path(inspect.getfile(inspect.unwrap(trainer_module.compute_loss))).resolve(),
    }
    sources["metrics"] = sources["trainer"].with_name("metrics.py")
    assert all(p.is_relative_to(args.source_root.resolve()) for p in sources.values()), sources
    hashes = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in sources.items()}
    harness_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.output.mkdir(parents=True)
    seed = args.output / "seed"
    frozen = json.loads(args.reference.read_text()) if args.reference else None
    if frozen:
        assert frozen["harness_sha256"] == harness_hash
        seed.mkdir()
        for name in frozen["fixture_hashes"]:
            shutil.copyfile(Path(frozen["fixture_dir"]) / name, seed / name)
        batches = torch.load(Path(frozen["batch_file"]), map_location="cpu", weights_only=False)
    else:
        torch.manual_seed(713)
        model = LlamaForCausalLM(
            LlamaConfig(
                vocab_size=64,
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
        batches = None
    fixture_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in seed.iterdir() if p.is_file()}
    if frozen:
        assert fixture_hashes == frozen["fixture_hashes"]

    original_get_batch = FakeDataLoader.get_batch
    captured_batches = []

    def get_batch(self):
        nonlocal batches
        if batches is None:
            batches = original_get_batch(self)
            pattern = torch.tensor([-1, 0, 1, 2**63 - 1], dtype=torch.int64)
            for batch in batches:
                batch["input_ids"] = batch["input_ids"].remainder(64)
                batch["token_policy_versions"] = pattern[torch.arange(batch["input_ids"].numel()) % 4].reshape_as(
                    batch["input_ids"]
                )
        captured_batches.append(copy.deepcopy(batches))
        return copy.deepcopy(batches)

    original_setup_optimizer = trainer_module.setup_optimizer
    observed = []

    def setup_optimizer(*a, **kw):
        optimizer = original_setup_optimizer(*a, **kw)
        named = list(a[1])

        def before_step(opt, _args, _kwargs):
            assert not observed, "exactly one real optimizer update"
            initial = {name: full(p) for name, p in named}
            gradients = {name: full(p.grad) for name, p in named if p.grad is not None}
            assert gradients and all(t.dtype == torch.float32 for t in (*initial.values(), *gradients.values()))
            assert all(torch.isfinite(t).all() for t in (*initial.values(), *gradients.values()))
            assert any(torch.count_nonzero(t) > 0 for t in gradients.values())
            observed.append({"initial": initial, "gradients": gradients})

        def after_step(opt, _args, _kwargs):
            observed[0]["updated"] = {name: full(p) for name, p in named}
            moments = {}
            for name, parameter in named:
                for key, value in opt.state.get(parameter, {}).items():
                    if isinstance(value, torch.Tensor):
                        moments[f"{name}/{key}"] = full(value)
            assert moments and all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in moments.values())
            observed[0]["moments"] = moments

        optimizer.register_step_pre_hook(before_step)
        optimizer.register_step_post_hook(after_step)
        return optimizer

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
    if frozen:
        previous_config = json.loads(args.reference.with_name("config.json").read_text())
        current_config = config.model_dump(mode="json")
        for item in (previous_config, current_config):
            item.pop("output_dir")
            item["model"].pop("name")
            item["tokenizer"].pop("name")
        assert current_config == previous_config, "only verified output/model/tokenizer paths may differ"
    FakeDataLoader.get_batch = get_batch
    trainer_module.setup_optimizer = setup_optimizer
    batch_file = args.output / "batches.pt"
    state_file = args.output / "full-state.pt"
    # Creating the baseline fixture consumes RNG; copying it must not give
    # candidate model setup a different random stream.
    torch.manual_seed(713)
    try:
        trainer_module.train(config)
    finally:
        FakeDataLoader.get_batch = original_get_batch
        trainer_module.setup_optimizer = original_setup_optimizer
        if captured_batches:
            torch.save(captured_batches[0], batch_file)
        if observed:
            torch.save(observed[0], state_file)
    assert len(observed) == len(captured_batches) == 1
    assert any(not torch.equal(t, observed[0]["updated"][k]) for k, t in observed[0]["initial"].items())
    assert fixture_hashes == {name: hashlib.sha256((seed / name).read_bytes()).hexdigest() for name in fixture_hashes}
    rows = [r for r in get_monitor().history if "entropy/all/mean" in r]
    assert len(rows) == 1
    metrics = rows[0]
    for key in ["loss/mean", "entropy/all/mean", "mismatch_kl/all/mean"]:
        assert math.isfinite(metrics[key])
    provenance = {k: v for k, v in metrics.items() if k.startswith(("observed_policy_version/", "sampled_weight_age/"))}
    original_metrics = {k: v for k, v in metrics.items() if k not in provenance}
    assert all(batch["rl_weights"] is None and batch["ref_kl_weights"] is None for batch in captured_batches[0]), (
        "This literal fake-cohort oracle requires the existing absent-component fallback to loss_mask"
    )
    table = [
        int(version)
        for batch in captured_batches[0]
        for version, eligible in zip(batch["token_policy_versions"].reshape(-1), batch["loss_mask"].reshape(-1))
        if bool(eligible)
    ]
    known, unknown, future, valid = (
        sum(v >= 0 for v in table),
        table.count(-1),
        sum(v > 0 for v in table),
        table.count(0),
    )
    expected = {
        "observed_policy_version/eligible_count": len(table),
        "observed_policy_version/known_count": known,
        "observed_policy_version/known_fraction": known / len(table),
        "observed_policy_version/unknown_count": unknown,
        "observed_policy_version/unknown_fraction": unknown / len(table),
        "observed_policy_version/future_count": future,
        "observed_policy_version/future_fraction": future / len(table),
        "sampled_weight_age/count": valid,
        "sampled_weight_age/mean": 0.0,
        "sampled_weight_age/max": 0.0,
    }
    if frozen:
        assert provenance == expected
        previous = torch.load(frozen["state_file"], map_location="cpu", weights_only=False)
        for group in ["initial", "gradients", "updated", "moments"]:
            assert observed[0][group].keys() == previous[group].keys()
            for name, tensor in observed[0][group].items():
                assert torch.equal(tensor, previous[group][name]), (group, name)
        assert original_metrics.keys() == frozen["original_metrics"].keys()
        for key, value in original_metrics.items():
            target = frozen["original_metrics"][key]
            assert (isinstance(value, float) and math.isnan(value) and math.isnan(target)) or value == target, key
    else:
        assert not provenance, "baseline must not already include the candidate diagnostics"
    assert hashes == {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in sources.items()}
    (args.output / "receipt.json").write_text(
        json.dumps(
            {
                "status": "passed",
                "pretrain_torch_seed": 713,
                "harness_sha256": harness_hash,
                "source_hashes": hashes,
                "fixture_dir": str(seed.resolve()),
                "fixture_hashes": fixture_hashes,
                "batch_file": str(batch_file.resolve()),
                "state_file": str(state_file.resolve()),
                "state_tensor_counts": {k: len(v) for k, v in observed[0].items()},
                "original_metrics": original_metrics,
                "provenance": provenance,
                "literal_expected": expected,
                "comparison": "exact full FP32 parameters/gradients/Adam moments and original metrics"
                if frozen
                else "baseline",
                "scope": "Actual one-rank trainer, one update, exact saved tiny model and CPU batch. Literal carrier fixture; no engine/critic-publication/CP-forward proof.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
