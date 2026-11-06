"""Audit stored real rollouts, candidate alignment and per-token diagnostics."""

import argparse
import json
from pathlib import Path

import msgspec
import numpy as np
from rg_mix_env import load_environment

from prime_rl.transport.types import TrainingBatch


def audit(root, report):
    env = load_environment(dataset_path="/data/rg-dataset", num_train_examples=10000, num_eval_examples=256)
    results = {"output": str(root), "batches": [], "token_diagnostics": [], "rescored": 0, "examples": []}
    for file in sorted((root / "run_default/rollouts").glob("step_*/train_rollouts.bin")):
        batch = msgspec.msgpack.decode(file.read_bytes(), type=TrainingBatch)
        chosen_in_head = tokens = 0
        for s in batch.examples:
            n = len(s.token_ids)
            assert all(len(v) == n for v in [s.mask, s.logprobs, s.advantages, s.temperatures])
            mask = np.asarray(s.mask)
            assert np.isfinite(np.asarray(s.logprobs)[mask]).all()
            assert np.isfinite(np.asarray(s.advantages)[mask]).all()
            assert (np.asarray(s.temperatures)[mask] == 1).all()
            tokens += int(mask.sum())
            if s.sampler_topk_ids is not None:
                ids = np.frombuffer(s.sampler_topk_ids.data, dtype=np.int32).reshape(s.sampler_topk_ids.shape)[mask]
                lp = np.frombuffer(s.sampler_topk_logprobs.data, dtype=np.float32).reshape(
                    s.sampler_topk_logprobs.shape
                )[mask]
                assert np.isfinite(lp).all() and (ids >= 0).all() and (ids < 151936).all()
                assert (np.exp(lp).sum(-1) <= 1.0001).all()
                matches = ids == np.asarray(s.token_ids)[mask, None]
                i, j = np.nonzero(matches)
                np.testing.assert_allclose(lp[i, j], np.asarray(s.logprobs)[mask][i], atol=1e-6, rtol=1e-6)
                chosen_in_head += len(i)
        results["batches"].append(
            {
                "step": batch.step,
                "samples": len(batch.examples),
                "sampled_tokens": tokens,
                "chosen_in_topk": chosen_in_head,
            }
        )
    for directory in sorted((root / "run_default/token_exports").glob("step_*")):
        values = {k: [] for k in ("trainer_logprobs", "inference_logprobs", "mismatch_kl", "importance_ratio")}
        for file in directory.glob("rank_*.jsonl"):
            for line in file.open():
                r = json.loads(line)
                m = np.asarray(r["loss_mask"])
                for k in values:
                    a = np.asarray(r[k], dtype=float)[m]
                    assert np.isfinite(a).all(), (file, k)
                    values[k].extend(a.tolist())
        results["token_diagnostics"].append(
            {
                "step": directory.name,
                **{
                    k: {"mean": float(np.mean(v)), "max": float(np.max(v)), "p99": float(np.quantile(v, 0.99))}
                    for k, v in values.items()
                },
            }
        )
    for file in sorted((root / "run_default/rollouts").glob("step_*/train_rollouts.jsonl")):
        for line in file.open():
            r = json.loads(line)
            if r["errors"]:
                continue
            completion = [n["message"] for n in r["nodes"] if n["message"]["role"] == "assistant"]
            if not completion:
                continue
            answer = int(r["task"]["answer"])
            parsed = str(env.parser.parse_answer(completion)).strip()
            reward = env.score_candidate(answer, parsed)
            assert abs(reward - r["rewards"]["reward"]) < 1e-9, (answer, reward, r["rewards"])
            results["rescored"] += 1
            variant = env._entry_map[answer][0]
            if variant not in {x["variant"] for x in results["examples"]}:
                results["examples"].append(
                    {
                        "variant": variant,
                        "answer_index": answer,
                        "reward": reward,
                        "question": r["task"]["prompt"],
                        "completion": completion,
                        "parsed_answer": parsed,
                    }
                )
    report.write_text(json.dumps(results, indent=2))
    print(json.dumps({k: v for k, v in results.items() if k != "examples"}))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("report", type=Path)
    a = p.parse_args()
    audit(a.root, a.report)
