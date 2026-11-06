"""Evaluate a frozen split on an exclusive, already-started inference endpoint."""

import argparse
import asyncio
import hashlib
import json
import urllib.request
from collections import defaultdict
from pathlib import Path

from prepare import MODEL, MODEL_REVISION
from rg_mix_env import load_environment
from transformers import AutoTokenizer
from verifiers.clients import resolve_client
from verifiers.types import ClientConfig


def request(base, path, body=None):
    payload = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        return response.read()


async def evaluate(args):
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation evidence: {args.output}")
    manifest = json.loads(args.splits.read_text())
    validation = args.split == "validation"
    dataset_path = manifest["validation_path"] if validation else manifest["train_path"]
    env = load_environment(
        dataset_path=dataset_path, num_train_examples=1 if validation else 10000, num_eval_examples=256
    )
    env.set_kwargs(max_seq_len=8192)
    dataset = env.get_eval_dataset()
    expected = manifest[args.split]
    assert [r["answer"] for r in dataset] == expected["answer_ids"]
    hashes = [
        hashlib.sha256(next(m["content"] for m in r["prompt"] if m["role"] == "user").encode()).hexdigest()
        for r in dataset
    ]
    assert hashes == expected["prompt_sha256"]
    if args.preflight:
        print(f"Verified {len(dataset)} frozen {args.split} prompts and answer IDs")
        return
    AutoTokenizer.from_pretrained(MODEL)
    client = resolve_client(
        ClientConfig(
            client_type="renderer",
            renderer_model_name=MODEL,
            renderer_config={"name": "qwen3"},
            renderer_pool_size=4,
            api_base_url=args.endpoint + "/v1",
            api_key_var="VLLM_API_KEY",
        )
    )
    if not (args.checkpoint / "STABLE").exists():
        raise ValueError("A stable checkpoint is required")
    request(args.endpoint, "/pause", {})
    try:
        request(args.endpoint, "/update_weights", {"weight_dir": str(args.checkpoint)})
    finally:
        request(args.endpoint, "/resume", {})
    args.output.mkdir(parents=True)
    sampling = {
        "temperature": 1.0,
        "top_p": 1.0,
        "extra_body": {
            "top_k": -1,
            "min_p": 0.0,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "repetition_penalty": 1.0,
            "seed": args.seed,
        },
    }
    (args.output / "config.json").write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "model_revision": MODEL_REVISION,
                "split": args.split,
                "split_manifest_sha256": hashlib.sha256(args.splits.read_bytes()).hexdigest(),
                "sampling": sampling,
                "seq_len": 8192,
                "n": 256,
            },
            indent=2,
        )
    )
    semaphore = asyncio.Semaphore(16)
    records = []
    with (args.output / "rollouts.jsonl").open("w", buffering=1) as file:

        async def one(row):
            async with semaphore:
                # Explicit per-prompt seeds keep streams paired across checkpoints/arms.
                settings = {
                    **sampling,
                    "extra_body": {**sampling["extra_body"], "seed": args.seed + int(row["answer"])},
                }
                result = await env.run_rollout(input=dict(row), client=client, model=MODEL, sampling_args=settings)
                index = int(row["answer"])
                record = {
                    "answer_index": index,
                    "variant": env._entry_map[index][0],
                    "prompt": row["prompt"],
                    "reward": result["reward"],
                    "completion": result["completion"],
                    "is_truncated": result.get("is_truncated"),
                    "error": result.get("error"),
                    "token_usage": result.get("token_usage"),
                }
                file.write(json.dumps(record, default=lambda x: x.model_dump(mode="json")) + "\n")
                records.append(record)

        await asyncio.gather(*(one(row) for row in dataset))
    by = defaultdict(list)
    for r in records:
        by[r["variant"]].append(r["reward"])
    summary = {
        "n": len(records),
        "reward": sum(r["reward"] for r in records) / len(records),
        "errors": sum(bool(r["error"]) for r in records),
        "truncation_rate": sum(bool(r["is_truncated"]) for r in records) / len(records),
        "by_variant": {k: {"n": len(v), "reward": sum(v) / len(v)} for k, v in by.items()},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--preflight", action="store_true")
    p.add_argument("--split", choices=["validation", "test"], required=True)
    p.add_argument("--splits", type=Path, default=Path(__file__).parent / "splits.json")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--endpoint", required=True, help="Exclusive evaluation server; never use a live training endpoint")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=4400)
    asyncio.run(evaluate(p.parse_args()))
