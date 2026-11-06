"""Frozen RG package versus native taskset on the exact saved research dataset."""

import argparse
import asyncio
import copy
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
from pathlib import Path

import rg_mix_env
import rg_mix_shared
import verifiers.v1 as vf
from rg_mix_v1.scoring import format_reward, parse_answer, response_for_scoring
from rg_mix_v1.taskset import RGMixConfig, RGMixData, RGMixTaskset
from verifiers.parsers.xml_parser import XMLParser


def file_receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def load_frozen(path):
    spec = importlib.util.spec_from_file_location("frozen_rg_mix_env", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parser_cases():
    return [
        [],
        [{"role": "user", "content": "<answer>ignore</answer>"}],
        [{"role": "assistant", "content": "no tags"}],
        [{"role": "assistant", "content": "<answer></answer>"}],
        [{"role": "assistant", "content": "<answer>  \n </answer>"}],
        [{"role": "assistant", "content": "<answer>partial"}],
        [{"role": "assistant", "content": "partial</answer>"}],
        [{"role": "assistant", "content": "<answer>first</answer><answer>last</answer>"}],
        [{"role": "assistant", "content": " <answer> x \n </answer> "}],
        [{"role": "assistant", "content": "outside<answer>x</answer>outside"}],
        [{"role": "assistant", "content": "<answer>old</answer>"}, {"role": "assistant", "content": "none"}],
        [
            {"role": "assistant", "content": "<answer>old</answer>"},
            {"role": "assistant", "content": "<answer>new</answer>"},
        ],
        [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "<answer>"},
                    {"type": "image_url", "image_url": {}},
                    {"type": "text", "text": "x</answer>"},
                ],
            }
        ],
        [vf.AssistantMessage(content="<answer>typed</answer>")],
        [{"role": "assistant", "content": None}],
        "<answer>first</answer><answer>last</answer>",
        "missing",
        "<answer></answer>",
    ]


async def run(args):
    assert not args.output.exists()
    assert importlib.metadata.version("reasoning-gym") == "0.1.25"
    assert importlib.metadata.version("verifiers") == "0.2.2.dev17"
    package_files = {
        "legacy": inspect.getfile(rg_mix_env),
        "shared": inspect.getfile(rg_mix_shared),
        "native": inspect.getfile(RGMixTaskset),
        "parser": inspect.getfile(parse_answer),
    }
    assert all(Path(path).resolve().is_relative_to(args.source_root.resolve()) for path in package_files.values())
    frozen = load_frozen(args.frozen_package / "rg_mix_env.py")
    metadata = json.loads((args.dataset / "metadata.json").read_text())
    assert metadata["num_train"] == 10000 and metadata["num_test"] == 256 and metadata["seed"] == 42
    kwargs = dict(dataset_path=str(args.dataset), num_train_examples=10000, num_eval_examples=256, seed=42)
    original = frozen.load_environment(**kwargs)
    retained = rg_mix_env.load_environment(**kwargs)
    assert frozen.TASK_VARIANTS == rg_mix_shared.TASK_VARIANTS
    assert frozen.DEFAULT_SYSTEM_PROMPT == rg_mix_shared.DEFAULT_SYSTEM_PROMPT
    assert original.rubric.weights == retained.rubric.weights == [1.0, 0.0]
    assert not original.rubric._get_group_reward_funcs()
    original_reward, original_format = original.rubric.funcs
    retained_reward, retained_format = retained.rubric.funcs
    selected = {}
    split_rows = {}
    for split in ("train", "eval"):
        native = RGMixTaskset(RGMixConfig(**kwargs, split=split)).load()
        old_rows = original.get_dataset() if split == "train" else original.get_eval_dataset()
        current_rows = retained.get_dataset() if split == "train" else retained.get_eval_dataset()
        assert len(native) == len(old_rows) == len(current_rows) == (10000 if split == "train" else 256)
        for row_index, (task, old, current) in enumerate(zip(native, old_rows, current_rows, strict=True)):
            data = task.data
            assert old == current
            assert data.idx == old["example_id"] == row_index
            assert data.answer == old["answer"]
            assert data.variant_id == old["info"]["variant_id"]
            assert old["prompt"] == [
                {"role": "system", "content": data.system_prompt},
                {"role": "user", "content": data.prompt},
            ]
            assert data.scorer_entry == metadata["entries_cache"][data.answer]
            payload = data.model_dump(mode="json")
            assert "value_function_prompt" not in payload
            assert RGMixData.model_validate(payload) == data
            selected.setdefault(data.variant_id, task)
        split_rows[split] = {
            "rows": len(native),
            "first_idx": native[0].data.idx,
            "first_global_answer": native[0].data.answer,
            "last_global_answer": native[-1].data.answer,
        }
    assert split_rows["eval"]["first_idx"] == 0 and split_rows["eval"]["first_global_answer"] == "10000"
    assert set(selected) == {variant["id"] for variant in frozen.TASK_VARIANTS}
    old_parser = XMLParser(fields=["answer"])
    parsing = []
    for index, completion in enumerate(parser_cases()):
        expected = old_parser.parse_answer(completion)
        assert parse_answer(completion) == expected
        assert response_for_scoring(completion) == str(expected).strip()
        expected_format = old_parser.get_format_reward_func()(completion)
        assert format_reward(completion) == expected_format
        parsing.append({"case": index, "extracted": expected, "format_metric": expected_format})
    scoring = []
    scorer_sources = {}
    partial_credit_seen = False
    for variant_id, task in selected.items():
        entry_before = copy.deepcopy(task.data.scorer_entry)
        old_entry_before = copy.deepcopy(original._entries_cache[int(task.data.answer)])
        answer = str(task.data.scorer_entry["answer"])
        contents = [
            f"<answer>{answer}</answer>",
            "<answer>wrong</answer>",
            answer,
            "<answer></answer>",
            "<answer>partial",
            f"<answer> \n {answer} \n </answer>",
            f"<answer>{answer}</answer><answer>wrong</answer>",
            f"<answer>wrong</answer><answer>{answer}</answer>",
            f"<answer>noise {answer} noise</answer>",
            "<answer>A=0</answer>",
        ]
        expected_by_case = {}
        for order in (list(range(len(contents))), list(reversed(range(len(contents))))):
            for index in order:
                completion = [{"role": "assistant", "content": contents[index]}]
                expected = await original_reward(completion=completion, answer=task.data.answer)
                assert await retained_reward(completion=completion, answer=task.data.answer) == expected
                expected_format = original_format(completion)
                assert retained_format(completion) == expected_format
                trace = vf.Trace(
                    task=vf.TraceTask(type="RGMixTask", data=task.data),
                    nodes=[vf.MessageNode(message=vf.AssistantMessage(content=contents[index]), sampled=True)],
                )
                await task.score(trace)
                assert trace.reward == expected
                assert trace.metrics == {"check_answer_reward_func": expected, "format_reward_func": expected_format}
                assert trace.rewards["format_reward_func"] == 0
                if index in expected_by_case:
                    assert expected_by_case[index] == expected
                expected_by_case[index] = expected
                partial_credit_seen |= 0 < expected < 1
        assert expected_by_case[0] == 1.0, (variant_id, "saved canonical answer positive control", expected_by_case[0])
        assert task.data.scorer_entry == entry_before
        assert original._entries_cache[int(task.data.answer)] == old_entry_before
        scorer_sources[variant_id] = file_receipt(inspect.getfile(type(task._scorer).score_answer))
        scoring.append(
            {
                "variant": variant_id,
                "global_answer": task.data.answer,
                "responses": len(contents),
                "orders": 2,
                "rewards": expected_by_case,
                "entry_unchanged": True,
            }
        )
    assert partial_credit_seen, "partial-credit preservation requires a positive control"
    receipt = {
        "split_rows": split_rows,
        "parser_cases": parsing,
        "scoring_cases": scoring,
        "partial_credit_positive_control": partial_credit_seen,
        "source_files": {key: file_receipt(path) for key, path in package_files.items()},
        "frozen_source": file_receipt(args.frozen_package / "rg_mix_env.py"),
        "actual_xml_parser": file_receipt(inspect.getfile(XMLParser)),
        "scorer_sources": scorer_sources,
        "dataset_metadata": file_receipt(args.dataset / "metadata.json"),
        "harness": file_receipt(__file__),
        "scope": "Exact saved10256-row research artifact; all train/eval selection, prompt bytes, private payload and parser checks. Actual frozen/retained legacy reward functions and native Task.score on5real variants, shared-vs-perTask scorer repeat/reversed order. No generation, model, GPU or policy-quality claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        json.dumps(
            {
                "rows": sum(x["rows"] for x in split_rows.values()),
                "parser_cases": len(parsing),
                "scorer_calls": sum(x["responses"] * x["orders"] for x in scoring),
                "partial_credit": partial_credit_seen,
            }
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--frozen-package", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
