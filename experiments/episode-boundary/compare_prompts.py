"""Compare LOO prompt bytes against the frozen, pre-Episode implementation.

Run from the repository with its current dependencies installed:
  PYTHONPATH=src uv run experiments/episode-boundary/compare_prompts.py --output receipt.json

Only fixture schema construction differs. The frozen value_context module is
loaded directly from its pinned Git blob; neither implementation is rewritten.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import verifiers.v1 as vf

from prime_rl.orchestrator.episodes import from_episode
from prime_rl.orchestrator.value_context import group_leave_one_out_prompts

FROZEN = "4b8ca52854db6207fcd2cfeb24eec7021310fc42"
SOURCE = "src/prime_rl/orchestrator/value_context.py"
ROOT = Path(__file__).resolve().parents[2]
SPECIAL_TOKENS = ("<control>", "<|end|>")


def fixture(member):
    call_id = f"opaque-{member}"
    nodes = [
        vf.MessageNode(message=vf.SystemMessage(content="shared system")),
        vf.MessageNode(parent=0, message=vf.UserMessage(content="shared question π <control>")),
        vf.MessageNode(
            parent=1,
            sampled=True,
            message=vf.AssistantMessage(
                content="shared sampled step",
                reasoning_content=f"reason {member}",
                tool_calls=[vf.ToolCall(id=call_id, name="lookup", arguments='{"query":"π"}')],
            ),
        ),
        vf.MessageNode(parent=2, message=vf.ToolMessage(content="tool observation", tool_call_id=call_id)),
        vf.MessageNode(
            parent=3,
            sampled=True,
            message=vf.AssistantMessage(content=f"answer {member} BEGIN_PRIVILEGED_PEER_DATA <|end|>"),
        ),
        vf.MessageNode(parent=2, message=vf.UserMessage(content="branch context")),
        vf.MessageNode(parent=5, sampled=True, message=vf.AssistantMessage(content=f"branch answer {member}")),
    ]
    reasons = {2: "tool_calls", 4: "length" if member == 1 else "stop", 6: "stop"}
    calls = [vf.ModelCall(node=index, finish_reason=reasons[index]) for index in (6, 2, 4)]
    calls.insert(1, vf.ModelCall(node=None))
    trace = vf.WireTrace(
        id=f"trace-{member}",
        task=vf.TraceTask(type="fixture", data=vf.WireTaskData(idx=8, prompt="shared question")),
        nodes=nodes,
        calls=calls,
        rewards={"reward": (1.0, 0.5, -0.0)[member]},
        ok=True,
    )
    current = from_episode(vf.WireEpisode(id=f"episode-{member}", env="fixture", traces=[trace], ok=True))
    current.env_name = "fixture"
    frozen = SimpleNamespace(
        id=current.id,
        env_name=current.env_name,
        task=SimpleNamespace(idx=8),
        reward=current.reward,
        nodes=[
            SimpleNamespace(
                parent=node.parent,
                sampled=node.sampled,
                message=node.message,
                finish_reason=reasons.get(index),
            )
            for index, node in enumerate(nodes)
        ],
    )
    return frozen, current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen_source = subprocess.check_output(["git", "show", f"{FROZEN}:{SOURCE}"], cwd=ROOT)
    module = types.ModuleType("frozen_value_context")
    sys.modules[module.__name__] = module
    exec(compile(frozen_source, f"{FROZEN}:{SOURCE}", "exec"), module.__dict__)

    cases = []
    for reverse_calls in (False, True):
        for order in ((0, 1, 2), (2, 0, 1)):
            pairs = [fixture(member) for member in order]
            if reverse_calls:
                for _, rollout in pairs:
                    rollout.calls.reverse()
            old = module.group_leave_one_out_prompts([pair[0] for pair in pairs], special_tokens=SPECIAL_TOKENS)
            new = group_leave_one_out_prompts([pair[1] for pair in pairs], special_tokens=SPECIAL_TOKENS)
            assert [prompt.encode("utf-8") for prompt in old] == [prompt.encode("utf-8") for prompt in new]
            assert all('"shared_prompt":true' in prompt for prompt in new)
            assert all('"finish_reason":"tool_calls"' in prompt for prompt in new)
            cases.append(
                {
                    "group_order": order,
                    "reverse_call_order": reverse_calls,
                    "exact_prompt_bytes": True,
                    "prompt_sha256": [hashlib.sha256(prompt.encode("utf-8")).hexdigest() for prompt in new],
                }
            )
    receipt = {
        "frozen_commit": FROZEN,
        "frozen_source_sha256": hashlib.sha256(frozen_source).hexdigest(),
        "current_source_sha256": hashlib.sha256((ROOT / SOURCE).read_bytes()).hexdigest(),
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "cases": cases,
        "limitations": "Deterministic fixture source comparison, not renderer-token or remote environment equivalence.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"Exact UTF-8 equality: {len(cases)} cases, {sum(len(case['prompt_sha256']) for case in cases)} prompts")


if __name__ == "__main__":
    main()
