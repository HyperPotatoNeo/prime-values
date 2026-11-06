"""Same-stack kernel equivalence and actual fullgraph capture, not a training tolerance calibration."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from flash_attn import flash_attn_varlen_func as fa2
from flash_attn_interface import flash_attn_varlen_func as fa3

from prime_rl.trainer.models.layers import flash_attn_ops as ops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = Path(ops.__file__).resolve()
    assert source.is_relative_to(args.source_root.resolve()), source
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(7301)
    cu = torch.tensor([0, 5, 13], dtype=torch.int32, device="cuda")
    records, saved = [], {}
    for version, dim, window, deterministic in [
        (2, 64, (-1, -1), False),
        (2, 12, (3, 0), True),
        (3, 64, (-1, -1), False),
        (3, 64, (3, 0), True),
    ]:
        name = f"fa{version}-dim{dim}-det{deterministic}"
        inputs = tuple(torch.randn((13, heads, dim), device="cuda", dtype=torch.bfloat16) for heads in (4, 2, 2))
        dout = torch.randn_like(inputs[0])
        public = fa2 if version == 2 else fa3
        opaque = ops.flash_attn_2_varlen_op if version == 2 else ops.flash_attn_3_varlen_op
        kwargs = dict(causal=True, window_size=window, deterministic=deterministic)

        def run(fn):
            q, k, v = (x.detach().clone().requires_grad_() for x in inputs)
            out = fn(q, k, v, cu, cu, 8, 8, **kwargs)
            if isinstance(out, tuple):
                out = out[0]
            grads = torch.autograd.grad(out, (q, k, v), dout)
            tensors = (out, *grads)
            assert all(torch.isfinite(t).all() for t in tensors)
            assert all(torch.count_nonzero(t) > 0 for t in grads)
            return tuple(t.detach().cpu() for t in tensors), tuple(out.stride())

        reference, reference_stride = run(public)
        repeat, _ = run(public)
        actual, actual_stride = run(opaque)
        saved[name] = dict(public=reference, repeat=repeat, opaque=actual)
        torch.save(saved, args.output / "tensors.pt")
        for expected, repeated, observed in zip(reference, repeat, actual, strict=True):
            torch.testing.assert_close(repeated, expected, rtol=0, atol=0)
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        assert reference_stride == actual_stride
        torch._dynamo.reset()
        before = torch._dynamo.utils.counters["stats"]["unique_graphs"]
        compiled, compiled_stride = run(torch.compile(opaque, fullgraph=True))
        captured = torch._dynamo.utils.counters["stats"]["unique_graphs"] - before
        saved[name]["compiled"] = compiled
        torch.save(saved, args.output / "tensors.pt")
        assert captured > 0, "must capture a real graph, not merely enter a compiled wrapper"
        for expected, observed in zip(reference, compiled, strict=True):
            torch.testing.assert_close(observed, expected, rtol=0, atol=0)
        assert compiled_stride == reference_stride
        records.append(dict(case=name, captured_graphs=captured, stride=reference_stride, exact=True))
    torch.save(saved, args.output / "tensors.pt")
    (args.output / "result.json").write_text(
        json.dumps(
            dict(
                source=str(source),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                torch=torch.__version__,
                cases=records,
                scope="BF16 zero-dropout shared attention only; CP kernels unchanged",
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
