import torch

from prime_rl.trainer.utils import TensorSummaries


def test_streaming_summaries_preserve_token_counts_without_retaining_inputs():
    summaries = TensorSummaries(device="cpu")
    values = torch.tensor([0.0, 1.0, 1.0], requires_grad=True)
    summaries.add("fraction", values[:1])
    summaries.add("fraction", values[1:])
    values.detach().fill_(20)
    torch.testing.assert_close(summaries.summaries["fraction"], torch.tensor([3.0, 2.0, 0.0, 1.0], dtype=torch.float64))
    assert not summaries.summaries["fraction"].requires_grad
    for _ in range(10):
        summaries.add("fraction", torch.ones(1000))
    assert summaries.summaries["fraction"].numel() == 4
    assert summaries.summaries["fraction"][0].item() == 10003
    assert summaries.summaries["fraction"][1].item() == 10002


def test_streaming_summaries_handle_empty_populations_and_large_values():
    summaries = TensorSummaries(device="cpu")
    summaries.add("empty", torch.empty(0))
    torch.testing.assert_close(
        summaries.summaries["empty"], torch.tensor([0.0, 0.0, float("inf"), -float("inf")], dtype=torch.float64)
    )
    summaries.add("large", torch.tensor([0.0, 1e30], dtype=torch.float64))
    summaries.add("large", torch.empty(0))
    torch.testing.assert_close(summaries.summaries["large"], torch.tensor([2.0, 1e30, 0.0, 1e30], dtype=torch.float64))


def test_streaming_summaries_keep_nonfinite_values_visible():
    summaries = TensorSummaries(device="cpu")
    values = torch.tensor([0.0, float("nan"), float("inf"), -float("inf")])
    summaries.add("raw", values)
    summaries.add("nonfinite", ~values.isfinite())
    assert torch.isnan(summaries.summaries["raw"][1])
    torch.testing.assert_close(
        summaries.summaries["nonfinite"], torch.tensor([4.0, 3.0, 0.0, 1.0], dtype=torch.float64)
    )
    for sign in (-1, 1):
        summaries.add(str(sign), torch.tensor([0.0, sign * float("inf")]))
        assert summaries.summaries[str(sign)][1] == sign * float("inf")
