import pytest
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


def test_masked_centered_moments_preserve_small_variance_at_large_offsets():
    summaries = TensorSummaries(device="cpu", include_std=True)
    summaries.add("offset", torch.tensor([1e12 + 1, float("nan")], dtype=torch.float64), torch.tensor([True, False]))
    summaries.add("offset", torch.tensor([1e12 - 1, 1e12 + 3], dtype=torch.float64))
    summaries.add("offset", torch.tensor([float("inf"), -float("inf")]), torch.tensor([False, False]))
    expected = torch.tensor([3, 3e12 + 3, 1e12 - 1, 1e12 + 3, 8], dtype=torch.float64)
    torch.testing.assert_close(summaries.summaries["offset"], expected, rtol=0, atol=0)
    assert not summaries.summaries["offset"].requires_grad
    summaries.add("empty", torch.tensor([float("nan")]), torch.tensor([False]))
    torch.testing.assert_close(
        summaries.summaries["empty"], torch.tensor([0, 0, float("inf"), -float("inf"), 0], dtype=torch.float64)
    )
    summaries.add("observed_nonfinite", torch.tensor([float("inf"), 1.0]))
    assert torch.isnan(summaries.summaries["observed_nonfinite"][4])


def test_metric_masks_reject_shape_or_dtype_mismatch():
    summaries = TensorSummaries(device="cpu")
    for mask in (torch.tensor([1, 0]), torch.tensor([True])):
        with pytest.raises(ValueError, match="boolean and match"):
            summaries.add("bad", torch.tensor([1.0, 2.0]), mask)
