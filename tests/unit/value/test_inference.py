import math

import pytest

from prime_rl.value.inference import reassemble_value_outputs


def test_reassemble_value_outputs_restores_original_order():
    result = reassemble_value_outputs([(1, [0.3]), (0, [0.1, 0.2])], [2, 1])

    assert result == [[0.1, 0.2], [0.3]]


@pytest.mark.parametrize(
    ("indexed", "expected_lengths", "match"),
    [
        ([(0, [0.1]), (0, [0.2])], [1], "more than once"),
        ([(0, [0.1])], [1, 1], "failed to materialize"),
        ([(1, [0.1])], [1], "invalid sample index"),
        ([(0, [])], [1], "expected 1"),
        ([(0, [math.nan])], [1], "non-finite"),
    ],
)
def test_reassemble_value_outputs_rejects_invalid_results(indexed, expected_lengths, match):
    with pytest.raises(RuntimeError, match=match):
        reassemble_value_outputs(indexed, expected_lengths)
