import pytest

from prime_rl.orchestrator.value_context import TokenPrefix


def test_token_prefix_after_bos_round_trips_alignment():
    prefix = TokenPrefix(token_ids=(90, 91), insert_at=1)

    assert prefix.apply([1, 2, 3]) == [1, 90, 91, 2, 3]
    assert prefix.project([0.1, 9.0, 8.0, 0.2, 0.3]) == [0.1, 0.2, 0.3]
    assert prefix.lift([False, True, True], fill=False) == [False, False, False, True, True]
    assert prefix.lift([0.0, 0.4, 0.5], fill=0.0) == [0.0, 0.0, 0.0, 0.4, 0.5]


def test_token_prefix_without_bos_inserts_at_zero():
    prefix = TokenPrefix(token_ids=(90, 91), insert_at=0)

    assert prefix.apply([2, 3]) == [90, 91, 2, 3]
    assert prefix.project([9.0, 8.0, 0.2, 0.3]) == [0.2, 0.3]


def test_token_prefix_rejects_invalid_insertion():
    with pytest.raises(ValueError, match="must be 0 or 1"):
        TokenPrefix(token_ids=(90,), insert_at=2)
