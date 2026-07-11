import pytest

from prime_rl.value.update_schedule import updates_for_batch


@pytest.mark.parametrize(
    ("value_version", "warmup_updates", "updates_per_batch", "expected"),
    [
        (0, 0, 4, 4),
        (0, 50, 4, 1),
        (49, 50, 4, 1),
        (50, 50, 4, 4),
        (51, 50, 2, 2),
    ],
)
def test_updates_for_batch(
    value_version: int,
    warmup_updates: int,
    updates_per_batch: int,
    expected: int,
) -> None:
    assert (
        updates_for_batch(
            value_version=value_version,
            warmup_updates=warmup_updates,
            updates_per_batch=updates_per_batch,
        )
        == expected
    )
