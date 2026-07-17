import pytest

from prime_rl.value.update_schedule import choose_next_operation


@pytest.mark.parametrize(
    ("has_inference", "has_training", "last_operation", "expected"),
    [
        (False, False, None, None),
        (True, False, None, "infer"),
        (False, True, None, "train"),
        (True, True, None, "infer"),
        (True, True, "infer", "train"),
        (True, True, "train", "infer"),
    ],
)
def test_choose_next_operation(has_inference, has_training, last_operation, expected):
    assert (
        choose_next_operation(
            has_inference=has_inference,
            has_training=has_training,
            last_operation=last_operation,
        )
        == expected
    )
