def updates_for_batch(
    *,
    value_version: int,
    warmup_updates: int,
    updates_per_batch: int,
) -> int:
    return 1 if value_version < warmup_updates else updates_per_batch
