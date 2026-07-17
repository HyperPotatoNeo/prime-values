from prime_rl.utils.logger import get_logger
from prime_rl.value.replay import ValueReplayBuffer
from prime_rl.value.transport import ValueRolloutReceiver


def admit_available_rollouts(
    receiver: ValueRolloutReceiver,
    replay: ValueReplayBuffer,
    *,
    wait_for_first: bool = False,
) -> int:
    """Admit one bounded FIFO slice and return its rollout count."""
    filling = not replay.can_sample
    rollouts = receiver.receive_available(replay.admission_limit, wait_for_first=wait_for_first)
    for rollout in rollouts:
        replay.add(rollout)
        if filling and (len(replay) == 1 or len(replay) % replay.batch_size == 0 or replay.can_sample):
            get_logger().info(f"Value replay filling {len(replay)}/{replay.refill_size} rollouts")
    return len(rollouts)
