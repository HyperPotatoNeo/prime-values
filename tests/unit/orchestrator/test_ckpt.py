import torch

from prime_rl.configs.orchestrator import CheckpointConfig
from prime_rl.orchestrator.ckpt import CheckpointManager
from prime_rl.orchestrator.types import Progress


def test_orchestrator_checkpoint_round_trips_algorithm_state(tmp_path):
    manager = CheckpointManager(tmp_path, CheckpointConfig())
    progress = Progress(step=3, total_tokens=17)
    algorithm_states = {"train": {"adaptive_tether": {"alpha": 0.2, "rho": 0.4}}}
    manager.save(progress, 3, algorithm_states=algorithm_states)

    restored = Progress()
    loaded_states = manager.load(restored, 3)

    assert restored == progress
    assert loaded_states == algorithm_states


def test_skip_progress_still_restores_algorithm_state(tmp_path):
    manager = CheckpointManager(tmp_path, CheckpointConfig(skip_progress=True))
    algorithm_states = {"train": {"adaptive_tether": {"alpha": 0.2}}}
    manager.save(Progress(step=8), 8, algorithm_states=algorithm_states)

    restored = Progress(step=1)
    assert manager.load(restored, 8) == algorithm_states
    assert restored.step == 1


def test_legacy_progress_only_checkpoint_loads_with_empty_algorithm_state(tmp_path):
    manager = CheckpointManager(tmp_path, CheckpointConfig())
    path = manager.get_ckpt_path(4)
    path.mkdir(parents=True)
    with open(path / "progress.pt", "wb") as handle:
        torch.save({"progress": Progress(step=4, total_samples=9)}, handle)

    restored = Progress()
    assert manager.load(restored, 4) == {}
    assert restored.step == 4
    assert restored.total_samples == 9
