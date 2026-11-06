import pytest

from prime_rl.utils.pathing import path_exists_fresh, validate_output_dir


def test_path_exists_fresh_missing_and_published_paths(tmp_path):
    path = tmp_path / "step_1" / "rank_0.bin"
    assert not path_exists_fresh(path)
    path.parent.mkdir()
    assert not path_exists_fresh(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(b"complete batch")
    temporary.rename(path)
    assert path_exists_fresh(path)
    assert path.read_bytes() == b"complete batch"
    path.unlink()
    assert not path_exists_fresh(path)


def test_nonexistent_dir_passes(tmp_path):
    output_dir = tmp_path / "does_not_exist"
    validate_output_dir(output_dir, resuming=False, clean=False)


def test_empty_dir_passes(tmp_path):
    output_dir = tmp_path / "empty"
    output_dir.mkdir()
    validate_output_dir(output_dir, resuming=False, clean=False)


def test_dir_with_only_logs_passes(tmp_path):
    output_dir = tmp_path / "has_logs"
    output_dir.mkdir()
    (output_dir / "logs").mkdir()
    (output_dir / "logs" / "trainer.log").touch()
    validate_output_dir(output_dir, resuming=False, clean=False)


def test_dir_with_checkpoints_raises(tmp_path):
    output_dir = tmp_path / "has_ckpt"
    output_dir.mkdir()
    (output_dir / "checkpoints").mkdir()
    (output_dir / "checkpoints" / "step_0").mkdir()
    with pytest.raises(FileExistsError, match="already contains checkpoints"):
        validate_output_dir(output_dir, resuming=False, clean=False)


def test_dir_with_checkpoints_passes_when_resuming(tmp_path):
    output_dir = tmp_path / "has_ckpt"
    output_dir.mkdir()
    (output_dir / "checkpoints").mkdir()
    (output_dir / "checkpoints" / "step_0").mkdir()
    validate_output_dir(output_dir, resuming=True, clean=False)


def test_dir_with_checkpoints_cleaned_when_flag_set(tmp_path):
    output_dir = tmp_path / "has_ckpt"
    output_dir.mkdir()
    (output_dir / "checkpoints").mkdir()
    (output_dir / "checkpoints" / "step_0").mkdir()
    (output_dir / "logs").mkdir()

    validate_output_dir(output_dir, resuming=False, clean=True)

    assert not output_dir.exists()


def test_clean_on_nonexistent_dir_is_noop(tmp_path):
    output_dir = tmp_path / "does_not_exist"
    validate_output_dir(output_dir, resuming=False, clean=True)
    assert not output_dir.exists()
