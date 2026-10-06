import errno
import threading
from pathlib import Path

import pytest

from scripts.atomic_publish import publish_directory_noreplace


def _source(root, name, payload):
    path = root / name
    path.mkdir()
    (path / "payload.txt").write_text(payload)
    return path


def test_atomic_publish_moves_complete_directory_without_replacement(tmp_path):
    source = _source(tmp_path, "source", "complete")
    destination = tmp_path / "published"
    publish_directory_noreplace(source, destination)
    assert not source.exists()
    assert (destination / "payload.txt").read_text() == "complete"


@pytest.mark.parametrize("destination_kind", ["empty", "nonempty", "symlink"])
def test_atomic_publish_preserves_existing_destination_and_source(tmp_path, destination_kind):
    source = _source(tmp_path, "source", "candidate")
    destination = tmp_path / "published"
    if destination_kind == "symlink":
        existing = _source(tmp_path, "existing", "winner")
        destination.symlink_to(existing, target_is_directory=True)
    else:
        destination.mkdir()
        if destination_kind == "nonempty":
            (destination / "payload.txt").write_text("winner")
    with pytest.raises(FileExistsError):
        publish_directory_noreplace(source, destination)
    assert (source / "payload.txt").read_text() == "candidate"
    if destination_kind == "nonempty":
        assert (destination / "payload.txt").read_text() == "winner"
    elif destination_kind == "symlink":
        assert destination.is_symlink()


def test_atomic_publish_has_exactly_one_winner_under_race(tmp_path):
    sources = [_source(tmp_path, f"source-{index}", str(index)) for index in range(2)]
    destination = tmp_path / "published"
    barrier = threading.Barrier(2)
    outcomes = []

    def publish(source):
        barrier.wait()
        try:
            publish_directory_noreplace(source, destination)
            outcomes.append("published")
        except FileExistsError:
            outcomes.append("exists")

    threads = [threading.Thread(target=publish, args=(source,)) for source in sources]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(outcomes) == ["exists", "published"]
    assert (destination / "payload.txt").read_text() in {"0", "1"}
    assert sum(source.exists() for source in sources) == 1


def test_atomic_publish_propagates_native_failure_without_mutation(tmp_path, monkeypatch):
    import scripts.atomic_publish as publisher

    source = _source(tmp_path, "source", "candidate")
    destination = tmp_path / "published"

    def fail(*_):
        raise OSError(errno.EXDEV, "injected")

    monkeypatch.setattr(publisher, "_native_rename_noreplace", fail)
    with pytest.raises(OSError) as error:
        publish_directory_noreplace(source, destination)
    assert error.value.errno == errno.EXDEV
    assert source.is_dir() and not destination.exists()


@pytest.mark.parametrize("path_kind", ["file", "symlink"])
def test_atomic_publish_rejects_non_directory_source(tmp_path, path_kind):
    source = tmp_path / "source"
    if path_kind == "file":
        source.write_text("payload")
    else:
        target = _source(tmp_path, "target", "payload")
        source.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="real directory"):
        publish_directory_noreplace(source, tmp_path / "published")
