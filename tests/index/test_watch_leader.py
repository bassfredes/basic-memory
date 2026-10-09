"""Leader election keeps one local watcher across MCP processes (fork patch)."""

import asyncio
from dataclasses import dataclass

import pytest

from basic_memory.index import watch_leader
from basic_memory.index.watch_leader import LeaderElectedWatch, try_acquire

pytestmark = pytest.mark.skipif(
    watch_leader.fcntl is None and watch_leader.msvcrt is None, reason="no OS file lock"
)


@dataclass
class FakeCoordinator:
    should_watch: bool = True
    starts: int = 0
    stops: int = 0

    async def start(self) -> None:
        self.starts += 1

    async def stop(self) -> None:
        self.stops += 1


def test_lock_is_exclusive_until_released(tmp_path):
    lock = tmp_path / "watch.lock"
    first = try_acquire(lock)
    assert first is not None
    assert try_acquire(lock) is None
    watch_leader.os.close(first)
    second = try_acquire(lock)
    assert second is not None
    watch_leader.os.close(second)


def test_holder_pid_stays_readable(tmp_path):
    lock = tmp_path / "watch.lock"
    fd = try_acquire(lock)
    try:
        # Another reader (cat watch.lock, a follower) must see the PID while it is held.
        assert lock.read_text() == str(watch_leader.os.getpid())
    finally:
        watch_leader.os.close(fd)


def test_lock_is_released_when_the_holder_process_exits(tmp_path):
    import subprocess
    import sys

    lock = tmp_path / "watch.lock"
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys,time; from basic_memory.index.watch_leader import try_acquire; "
            "from pathlib import Path; fd=try_acquire(Path(sys.argv[1])); "
            "print('held' if fd is not None else 'free', flush=True); time.sleep(60)",
            str(lock),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout.readline().strip() == "held"
        assert try_acquire(lock) is None
    finally:
        holder.kill()
        holder.wait()
        holder.stdout.close()
    # Windows releases a dead process's locks asynchronously; followers retry anyway.
    import time

    deadline = time.monotonic() + 5
    while (fd := try_acquire(lock)) is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert fd is not None
    watch_leader.os.close(fd)


@pytest.mark.asyncio
async def test_only_lock_holder_watches_and_follower_takes_over(tmp_path):
    lock = tmp_path / "watch.lock"
    leader_coord, follower_coord = FakeCoordinator(), FakeCoordinator()
    leader = LeaderElectedWatch(leader_coord, lock, retry_seconds=0.05)
    follower = LeaderElectedWatch(follower_coord, lock, retry_seconds=0.05)

    await leader.start()
    await follower.start()
    assert leader.is_leader and leader_coord.starts == 1
    # Negative control: while the leader runs, the follower never starts a watcher.
    await asyncio.sleep(0.2)
    assert not follower.is_leader and follower_coord.starts == 0

    await leader.stop()
    for _ in range(40):
        if follower_coord.starts:
            break
        await asyncio.sleep(0.05)
    assert follower.is_leader and follower_coord.starts == 1

    await follower.stop()
    assert follower_coord.stops == 1 and not follower.is_leader


@pytest.mark.asyncio
async def test_disabled_watching_or_env_keeps_upstream_behavior(tmp_path, monkeypatch):
    lock = tmp_path / "watch.lock"
    skipped = FakeCoordinator(should_watch=False)
    await LeaderElectedWatch(skipped, lock).start()
    assert skipped.starts == 1 and not lock.exists()

    monkeypatch.setenv(watch_leader.LOCK_ENV, "false")
    held = try_acquire(lock)
    try:
        independent = FakeCoordinator()
        await LeaderElectedWatch(independent, lock).start()
        assert independent.starts == 1
    finally:
        watch_leader.os.close(held)


@pytest.mark.asyncio
async def test_unusable_lock_path_still_watches(tmp_path):
    blocker = tmp_path / "not-a-dir"
    blocker.write_text("x")
    coord = FakeCoordinator()
    await LeaderElectedWatch(coord, blocker / "watch.lock").start()
    assert coord.starts == 1
