"""Run one local watcher across MCP processes that share a data directory.

Every MCP client session starts its own server process. On WSL, where watchfiles
falls back to polling, each of those processes would poll and re-index the same
project tree, multiplying CPU, filesystem-bridge load and duplicate indexing.
Only the process holding an exclusive lock on ``<data dir>/watch.lock`` runs the
watcher; the others serve requests and retry periodically. The OS drops the lock
(flock on POSIX, a msvcrt byte-range lock on Windows) together with its process,
so a follower takes over (including the coordinator's normal startup recovery
scan) when the holder exits.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import suppress
from pathlib import Path

from loguru import logger

from basic_memory.index.watch_coordinator import WatchCoordinator

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows uses msvcrt below
    fcntl = None  # type: ignore[assignment]

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX uses fcntl above
    msvcrt = None  # type: ignore[assignment]

LOCK_ENV = "BASIC_MEMORY_WATCH_LEADER_LOCK"
RETRY_ENV = "BASIC_MEMORY_WATCH_LEADER_RETRY_SECONDS"
DEFAULT_RETRY_SECONDS = 30.0
# Windows locks a byte range, and a locked byte cannot be read by other processes.
# Lock one byte far past the PID text so the holder's PID stays readable.
WINDOWS_LOCK_OFFSET = 1 << 20


def leader_lock_enabled() -> bool:
    """Leader election is on by default wherever an OS file lock exists."""
    return (fcntl is not None or msvcrt is not None) and os.getenv(LOCK_ENV, "true").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def try_acquire(lock_path: Path) -> int | None:
    """Return a locked file descriptor, or None when another process holds it."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if not _lock_nonblocking(fd):
            os.close(fd)
            return None
    except OSError:
        os.close(fd)
        raise
    os.lseek(fd, 0, os.SEEK_SET)
    os.ftruncate(fd, 0)
    os.write(fd, str(os.getpid()).encode())
    return fd


def _lock_nonblocking(fd: int) -> bool:
    """Take an exclusive lock without waiting; False when another process holds it."""
    if fcntl is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        return True
    assert msvcrt is not None
    os.lseek(fd, WINDOWS_LOCK_OFFSET, os.SEEK_SET)
    try:
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
    except PermissionError:
        # LK_NBLCK fails with EACCES when the byte is locked by another handle.
        return False
    return True


class LeaderElectedWatch:
    """Wrap a WatchCoordinator so only the lock holder starts it."""

    def __init__(
        self,
        coordinator: WatchCoordinator,
        lock_path: Path,
        retry_seconds: float | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.lock_path = lock_path
        self.retry_seconds = (
            retry_seconds
            if retry_seconds is not None
            else float(os.getenv(RETRY_ENV, DEFAULT_RETRY_SECONDS))
        )
        self._fd: int | None = None
        self._wait_task: asyncio.Task[None] | None = None

    @property
    def is_leader(self) -> bool:
        return self._fd is not None

    async def start(self) -> None:
        # Trigger: watching disabled (test/cloud/index_changes=false) or no OS file lock.
        # Why: those paths keep their existing behavior exactly.
        if not self.coordinator.should_watch or not leader_lock_enabled():
            await self.coordinator.start()
            return
        try:
            self._fd = try_acquire(self.lock_path)
        except OSError as exc:
            # An unusable lock file must not cost the user their index updates.
            logger.warning(f"Watch lock unavailable ({exc}); starting local watcher anyway")
            await self.coordinator.start()
            return
        if self._fd is not None:
            logger.info(f"Holding watch lock {self.lock_path}; starting local watcher")
            await self.coordinator.start()
            return
        logger.info("Another Basic Memory process holds the watch lock; serving without a watcher")
        self._wait_task = asyncio.create_task(self._await_leadership())

    async def _await_leadership(self) -> None:
        while True:
            await asyncio.sleep(self.retry_seconds)
            try:
                fd = try_acquire(self.lock_path)
            except OSError as exc:  # pragma: no cover - transient filesystem error
                logger.warning(f"Watch lock retry failed: {exc}")
                continue
            if fd is None:
                continue
            self._fd = fd
            logger.info("Acquired watch lock after the previous holder exited; starting watcher")
            try:
                await self.coordinator.start()
            except Exception as exc:  # pragma: no cover - coordinator already logs details
                logger.error(f"Local watcher failed to start after takeover: {exc}")
                self._release()
            return

    def _release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    async def stop(self) -> None:
        if self._wait_task is not None:
            self._wait_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._wait_task
            self._wait_task = None
        await self.coordinator.stop()
        self._release()
