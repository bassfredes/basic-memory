"""Run one local watcher across MCP processes that share a data directory.

Every MCP client session starts its own server process. On WSL, where watchfiles
falls back to polling, each of those processes would poll and re-index the same
project tree, multiplying CPU, filesystem-bridge load and duplicate indexing.
Only the process holding an exclusive lock on ``<data dir>/watch.lock`` runs the
watcher; the others serve requests and retry periodically. The kernel drops an
flock together with its process, so a follower takes over (including the
coordinator's normal startup recovery scan) when the holder exits.
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
except ImportError:  # pragma: no cover - Windows has no flock; keep upstream behavior
    fcntl = None  # type: ignore[assignment]

LOCK_ENV = "BASIC_MEMORY_WATCH_LEADER_LOCK"
RETRY_ENV = "BASIC_MEMORY_WATCH_LEADER_RETRY_SECONDS"
DEFAULT_RETRY_SECONDS = 30.0


def leader_lock_enabled() -> bool:
    """Leader election is on by default wherever flock exists."""
    return fcntl is not None and os.getenv(LOCK_ENV, "true").lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def try_acquire(lock_path: Path) -> int | None:
    """Return a locked file descriptor, or None when another process holds it."""
    assert fcntl is not None
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    except OSError:
        os.close(fd)
        raise
    os.ftruncate(fd, 0)
    os.write(fd, str(os.getpid()).encode())
    return fd


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
        # Trigger: watching disabled (test/cloud/index_changes=false) or no flock.
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
