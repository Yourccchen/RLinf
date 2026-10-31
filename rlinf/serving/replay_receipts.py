# Copyright 2026 The RLinf Authors.

"""Durable, optional receipts for locally recorded ARIO episodes."""

import hashlib
import json
import logging
import queue
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ReplayReceipts:
    """Track actual replay insertion without changing trajectory serialization."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lock = threading.RLock()
        self._episodes: dict[str, dict[str, Any]] = {}
        self._pending: set[str] = set()
        self._errors: dict[str, str] = {}
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=128)
        self._thread: threading.Thread | None = None
        self._closed = False

    def _path(self, episode_id: str) -> Path:
        name = hashlib.sha256(episode_id.encode()).hexdigest()
        return self.root / f"{name}.json"

    def start(self, episode_id: str, instruction: str, source_episode: str) -> None:
        with self._lock:
            if episode_id in self._episodes:
                raise ValueError(f"ARIO episode already exists: {episode_id}")
            self._episodes[episode_id] = {
                "episode_id": source_episode,
                "instruction": instruction,
                "finalized": False,
                "transitions": [],
            }
            self._save(episode_id)

    def contains(self, episode_id: str) -> bool:
        with self._lock:
            return episode_id in self._episodes

    def expect(self, episode_id: str, transitions: list[dict]) -> None:
        with self._lock:
            self._episodes[episode_id]["transitions"] = transitions
            self._save(episode_id)

    def accept(self, episode_id: str, weights_id: str) -> None:
        with self._lock:
            for item in self._episodes[episode_id]["transitions"]:
                if item["weights_id"] == weights_id:
                    item["accepted"] = True
                    break
            self._save(episode_id)

    def finish(self, episode_id: str) -> None:
        with self._lock:
            self._episodes[episode_id]["finalized"] = True
            self._save(episode_id)

    def status(self, episode_id: str) -> dict[str, Any]:
        """Read a durable receipt; call off the inference/control thread."""
        self._queue.join()
        with self._lock:
            error = self._errors.get(episode_id)
        if error is not None:
            return {"episode_id": episode_id, "complete": False, "error": error}
        path = self._path(episode_id)
        if not path.exists():
            return {"episode_id": episode_id, "complete": False}
        result = json.loads(path.read_text())
        result["complete"] = result["finalized"] and all(
            item["accepted"] for item in result["transitions"]
        )
        return result

    def close(self) -> None:
        """Drain accepted updates and release the optional writer thread."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            thread = self._thread
        if thread is not None:
            self._queue.put(None)
            thread.join()

    def _save(self, episode_id: str) -> None:
        if self._closed:
            raise RuntimeError("Replay receipt writer is closed")
        if episode_id in self._pending or episode_id in self._errors:
            return
        if self._thread is None:
            self._thread = threading.Thread(
                target=self._write, daemon=True, name="replay-receipts"
            )
            self._thread.start()
        self._pending.add(episode_id)
        try:
            self._queue.put_nowait(episode_id)
        except queue.Full:
            self._pending.discard(episode_id)
            self._errors[episode_id] = "Replay receipt writer queue is full"
            logger.error("Replay receipt writer queue is full: %s", episode_id)

    def _write(self) -> None:
        written = set()
        while True:
            episode_id = self._queue.get()
            try:
                if episode_id is None:
                    return
                with self._lock:
                    self._pending.discard(episode_id)
                    if episode_id in self._errors:
                        continue
                    payload = json.dumps(self._episodes[episode_id])
                self.root.mkdir(parents=True, exist_ok=True)
                path = self._path(episode_id)
                if episode_id not in written and path.exists():
                    raise ValueError(f"ARIO receipt already exists: {episode_id}")
                temporary = path.with_suffix(".tmp")
                temporary.write_text(payload)
                temporary.replace(path)
                written.add(episode_id)
            except Exception as exc:
                with self._lock:
                    self._errors[episode_id] = str(exc)
                logger.exception("Replay receipt persistence failed: %s", episode_id)
            finally:
                self._queue.task_done()
