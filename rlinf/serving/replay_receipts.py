# Copyright 2026 The RLinf Authors.

"""Durable, optional receipts for locally recorded ARIO episodes."""

import hashlib
import json
import logging
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

    def _path(self, episode_id: str) -> Path:
        name = hashlib.sha256(episode_id.encode()).hexdigest()
        return self.root / f"{name}.json"

    def start(
        self, episode_id: str, instruction: str, stride: int, source_episode: str
    ) -> None:
        with self._lock:
            if self._path(episode_id).exists() or episode_id in self._episodes:
                raise ValueError(f"ARIO episode already exists: {episode_id}")
            self._episodes[episode_id] = {
                "episode_id": source_episode,
                "instruction": instruction,
                "stride": stride,
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
        with self._lock:
            path = self._path(episode_id)
            if not path.exists():
                return {"episode_id": episode_id, "complete": False}
            result = json.loads(path.read_text())
            result["complete"] = result["finalized"] and all(
                item["accepted"] for item in result["transitions"]
            )
            return result

    def _save(self, episode_id: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self._path(episode_id)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._episodes[episode_id]))
        temporary.replace(path)
