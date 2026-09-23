# Copyright 2026 The RLinf Authors.

"""Versioned, backward-compatible contract shared with SsEvalPlatform/RWI."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping

import numpy as np

RLT_SSEVAL_PROTOCOL_VERSION = "songling-rlt-sseval-v1"
ACTION_DIM = 14
# Default used by the checked-in Songling recipe and tests. The wire contract
# does not pin this value: execute length comes from Stage1
# ``num_action_chunks`` and is advertised as Policy ``chunk_len``.
CHUNK_LEN = 50


def songling_chunk_shape(chunk_len: int) -> tuple[int, int]:
    """Return the Songling ``(C, action_dim)`` chunk shape for ``chunk_len``."""
    horizon = int(chunk_len)
    if horizon < 1:
        raise ValueError(f"chunk_len must be positive, got {horizon}.")
    return horizon, ACTION_DIM


class SelectedMode(str, Enum):
    VLA = "vla"
    ACTOR = "actor"
    HUMAN = "human"


def _array(value: Any, *, name: str, dtype: Any, ndim: int) -> np.ndarray:
    result = np.asarray(value, dtype=dtype)
    if result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got {result.shape}.")
    if np.issubdtype(result.dtype, np.floating) and not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values.")
    return np.ascontiguousarray(result)


@dataclass(frozen=True)
class DualActionCandidates:
    episode_id: str
    chunk_id: int
    vla_action: np.ndarray
    actor_action: np.ndarray
    actor_ready: bool
    actor_version: int
    feature_checkpoint_hash: int
    reference_seed: int

    def __post_init__(self) -> None:
        episode_id = str(self.episode_id).strip()
        if not episode_id:
            raise ValueError("episode_id must be non-empty.")
        if int(self.chunk_id) < 0:
            raise ValueError("chunk_id must be non-negative.")
        object.__setattr__(self, "episode_id", episode_id)
        object.__setattr__(self, "chunk_id", int(self.chunk_id))
        vla = _array(self.vla_action, name="vla_action", dtype=np.float32, ndim=2)
        actor = _array(self.actor_action, name="actor_action", dtype=np.float32, ndim=2)
        if vla.shape[0] < 1 or vla.shape[1] != ACTION_DIM:
            raise ValueError(
                f"vla_action must have shape (C, {ACTION_DIM}) with C>=1, "
                f"got {vla.shape}."
            )
        if actor.shape[0] < 1 or actor.shape[1] != ACTION_DIM:
            raise ValueError(
                f"actor_action must have shape (C, {ACTION_DIM}) with C>=1, "
                f"got {actor.shape}."
            )
        if vla.shape[0] < actor.shape[0]:
            raise ValueError(
                "VLA horizon must cover the Actor execute horizon, got "
                f"vla={vla.shape}, actor={actor.shape}."
            )
        object.__setattr__(self, "vla_action", vla)
        object.__setattr__(self, "actor_action", actor)
        object.__setattr__(self, "actor_ready", bool(self.actor_ready))
        object.__setattr__(self, "actor_version", int(self.actor_version))
        object.__setattr__(
            self, "feature_checkpoint_hash", int(self.feature_checkpoint_hash)
        )
        object.__setattr__(self, "reference_seed", int(self.reference_seed))

    def to_action_kwargs(self) -> dict[str, Any]:
        return {
            "rlt_protocol_version": RLT_SSEVAL_PROTOCOL_VERSION,
            "episode_id": self.episode_id,
            "chunk_id": self.chunk_id,
            "actor_action": self.actor_action,
            "actor_ready": self.actor_ready,
            "actor_version": self.actor_version,
            "feature_checkpoint_hash": self.feature_checkpoint_hash,
            "reference_seed": self.reference_seed,
            "chunk_len": int(self.actor_action.shape[0]),
            "vla_chunk_len": int(self.vla_action.shape[0]),
        }


@dataclass(frozen=True)
class TransitionFeedback:
    episode_id: str
    chunk_id: int
    selected_mode: SelectedMode
    executed_actions: np.ndarray
    valid_step_mask: np.ndarray
    rewards: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    intervene_flags: np.ndarray
    rlt_switch_flags: np.ndarray
    actor_version: int
    timestamps: Mapping[str, Any]

    @classmethod
    def from_mapping(
        cls, payload: Mapping[str, Any], *, chunk_len: int | None = None
    ) -> "TransitionFeedback":
        version = payload.get("rlt_protocol_version")
        if version != RLT_SSEVAL_PROTOCOL_VERSION:
            raise ValueError(
                f"Unsupported RLT feedback protocol {version!r}; "
                f"expected {RLT_SSEVAL_PROTOCOL_VERSION!r}."
            )
        episode_id = str(payload.get("episode_id") or "").strip()
        if not episode_id:
            raise ValueError("transition_feedback.episode_id must be non-empty.")
        mode = SelectedMode(str(payload.get("selected_mode")))
        actions = _array(
            payload.get("executed_actions"),
            name="executed_actions",
            dtype=np.float32,
            ndim=2,
        )
        expected = songling_chunk_shape(
            int(actions.shape[0] if chunk_len is None else chunk_len)
        )
        if actions.shape != expected:
            raise ValueError(
                "executed_actions must be padded to "
                f"{expected}, got {actions.shape}."
            )

        def vector(name: str, dtype: Any) -> np.ndarray:
            value = _array(payload.get(name), name=name, dtype=dtype, ndim=1)
            if value.shape != (expected[0],):
                raise ValueError(
                    f"{name} must have shape {(expected[0],)}, got {value.shape}."
                )
            return value

        valid = vector("valid_step_mask", np.bool_)
        if not valid.any():
            raise ValueError("valid_step_mask must contain at least one valid step.")
        rewards = vector("rewards", np.float32)
        terminated = vector("terminated", np.bool_)
        truncated = vector("truncated", np.bool_)
        intervene = vector("intervene_flags", np.bool_)
        rlt_switch = vector("rlt_switch_flags", np.bool_)
        if mode is SelectedMode.HUMAN and not intervene[valid].all():
            raise ValueError(
                "Human-mode valid steps must all set intervene_flags=True."
            )
        if mode is not SelectedMode.HUMAN and intervene[valid].any():
            raise ValueError("Only human-mode steps may set intervene_flags=True.")
        return cls(
            episode_id=episode_id,
            chunk_id=int(payload.get("chunk_id")),
            selected_mode=mode,
            executed_actions=actions,
            valid_step_mask=valid,
            rewards=rewards,
            terminated=terminated,
            truncated=truncated,
            intervene_flags=intervene,
            rlt_switch_flags=rlt_switch,
            actor_version=int(payload.get("actor_version", 0)),
            timestamps=dict(payload.get("timestamps") or {}),
        )

    @property
    def key(self) -> tuple[str, int]:
        return self.episode_id, self.chunk_id

    @property
    def done(self) -> bool:
        valid = self.valid_step_mask
        return bool((self.terminated[valid] | self.truncated[valid]).any())

    @property
    def has_labeled_outcome(self) -> bool:
        """True when success/failure terminated the episode, not abort."""
        valid = self.valid_step_mask
        return bool(self.terminated[valid].any())
