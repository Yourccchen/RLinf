# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch


def decode_config_sequence(value: Any, field_name: str) -> list[Any]:
    """Decode a YAML sequence or a JSON-array environment variable."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{field_name} must be a JSON array, got {value!r}."
            ) from exc
    if not isinstance(value, (Sequence, np.ndarray)):
        raise ValueError(
            f"{field_name} must be a sequence, got {type(value).__name__}."
        )
    return list(value)


class SonglingActionCodec:
    """Map normalized Stage2 actions to physical absolute Songling qpos."""

    def __init__(self, action_low: Any, action_high: Any) -> None:
        low = np.asarray(action_low, dtype=np.float32).reshape(-1)
        high = np.asarray(action_high, dtype=np.float32).reshape(-1)
        if low.shape != (14,) or high.shape != (14,):
            raise ValueError(
                "Songling action limits must each contain 14 values, got "
                f"low={low.shape}, high={high.shape}."
            )
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError("Songling action limits must be finite.")
        if np.any(high <= low):
            bad = np.flatnonzero(high <= low).tolist()
            raise ValueError(f"Songling action_high must exceed action_low at {bad}.")
        self.low = low
        self.high = high
        self._scale = (high - low) / 2.0
        self._bias = (high + low) / 2.0

    @classmethod
    def from_config(cls, cfg: Any) -> "SonglingActionCodec":
        return cls(
            decode_config_sequence(cfg.get("action_low"), "action_low"),
            decode_config_sequence(cfg.get("action_high"), "action_high"),
        )

    @staticmethod
    def _validate_actions(actions: Any, name: str) -> None:
        shape = tuple(actions.shape)
        if not shape or shape[-1] != 14:
            raise ValueError(f"{name} must have last dimension 14, got {shape}.")
        finite = (
            torch.isfinite(actions).all()
            if torch.is_tensor(actions)
            else np.isfinite(actions).all()
        )
        if not bool(finite):
            raise ValueError(f"{name} must contain only finite values.")

    def encode(self, physical_actions: Any, *, clip: bool = False):
        """Convert physical absolute qpos to the Actor/Critic [-1, 1] domain."""
        if torch.is_tensor(physical_actions):
            self._validate_actions(physical_actions, "physical_actions")
            low = torch.as_tensor(
                self.low, device=physical_actions.device, dtype=physical_actions.dtype
            )
            high = torch.as_tensor(
                self.high, device=physical_actions.device, dtype=physical_actions.dtype
            )
            normalized = 2.0 * (physical_actions - low) / (high - low) - 1.0
            return normalized.clamp(-1.0, 1.0) if clip else normalized
        values = np.asarray(physical_actions, dtype=np.float32)
        self._validate_actions(values, "physical_actions")
        normalized = (values - self._bias) / self._scale
        return np.clip(normalized, -1.0, 1.0) if clip else normalized

    def decode(self, normalized_actions: Any, *, clip: bool = True):
        """Convert Actor/Critic actions to physical absolute qpos."""
        if torch.is_tensor(normalized_actions):
            self._validate_actions(normalized_actions, "normalized_actions")
            values = normalized_actions.clamp(-1.0, 1.0) if clip else normalized_actions
            scale = torch.as_tensor(
                self._scale, device=values.device, dtype=values.dtype
            )
            bias = torch.as_tensor(self._bias, device=values.device, dtype=values.dtype)
            return values * scale + bias
        values = np.asarray(normalized_actions, dtype=np.float32)
        self._validate_actions(values, "normalized_actions")
        if clip:
            values = np.clip(values, -1.0, 1.0)
        return values * self._scale + self._bias
