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

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from rlinf.data.schema.embodied_types import Trajectory

RLT_KEYS = ("z_rl", "proprio", "ref_chunk")


def transition_window_starts(
    num_steps: int, *, chunk_len: int, stride: int
) -> list[int]:
    if num_steps <= 0:
        raise ValueError("An offline episode must contain at least one action step.")
    if chunk_len <= 0 or stride <= 0:
        raise ValueError("chunk_len and transition_stride must be positive.")
    return list(range(0, num_steps, stride))


def _step_tensor(episode: Mapping[str, Any], key: str, num_steps: int, dtype):
    if key not in episode:
        raise ValueError(f"Offline episode is missing required field {key!r}.")
    value = torch.as_tensor(episode[key], dtype=dtype)
    if value.shape[0] != num_steps:
        raise ValueError(
            f"Offline {key} length must be {num_steps}, got {value.shape[0]}."
        )
    return value


def _feature_at(
    features: Sequence[Mapping[str, Any]], index: int
) -> dict[str, torch.Tensor]:
    if not 0 <= index < len(features):
        raise IndexError(f"Feature index {index} outside [0, {len(features)}).")
    result = {}
    for key in RLT_KEYS:
        if key not in features[index]:
            raise ValueError(f"Offline feature {index} is missing {key!r}.")
        result[key] = (
            torch.as_tensor(features[index][key]).detach().cpu().to(torch.float32)
        )
    return result


def _pad_first_dim(
    value: torch.Tensor, length: int, fill: float | bool = 0
) -> torch.Tensor:
    if value.shape[0] > length:
        raise ValueError(f"Cannot pad length {value.shape[0]} down to {length}.")
    if value.shape[0] == length:
        return value
    pad_shape = (length - value.shape[0], *value.shape[1:])
    padding = torch.full(pad_shape, fill, dtype=value.dtype)
    return torch.cat([value, padding], dim=0)


def build_offline_rlt_trajectories(
    episode: Mapping[str, Any],
    features: Sequence[Mapping[str, Any]],
    *,
    chunk_len: int = 10,
    action_dim: int = 14,
    transition_stride: int = 2,
    model_weights_id: str = "offline",
) -> list[Trajectory]:
    """Convert step-aligned episode data into overlapping chunk transitions.

    ``features`` must contain one RLT observation for every environment state,
    hence ``len(features) == num_action_steps + 1``. Actions are expected in the
    normalized Actor/Critic domain and must be the actions actually executed.
    """
    if "executed_actions" not in episode:
        raise ValueError(
            "Offline TD3 requires executed_actions; proposed policy actions are invalid."
        )
    actions = torch.as_tensor(episode["executed_actions"], dtype=torch.float32)
    if actions.ndim != 2 or actions.shape[1] != action_dim:
        raise ValueError(
            f"executed_actions must have shape [T,{action_dim}], got {tuple(actions.shape)}."
        )
    if not torch.isfinite(actions).all():
        raise ValueError("executed_actions must be finite.")
    num_steps = int(actions.shape[0])
    if len(features) != num_steps + 1:
        raise ValueError(
            "Offline features must contain T+1 states, got "
            f"T={num_steps}, features={len(features)}."
        )
    rewards = _step_tensor(episode, "rewards", num_steps, torch.float32).reshape(-1)
    terminations = _step_tensor(episode, "terminated", num_steps, torch.bool).reshape(
        -1
    )
    truncations = _step_tensor(episode, "truncated", num_steps, torch.bool).reshape(-1)
    if not torch.isfinite(rewards).all():
        raise ValueError("Offline rewards must be finite.")
    if not (terminations | truncations).any():
        raise ValueError(
            "Offline TD3 requires explicit success/failure/timeout terminal labels."
        )
    intervention = torch.as_tensor(
        episode.get("intervene_flags", torch.zeros(num_steps)), dtype=torch.bool
    ).reshape(-1)
    if intervention.shape[0] != num_steps:
        raise ValueError("intervene_flags must be step-aligned with executed_actions.")

    trajectories = []
    for start in transition_window_starts(
        num_steps, chunk_len=chunk_len, stride=transition_stride
    ):
        stop = min(start + chunk_len, num_steps)
        actual_len = stop - start
        window_done = terminations[start:stop] | truncations[start:stop]
        done_positions = torch.nonzero(window_done, as_tuple=False).reshape(-1)
        if done_positions.numel() > 0:
            actual_len = int(done_positions[0]) + 1
            stop = start + actual_len
        valid = torch.zeros(chunk_len, dtype=torch.bool)
        valid[:actual_len] = True
        curr_obs = _feature_at(features, start)
        next_obs = _feature_at(features, stop)
        action_chunk = _pad_first_dim(actions[start:stop], chunk_len)
        reward_chunk = _pad_first_dim(rewards[start:stop], chunk_len)
        terminated_chunk = _pad_first_dim(terminations[start:stop], chunk_len, False)
        truncated_chunk = _pad_first_dim(truncations[start:stop], chunk_len, False)
        intervene_chunk = _pad_first_dim(intervention[start:stop], chunk_len, False)
        trajectories.append(
            Trajectory(
                max_episode_length=1,
                model_weights_id=model_weights_id,
                actions=action_chunk.reshape(1, 1, -1),
                intervene_flags=intervene_chunk.reshape(1, 1, -1),
                rewards=reward_chunk.reshape(1, 1, -1),
                terminations=terminated_chunk.reshape(1, 1, -1),
                truncations=truncated_chunk.reshape(1, 1, -1),
                dones=(terminated_chunk | truncated_chunk).reshape(1, 1, -1),
                versions=torch.zeros((1, 1, 1), dtype=torch.float32),
                forward_inputs={
                    "action": action_chunk.reshape(1, 1, -1),
                    "valid_step_mask": valid.reshape(1, 1, -1),
                    "record_transition": torch.ones((1, 1, 1), dtype=torch.bool),
                },
                curr_obs={
                    key: value.reshape(1, 1, *value.shape)
                    for key, value in curr_obs.items()
                },
                next_obs={
                    key: value.reshape(1, 1, *value.shape)
                    for key, value in next_obs.items()
                },
            )
        )
        if done_positions.numel() > 0:
            break
    return trajectories
