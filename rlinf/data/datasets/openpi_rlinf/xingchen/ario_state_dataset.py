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

"""Read Xingchen semantic pose fields from unified ARIO state.pt episodes."""

from __future__ import annotations

import dataclasses
import json
from typing import Any

import numpy as np
import torch
from openpi.datasets.ario_dataset import ArioConfig, ArioStreamingDataset
from tqdm import tqdm

from rlinf.utils.logging import get_logger

logger = get_logger()


def extract_xingchen_state(state: dict[str, torch.Tensor]) -> np.ndarray:
    """Return [T, 31] absolute poses, head joints, and raw gripper values.

    Order: torso pose9, head2, left pose9, left gripper1, right pose9,
    right gripper1. Semantic fields retain source units and coordinate axes;
    canonical55 EEF axes and gripper polarity conversions are not applied.
    """
    fields = (
        ("endpose_torso", 9),
        ("qpos_head", 2),
        ("endpose_left", 9),
        ("gripper_left", 1),
        ("endpose_right", 9),
        ("gripper_right", 1),
    )
    parts = []
    for name, width in fields:
        value = state[name]
        if value.ndim != 2 or value.shape[1] != width:
            raise ValueError(f"{name} must have shape [T, {width}].")
        if parts and len(value) != len(parts[0]):
            raise ValueError("Xingchen state fields must have matching frame counts.")
        if not torch.isfinite(value).all():
            raise ValueError(f"{name} contains non-finite values.")
        parts.append(value)
    return torch.cat(parts, dim=-1).float().numpy()


class XingchenStateDataset(ArioStreamingDataset):
    """ARIO streaming with unified state.pt and the Xingchen 31-D pose layout.

    Episode discovery, filtering, cameras, frame sampling, and action chunks
    use the upstream implementation. Only state loading and indexing differ.
    """

    def __init__(self, config: ArioConfig, action_horizon: int):
        # Upstream validates its two native format names. This adapter provides
        # the unified-state implementation of Xingchen's existing pose layout.
        super().__init__(
            config=dataclasses.replace(config, data_format="xingchen"),
            action_horizon=action_horizon,
        )

    def _build_state_and_actions(
        self, s3: Any, bucket: str, prefix: str
    ) -> tuple[np.ndarray, np.ndarray]:
        path = self._get_cached_object(s3, bucket, prefix + "state.pt")
        state = torch.load(path, map_location="cpu", weights_only=True)
        values = extract_xingchen_state(state)
        return values, values

    def _build_index(self) -> None:
        s3 = self._get_s3()
        valid_episodes = []
        cumulative = 0
        for bucket, prefix in tqdm(
            self._episodes, desc="Building Xingchen frame index"
        ):
            # Fail on malformed or unreadable state instead of silently dropping
            # it from both training and the normalization population.
            states, _ = self._build_state_and_actions(s3, bucket, prefix)
            if len(states) < self._config.min_frames:
                continue
            if self._config.load_instructions:
                path = self._get_cached_object(s3, bucket, prefix + "instructions.json")
                instructions = json.loads(path.read_text(encoding="utf-8"))
                self._instructions[prefix] = self._parse_instruction_segments(
                    instructions, self._config.instruction_field
                )
            count = len(range(0, len(states), self._config.video_downsample_rate))
            valid_episodes.append((bucket, prefix))
            self._episode_lengths.append(count)
            cumulative += count
            self._cumulative.append(cumulative)
        self._episodes = valid_episodes
        logger.info(
            "Xingchen ARIO: %d episodes, %d frames", len(valid_episodes), cumulative
        )


def create_ario_dataset(
    config: ArioConfig, action_horizon: int
) -> ArioStreamingDataset:
    """Build a native ARIO dataset or the unified Xingchen state adapter."""
    if config.data_format == "xingchen_state":
        return XingchenStateDataset(config, action_horizon)
    return ArioStreamingDataset(config=config, action_horizon=action_horizon)
