# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Policy transforms for Songling dual-arm qpos control (ARIO canonical55).

The 14-D state/action vector is pulled out of ``state.pt``'s ``__canonical55__``
by ``ArioStreamingDataset`` (data_format="songling_canonical55"):

    qpos_left(6) + gripper_left(1) + qpos_right(6) + gripper_right(1) = 14

Mirrors ``openpi.policies.songling_policy`` so both stacks stay in step.
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

ACTION_DIM = 14


def make_songling_example() -> dict:
    """Create a random Songling policy input example."""
    return {
        "observation/image": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/cam_high": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/cam_left_wrist": np.random.randint(
            256, size=(240, 320, 3), dtype=np.uint8
        ),
        "observation/cam_right_wrist": np.random.randint(
            256, size=(240, 320, 3), dtype=np.uint8
        ),
        "observation/state": np.random.rand(ACTION_DIM).astype(np.float32),
        "prompt": "fold clothes",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class SonglingInputs(transforms.DataTransformFn):
    """Map Songling observations to the three image slots expected by pi0.5."""

    model_type: _model.ModelType
    default_prompt: str | None = None

    def __call__(self, data: dict) -> dict:
        if "observation/cam_high" in data:
            base_image = _parse_image(data["observation/cam_high"])
        else:
            base_image = _parse_image(data["observation/image"])

        if "observation/cam_left_wrist" in data:
            left_wrist_image = _parse_image(data["observation/cam_left_wrist"])
            left_wrist_mask = np.True_
        else:
            left_wrist_image = np.zeros_like(base_image)
            left_wrist_mask = np.False_

        if "observation/cam_right_wrist" in data:
            right_wrist_image = _parse_image(data["observation/cam_right_wrist"])
            right_wrist_mask = np.True_
        else:
            right_wrist_image = np.zeros_like(base_image)
            right_wrist_mask = np.False_

        inputs = {
            "state": np.asarray(data["observation/state"]),
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": left_wrist_mask,
                "right_wrist_0_rgb": right_wrist_mask,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"])
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        elif self.default_prompt is not None:
            inputs["prompt"] = self.default_prompt
        return inputs


@dataclasses.dataclass(frozen=True)
class SonglingOutputs(transforms.DataTransformFn):
    """Return only the 14 physical Songling action dimensions."""

    output_action_dim: int = ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.output_action_dim])}
