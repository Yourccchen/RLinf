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

"""Policy transforms for Xingchen (Astribot-S1) with ARIO-format data.

The 31-D action/state vector layout:
    endpose_torso(9) + qpos_head(2) + endpose_left(9) + gripper_left(1)
        + endpose_right(9) + gripper_right(1) = 31
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

ACTION_DIM = 31


def make_xingchen_example() -> dict:
    """Creates a random input example for the Xingchen policy."""
    return {
        "observation/image": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/cam_high": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/cam_left_wrist": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
        "observation/cam_right_wrist": np.random.randint(256, size=(240, 320, 3), dtype=np.uint8),
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
class XingchenInputs(transforms.DataTransformFn):
    """Maps Xingchen/ARIO data fields to the format expected by pi0.5.

    Supports both single-view (observation/image) and multi-view
    (observation/cam_high, observation/cam_left_wrist, observation/cam_right_wrist).
    When a camera is absent the slot is filled with a zero image and its mask is False.
    """

    model_type: _model.ModelType
    default_prompt: str | None = None

    def __call__(self, data: dict) -> dict:
        # Multi-view: prefer dedicated camera keys, fall back to single image.
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
            "state": data["observation/state"],
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
            inputs["actions"] = data["actions"]

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        elif "task" in data:
            inputs["prompt"] = data["task"]
        elif self.default_prompt is not None:
            inputs["prompt"] = self.default_prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class XingchenOutputs(transforms.DataTransformFn):
    """Extracts the first ACTION_DIM action dimensions from model output."""

    output_action_dim: int = ACTION_DIM

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., : self.output_action_dim])}
