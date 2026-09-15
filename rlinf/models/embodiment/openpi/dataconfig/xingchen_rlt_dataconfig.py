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

"""DataConfig for Xingchen (Astribot-S1) ARIO-format LeRobot datasets.

Supports both single-view (image) and multi-view (cam_high / cam_left_wrist /
cam_right_wrist) recordings.  The 31-D action/state vector layout is:
    endpose_torso(9) + qpos_head(2) + endpose_left(9) + gripper_left(1)
        + endpose_right(9) + gripper_right(1) = 31
"""

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import xingchen_policy


@dataclasses.dataclass(frozen=True)
class XingchenRLTDataConfig(DataConfigFactory):
    """DataConfig for Xingchen/ARIO LeRobot datasets (Stage-1 RLT SFT)."""

    default_prompt: str | None = "fold clothes"
    # Set to True if the LeRobot dataset stores images under cam_high /
    # cam_left_wrist / cam_right_wrist keys instead of a single "image" key.
    multi_view: bool = True
    output_action_dim: int = xingchen_policy.ACTION_DIM

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        # Build repack mapping depending on whether the dataset uses multi-view
        # keys or a single image key.
        if self.multi_view:
            repack_structure = {
                "observation/cam_high": "cam_high",
                "observation/cam_left_wrist": "cam_left_wrist",
                "observation/cam_right_wrist": "cam_right_wrist",
                "observation/state": "state",
                "actions": "actions",
            }
        else:
            repack_structure = {
                "observation/image": "image",
                "observation/state": "state",
                "actions": "actions",
            }

        # Add prompt/task key if present in the dataset.
        repack_structure["prompt"] = "prompt"

        repack_transform = _transforms.Group(
            inputs=[_transforms.RepackTransform(repack_structure)]
        )

        data_transforms = _transforms.Group(
            inputs=[
                xingchen_policy.XingchenInputs(
                    model_type=model_config.model_type,
                    default_prompt=self.default_prompt,
                )
            ],
            outputs=[
                xingchen_policy.XingchenOutputs(
                    output_action_dim=self.output_action_dim,
                )
            ],
        )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )
