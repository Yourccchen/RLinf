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

"""DataConfig for Songling dual-arm ARIO data (canonical55 -> 14-D qpos).

Mirrors ``openpi.training.config.ArioSonglingDataConfig`` so the RLinf RLT
Stage-1 run consumes the same transform pipeline as the JAX openpi_Ario runs.
"""

import dataclasses
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import songling_policy


@dataclasses.dataclass(frozen=True)
class SonglingRLTDataConfig(DataConfigFactory):
    """DataConfig for Songling ARIO canonical55 datasets (Stage-1 RLT SFT)."""

    default_prompt: str = "fold clothes"
    # Joint deltas for the two 6-DoF arms, absolute values for the two grippers.
    use_delta_actions: bool = True
    output_action_dim: int = songling_policy.ACTION_DIM

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        # ArioStreamingDataset already emits observation/* and actions directly,
        # so the repack is an identity mapping over the keys it produces.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation/image",
                        "observation/cam_high": "observation/cam_high",
                        "observation/cam_left_wrist": "observation/cam_left_wrist",
                        "observation/cam_right_wrist": "observation/cam_right_wrist",
                        "observation/state": "observation/state",
                        "actions": "actions",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                songling_policy.SonglingInputs(
                    model_type=model_config.model_type,
                    default_prompt=self.default_prompt,
                )
            ],
            outputs=[
                songling_policy.SonglingOutputs(
                    output_action_dim=self.output_action_dim
                )
            ],
        )

        if self.use_delta_actions:
            # [qpos_left 6 -> delta, gripper_left 1 -> absolute,
            #  qpos_right 6 -> delta, gripper_right 1 -> absolute]
            delta_action_mask = _transforms.make_bool_mask(6, -1, 6, -1)
            data_transforms = data_transforms.push(
                inputs=[_transforms.DeltaActions(delta_action_mask)],
                outputs=[_transforms.AbsoluteActions(delta_action_mask)],
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
