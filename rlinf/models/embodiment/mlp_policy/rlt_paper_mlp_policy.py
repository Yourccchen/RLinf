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

"""Paper-faithful Gaussian action-chunk policy for RLT Stage 2."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.mlp_policy.rlt_td3_mlp_policy import (
    TwinQCritic,
    _make_td3_mlp,
)


class RLTPaperGaussianActor(nn.Module):
    """Predict a direct Gaussian action chunk conditioned on the VLA proposal."""

    def __init__(
        self,
        *,
        state_dim: int,
        action_chunk_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 2,
        fixed_std: float = 0.002,
    ) -> None:
        super().__init__()
        self.fixed_std = float(fixed_std)
        if self.fixed_std <= 0.0:
            raise ValueError(f"fixed_std must be positive, got {self.fixed_std}.")
        self.mlp = _make_td3_mlp(
            input_dim=int(action_chunk_dim) + int(state_dim),
            output_dim=int(action_chunk_dim),
            hidden_dim=int(hidden_dim),
            num_hidden_layers=int(num_hidden_layers),
        )

    @staticmethod
    def _drop_reference(
        reference: torch.Tensor,
        probability: float,
    ) -> torch.Tensor:
        if probability <= 0.0:
            return reference
        if probability > 1.0:
            raise ValueError(
                f"reference_dropout_prob must be within [0, 1], got {probability}."
            )
        keep = torch.rand(reference.shape[0], 1, device=reference.device) >= probability
        return reference * keep.to(dtype=reference.dtype)

    def forward(
        self,
        state: torch.Tensor,
        reference: torch.Tensor,
        *,
        deterministic: bool = False,
        apply_reference_dropout: bool = False,
        reference_dropout_prob: float = 0.0,
        apply_action_noise: bool | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return tanh-squashed actions and pre-squash Gaussian log-probabilities."""
        actor_reference = (
            self._drop_reference(reference, float(reference_dropout_prob))
            if apply_reference_dropout
            else reference
        )
        mean = self.mlp(torch.cat([actor_reference, state], dim=-1))
        use_noise = (
            (not deterministic)
            if apply_action_noise is None
            else bool(apply_action_noise)
        )
        use_noise = use_noise and not deterministic
        distribution = Normal(mean, torch.full_like(mean, self.fixed_std))
        pre_tanh = distribution.rsample() if use_noise else mean
        return torch.tanh(pre_tanh), distribution.log_prob(pre_tanh)


class RLTPaperMLPPolicy(nn.Module, BasePolicy):
    """RLT paper Stage 2 actor and twin-Q critic over frozen RL-token features."""

    def __init__(
        self,
        z_dim: int,
        proprio_dim: int,
        action_dim: int,
        num_action_chunks: int,
        ref_num_action_chunks: int | None = None,
        add_q_head: bool = True,
        q_head_type: str = "default",
        mlp_hidden_dim: int = 256,
        mlp_num_hidden_layers: int = 2,
        fixed_std: float = 0.002,
    ) -> None:
        super().__init__()
        if not add_q_head:
            raise ValueError("RLTPaperMLPPolicy requires add_q_head=True.")
        if q_head_type != "default":
            raise ValueError(
                "RLTPaperMLPPolicy only supports q_head_type='default', got "
                f"{q_head_type!r}."
            )
        self.z_dim = int(z_dim)
        self.proprio_dim = int(proprio_dim)
        self.step_action_dim = int(action_dim)
        self.chunk_len = int(num_action_chunks)
        self.ref_chunk_len = (
            self.chunk_len
            if ref_num_action_chunks is None
            else int(ref_num_action_chunks)
        )
        if self.ref_chunk_len < self.chunk_len:
            raise ValueError(
                "ref_num_action_chunks must be >= num_action_chunks, got "
                f"{self.ref_chunk_len} < {self.chunk_len}."
            )
        self.action_dim = self.step_action_dim
        self.num_action_chunks = self.chunk_len
        self.flat_action_dim = self.chunk_len * self.step_action_dim
        self.state_dim = self.z_dim + self.proprio_dim
        self.torch_compile_enabled = False

        self.actor = RLTPaperGaussianActor(
            state_dim=self.state_dim,
            action_chunk_dim=self.flat_action_dim,
            hidden_dim=mlp_hidden_dim,
            num_hidden_layers=mlp_num_hidden_layers,
            fixed_std=fixed_std,
        )
        self.q_head = TwinQCritic(
            state_dim=self.state_dim,
            action_chunk_dim=self.flat_action_dim,
            hidden_dim=mlp_hidden_dim,
            num_hidden_layers=mlp_num_hidden_layers,
        )

    def preprocess_env_obs(self, env_obs: dict) -> dict:
        device = next(self.parameters()).device
        return {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in env_obs.items()
        }

    @staticmethod
    def _flatten_batch(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() <= 2:
            return tensor
        return tensor.reshape(tensor.shape[0], -1)

    def _get_z(self, obs: dict) -> torch.Tensor:
        value = self._flatten_batch(obs["z_rl"])
        if value.shape[-1] != self.z_dim:
            raise ValueError(
                f"Expected z_rl last dim {self.z_dim}, got {tuple(value.shape)}."
            )
        return value

    def _get_proprio(self, obs: dict) -> torch.Tensor:
        value = self._flatten_batch(obs["proprio"])
        if value.shape[-1] != self.proprio_dim:
            raise ValueError(
                "Expected proprio last dim "
                f"{self.proprio_dim}, got {tuple(value.shape)}."
            )
        return value

    def _get_ref_chunk(self, obs: dict) -> torch.Tensor:
        value = self._flatten_batch(obs["ref_chunk"])
        expected = self.ref_chunk_len * self.step_action_dim
        if value.shape[-1] != expected:
            raise ValueError(
                f"Expected ref_chunk flattened dim {expected}, got {tuple(value.shape)}."
            )
        value = value.reshape(-1, self.ref_chunk_len, self.step_action_dim)
        return value[:, : self.chunk_len].reshape(value.shape[0], -1)

    def _state(self, obs: dict) -> torch.Tensor:
        return torch.cat([self._get_z(obs), self._get_proprio(obs)], dim=-1)

    def _format_chunk_actions(self, actions: torch.Tensor) -> torch.Tensor:
        value = self._flatten_batch(actions)
        if value.shape[-1] != self.flat_action_dim:
            raise ValueError(
                f"Expected action dim {self.flat_action_dim}, got {tuple(value.shape)}."
            )
        return value.reshape(-1, self.chunk_len, self.step_action_dim)

    def default_forward(self, **kwargs):
        """Reject PPO-style forward calls for the off-policy RLT head."""
        del kwargs
        raise NotImplementedError(
            "RLTPaperMLPPolicy does not use PPO-style default_forward."
        )

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        obs = kwargs.get("obs")
        if obs is not None:
            kwargs["obs"] = self.preprocess_env_obs(obs)
        next_obs = kwargs.get("next_obs")
        if next_obs is not None:
            kwargs["next_obs"] = self.preprocess_env_obs(next_obs)
        if forward_type in (ForwardType.SAC, ForwardType.CROSSQ):
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        if forward_type == ForwardType.CROSSQ_Q:
            return self.crossq_q_forward(**kwargs)
        if forward_type == ForwardType.SFT:
            raise NotImplementedError("RLTPaperMLPPolicy does not implement SFT.")
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(
            f"RLTPaperMLPPolicy does not support forward_type={forward_type!r}."
        )

    def sac_forward(
        self,
        obs: dict,
        *,
        apply_reference_dropout: bool = False,
        reference_dropout_prob: float | None = None,
        deterministic: bool = False,
        apply_action_noise: bool | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, None]:
        del kwargs
        action, log_prob = self.actor(
            self._state(obs),
            self._get_ref_chunk(obs),
            deterministic=deterministic,
            apply_reference_dropout=apply_reference_dropout,
            reference_dropout_prob=float(reference_dropout_prob or 0.0),
            apply_action_noise=apply_action_noise,
        )
        return action, log_prob, None

    def sac_q_forward(
        self,
        obs: dict,
        actions: torch.Tensor,
        shared_feature=None,
        detach_encoder: bool = False,
    ) -> torch.Tensor:
        del shared_feature
        state = self._state(obs)
        if detach_encoder:
            state = state.detach()
        return self.q_head(state, self._format_chunk_actions(actions).flatten(1))

    def crossq_q_forward(
        self,
        obs: dict,
        actions: torch.Tensor,
        next_obs: dict | None = None,
        next_actions: torch.Tensor | None = None,
        shared_feature=None,
        detach_encoder: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        data_q = self.sac_q_forward(
            obs,
            actions,
            shared_feature=shared_feature,
            detach_encoder=detach_encoder,
        )
        if next_obs is None or next_actions is None:
            return data_q, data_q.new_zeros(data_q.shape)
        next_q = self.sac_q_forward(
            next_obs,
            next_actions,
            detach_encoder=detach_encoder,
        )
        return data_q, next_q

    @torch.inference_mode()
    def predict_action_batch(
        self,
        env_obs,
        calculate_logprobs=True,
        calculate_values=True,
        return_obs=True,
        mode="train",
        **kwargs,
    ):
        del calculate_logprobs, calculate_values, kwargs
        obs = self.preprocess_env_obs(env_obs)
        action, log_prob, _ = self.sac_forward(
            obs,
            deterministic=(mode == "eval"),
            apply_action_noise=(mode != "eval"),
        )
        chunk_actions = self._format_chunk_actions(action)
        forward_inputs = {"action": action, "model_action": action}
        if return_obs:
            forward_inputs.update(obs)
        result = {
            "prev_logprobs": log_prob,
            "prev_values": torch.zeros_like(action[..., :1]),
            "forward_inputs": forward_inputs,
        }
        return chunk_actions, result

    def set_critic_requires_grad(self, requires_grad: bool) -> None:
        for parameter in self.q_head.parameters():
            parameter.requires_grad_(requires_grad)
