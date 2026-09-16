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

import torch

from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Worker
from rlinf.workers.actor.fsdp_rlt_ac_policy_worker import (
    RLTACFSDPPolicy,
    RLTACLossMixin,
)


class RLTTD3LossMixin(RLTACLossMixin):
    """Ablation-style TD3 actor objective over current RLT replay fields."""

    @staticmethod
    def _is_rollout_parameter(name: str) -> bool:
        return name == "actor" or name.startswith("actor.")

    def setup_model_and_optimizer(self, initialize_target=False) -> None:
        super().setup_model_and_optimizer(initialize_target=initialize_target)
        self.param_names_need_sync = [
            name
            for name in self.param_names_need_sync
            if self._is_rollout_parameter(name)
        ]
        if not self.param_names_need_sync:
            raise RuntimeError(
                "RLT TD3 Actor has no parameters selected for rollout sync."
            )

    def _next_actions_for_critic_target(self, next_obs):
        # RLT uses the current actor with stop-gradient and target critics.
        # The surrounding critic-target block runs under torch.no_grad().
        return self.model(
            forward_type=ForwardType.SAC,
            apply_reference_dropout=False,
            apply_action_noise=bool(
                self.cfg.algorithm.get("target_action_noise", True)
            ),
            obs=next_obs,
        )

    def _clip_optimizer_grad_norm(self, optimizer, max_norm: float):
        # FSDP.clip_grad_norm_ performs the required cross-rank reduction for
        # sharded parameters. Temporarily hide gradients owned by the other
        # optimizer so actor and critic are clipped independently.
        selected_ids = {
            id(param) for group in optimizer.param_groups for param in group["params"]
        }
        hidden_grads = []
        for param in self.model.parameters():
            if param.grad is not None and id(param) not in selected_ids:
                hidden_grads.append((param, param.grad))
                param.grad = None
        try:
            if not any(
                param.grad is not None
                for group in optimizer.param_groups
                for param in group["params"]
            ):
                return torch.zeros((), device=self.device)
            return self.model.clip_grad_norm_(max_norm=max_norm)
        finally:
            for param, grad in hidden_grads:
                param.grad = grad

    def _clip_critic_grad_norm(self, max_norm: float):
        return self._clip_optimizer_grad_norm(self.qf_optimizer, max_norm)

    def _clip_actor_grad_norm(self, max_norm: float):
        self.qf_optimizer.zero_grad(set_to_none=True)
        return self._clip_optimizer_grad_norm(self.optimizer, max_norm)

    def _human_mask(
        self,
        intervene_flags: torch.Tensor | None,
        *,
        batch_size: int,
        chunk_len: int,
        action_dim: int,
        device: torch.device,
    ) -> torch.Tensor:
        if intervene_flags is None:
            return torch.zeros(
                (batch_size, chunk_len),
                dtype=torch.bool,
                device=device,
            )

        flags = self._flatten_chunk(intervene_flags).to(device=device).bool()
        if flags.shape[-1] == chunk_len:
            return flags.reshape(batch_size, chunk_len)
        return flags.reshape(batch_size, chunk_len, action_dim).any(dim=-1)

    def _td3_bc_metrics(
        self,
        pi: torch.Tensor,
        actions: torch.Tensor,
        ref_chunk: torch.Tensor,
        intervene_flags: torch.Tensor | None,
        valid_step_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        chunk_len, action_dim = self._chunk_shape()
        pi_chunk = self._flatten_chunk(pi).reshape(-1, chunk_len, action_dim)
        action_chunk = self._flatten_chunk(actions).reshape(-1, chunk_len, action_dim)
        ref_chunk = self._flatten_chunk(ref_chunk).reshape(-1, chunk_len, action_dim)

        human_mask = self._human_mask(
            intervene_flags,
            batch_size=pi_chunk.shape[0],
            chunk_len=chunk_len,
            action_dim=action_dim,
            device=pi_chunk.device,
        )
        if valid_step_mask is None:
            valid_mask = torch.ones_like(human_mask)
        else:
            valid_mask = valid_step_mask.to(
                device=pi_chunk.device, dtype=torch.bool
            ).reshape_as(human_mask)
        human_mask = human_mask & valid_mask
        bc_target = torch.where(human_mask[..., None], action_chunk, ref_chunk)
        bc_error = torch.mean(torch.square(pi_chunk - bc_target), dim=-1)
        bc_loss = torch.sum(bc_error * valid_mask.to(bc_error.dtype)) / torch.clamp(
            torch.sum(valid_mask.to(bc_error.dtype)), min=1.0
        )

        policy_mask = valid_mask & ~human_mask
        ref_error = torch.mean(torch.square(pi_chunk - ref_chunk), dim=-1)
        human_error = torch.mean(torch.square(pi_chunk - action_chunk), dim=-1)
        bc_ref = torch.sum(ref_error * policy_mask.to(ref_error.dtype)) / torch.clamp(
            torch.sum(policy_mask.to(ref_error.dtype)), min=1.0
        )
        bc_human = torch.sum(
            human_error * human_mask.to(human_error.dtype)
        ) / torch.clamp(torch.sum(human_mask.to(human_error.dtype)), min=1.0)

        human_ratio = torch.mean(human_mask.to(torch.float32)).item()
        metrics = {
            "bc_loss": bc_loss.detach().item(),
            "bc_ref_loss": bc_ref.detach().item(),
            "bc_human_loss": bc_human.detach().item(),
            "human_mask_ratio": human_ratio,
            "policy_mask_ratio": 1.0 - human_ratio,
        }
        return bc_loss, metrics

    def _td3_actor_q(self, all_q_values: torch.Tensor) -> torch.Tensor:
        # Q1 supplies the policy gradient; twin-min is only for bootstrap.
        return self._q1(all_q_values)

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        if getattr(self, "qf_optimizer", None) is not None:
            self.qf_optimizer.zero_grad(set_to_none=True)

        curr_obs = batch["curr_obs"]
        reference_dropout_prob = float(
            self.cfg.algorithm.get("reference_dropout_prob", 0.0)
        )
        pi, log_pi, _ = self.model(
            forward_type=ForwardType.SAC,
            obs=curr_obs,
            apply_reference_dropout=reference_dropout_prob > 0.0,
            reference_dropout_prob=reference_dropout_prob,
            apply_action_noise=bool(
                self.cfg.algorithm.get("actor_update_action_noise", True)
            ),
        )
        if log_pi.ndim == 1:
            log_pi = log_pi.unsqueeze(-1)
        log_pi = log_pi.sum(dim=-1, keepdim=True)

        # Keep the FSDP-wrapped model's parameter trainability stable across
        # the actor and critic forwards. Dynamically toggling critic
        # ``requires_grad`` here can leave FSDP post-backward hooks in an
        # invalid state, while the actor optimizer still excludes critic params.
        all_qf_pi = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=curr_obs,
            actions=pi,
            detach_encoder=True,
        )

        num_q_values = all_qf_pi.shape[-1]
        metrics = {
            f"q_value_{q_id}": all_qf_pi[..., q_id].mean().item()
            for q_id in range(num_q_values)
        }
        qf_pi = self._td3_actor_q(all_qf_pi)
        metrics["q_pi"] = qf_pi.mean().item()

        ref_chunk = self._ref_chunk(curr_obs)
        bc_loss, td3_metrics = self._td3_bc_metrics(
            pi=pi,
            actions=batch["actions"],
            ref_chunk=ref_chunk,
            intervene_flags=batch.get("intervene_flags", None),
            valid_step_mask=self._valid_step_mask(batch, batch["rewards"]),
        )
        metrics.update(td3_metrics)

        bc_weight, q_weight, weight_metrics = self._actor_objective_weights()
        actor_loss = -q_weight * qf_pi.mean() + bc_weight * bc_loss
        metrics.update(weight_metrics)
        metrics["weighted_q"] = (q_weight * qf_pi.mean()).detach().item()
        metrics["weighted_bc"] = (bc_weight * bc_loss).detach().item()
        metrics["action_ref_abs_mean"] = (
            (self._flatten_chunk(pi) - self._flatten_chunk(ref_chunk))
            .abs()
            .mean()
            .detach()
            .item()
        )
        metrics["reference_dropout_prob"] = reference_dropout_prob

        entropy = -log_pi.mean()
        return actor_loss, entropy, metrics


class RLTTD3FSDPPolicy(RLTTD3LossMixin, RLTACFSDPPolicy):
    """Synchronous RLT TD3 worker using the current RLT rollout data flow."""
