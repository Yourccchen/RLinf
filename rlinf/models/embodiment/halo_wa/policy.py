# Licensed under the Apache License, Version 2.0.
"""Expose the official HALO-WA heads through the existing RLT policy API."""

from collections.abc import Mapping

import torch
from torch import nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.halo_wa import attention


class VisualActor(nn.Module):
    """Own visual projection and actor weights so rollout publication is atomic."""

    def __init__(
        self,
        visual_dim,
        hidden_dim,
        num_heads,
        max_visual_tokens,
        proprio_dim,
        action_dim,
        chunk_len,
    ):
        super().__init__()
        self.visual_compressor = nn.Sequential(
            nn.Linear(visual_dim, hidden_dim), nn.GELU(), nn.LayerNorm(hidden_dim)
        )
        self.visual_pos_embed = nn.Parameter(
            torch.zeros(1, max_visual_tokens, hidden_dim)
        )
        nn.init.normal_(self.visual_pos_embed, mean=0.0, std=0.02)
        self.head = attention.CrossAttentionActor(
            hidden_dim, num_heads, 0.0, proprio_dim, action_dim, chunk_len
        )
        attention.init_actor_output_small(self.head)

    def encode_visual(self, tokens: torch.Tensor) -> torch.Tensor:
        """Project each spatial token and retain its ordered position."""
        return (
            self.visual_compressor(tokens) + self.visual_pos_embed[:, : tokens.shape[1]]
        )

    def forward(self, tokens, proprio, reference):
        raw, _ = self.head(self.encode_visual(tokens), proprio, reference)
        # Existing Songling codec maps physical action bounds to [-1, 1].
        return raw.tanh()


class HaloWAPolicy(nn.Module, BasePolicy):
    """Independent attention model; MLP classes and checkpoints stay unchanged."""

    def __init__(
        self,
        visual_dim: int,
        proprio_dim: int = 14,
        action_dim: int = 14,
        num_action_chunks: int = 12,
        hidden_dim: int = 512,
        num_heads: int = 8,
        max_visual_tokens: int = 4096,
    ):
        super().__init__()
        if (
            min(
                visual_dim,
                proprio_dim,
                action_dim,
                num_action_chunks,
                hidden_dim,
                num_heads,
                max_visual_tokens,
            )
            < 1
            or hidden_dim % num_heads
        ):
            raise ValueError(
                "Require positive dimensions and hidden_dim divisible by num_heads"
            )
        self.visual_dim = int(visual_dim)
        self.max_visual_tokens = int(max_visual_tokens)
        self.proprio_dim = int(proprio_dim)
        self.action_dim = int(action_dim)
        self.chunk_len = self.ref_chunk_len = self.num_action_chunks = int(
            num_action_chunks
        )
        self.actor = VisualActor(
            visual_dim,
            hidden_dim,
            num_heads,
            max_visual_tokens,
            proprio_dim,
            action_dim,
            self.chunk_len,
        )
        self.q_head = attention.CrossAttentionCritic(
            hidden_dim, num_heads, 0.0, proprio_dim, action_dim
        )

    def _inputs(self, obs: Mapping[str, torch.Tensor]):
        param = next(self.parameters())
        z = obs["z_rl"].to(device=param.device, dtype=param.dtype)
        p = obs["proprio"].to(device=param.device, dtype=param.dtype)
        ref = obs["ref_chunk"].to(device=param.device, dtype=param.dtype)
        if (
            z.ndim != 3
            or z.shape[-1] != self.visual_dim
            or not 0 < z.shape[1] <= self.max_visual_tokens
        ):
            raise ValueError(
                "Invalid visual tokens: expected [B,N,visual_dim] within max_visual_tokens"
            )
        if p.shape != (z.shape[0], self.proprio_dim):
            raise ValueError("Invalid proprio shape")
        if (
            ref.ndim != 3
            or ref.shape[0] != z.shape[0]
            or ref.shape[1] < self.chunk_len
            or ref.shape[2] != self.action_dim
        ):
            raise ValueError("Reference must cover the action chunk")
        if not all(torch.isfinite(t).all() for t in (z, p, ref)):
            raise ValueError("HALO-WA observations must be finite")
        return z, p, ref[:, : self.chunk_len]

    def encode_visual(self, obs):
        """Encode with current projection, also shared by the target heads."""
        return self.actor.encode_visual(self._inputs(obs)[0])

    def sac_forward(self, obs, visual_features=None, **kwargs):
        z, p, ref = self._inputs(obs)
        if visual_features is None:
            action = self.actor(z, p, ref).flatten(1)
        else:
            action = self.actor.head(visual_features, p, ref)[0].tanh().flatten(1)
        return action, torch.zeros_like(action), None

    def sac_q_forward(
        self, obs, actions, detach_encoder=False, visual_features=None, **kwargs
    ):
        z, p, ref = self._inputs(obs)
        visual = (
            self.actor.encode_visual(z) if visual_features is None else visual_features
        )
        if detach_encoder:
            visual = visual.detach()
        actions = actions.reshape(z.shape[0], self.chunk_len, self.action_dim)
        q1, q2, _ = self.q_head(visual, p, ref, actions)
        return torch.cat((q1, q2), dim=-1)

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        return self.default_forward(**kwargs)

    def default_forward(self, **kwargs):
        raise NotImplementedError("HALO-WA uses the SsEval actor/critic entrypoints")

    @torch.inference_mode()
    def predict_action_batch(self, env_obs, mode="eval", return_obs=True, **kwargs):
        action, logs, _ = self.sac_forward(env_obs)
        inputs = dict(action=action, model_action=action)
        if return_obs:
            inputs.update(env_obs)
        return action.reshape(-1, self.chunk_len, self.action_dim), dict(
            prev_logprobs=logs,
            prev_values=torch.zeros_like(action[:, :1]),
            forward_inputs=inputs,
        )

    def set_critic_requires_grad(self, requires_grad: bool):
        for p in self.q_head.parameters():
            p.requires_grad_(requires_grad)
