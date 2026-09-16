# Copyright 2026 The RLinf Authors.

"""In-process RLT runtime called by RealWorldInference's RlinfPolicy."""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.rollout import (
    predict_rlt_candidates,
    validate_rlt_stage2_configs,
)
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.envs.remote_songling import SonglingActionCodec
from rlinf.models import get_model
from rlinf.serving.sseval_contract import (
    DualActionCandidates,
    TransitionFeedback,
)
from rlinf.serving.sseval_learner import (
    InProcessLearnerConfig,
    InProcessRLTTD3Learner,
)


@dataclass
class _PendingInference:
    episode_id: str
    chunk_id: int
    replay_obs: dict[str, torch.Tensor]
    actor_version: int


def _single_observation(observation: Mapping[str, Any], instruction: str) -> dict:
    state = torch.as_tensor(observation["states"], dtype=torch.float32)
    main = torch.as_tensor(observation["main_images"])
    wrist = torch.as_tensor(observation["wrist_images"])
    if state.shape == (14,):
        state = state.unsqueeze(0)
    if main.ndim == 3:
        main = main.unsqueeze(0)
    if wrist.ndim == 4:
        wrist = wrist.unsqueeze(0)
    if state.shape != (1, 14):
        raise ValueError(f"SsEval Songling state must be [1,14], got {state.shape}.")
    if main.ndim != 4 or main.shape[0] != 1:
        raise ValueError(f"main_images must be [1,H,W,3], got {main.shape}.")
    if wrist.ndim != 5 or wrist.shape[:2] != (1, 2):
        raise ValueError(f"wrist_images must be [1,2,H,W,3], got {wrist.shape}.")
    return {
        "states": state,
        "main_images": main,
        "wrist_images": wrist,
        "task_descriptions": [
            str(observation.get("task_description") or instruction).strip()
        ],
    }


def _unbatch_replay_obs(obs: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result = {}
    for key, value in obs.items():
        value = value.detach().to(torch.float32).cpu()
        if value.ndim > 1 and value.shape[0] == 1:
            value = value[0]
        result[key] = value.contiguous()
    return result


def _build_feedback_trajectory(
    pending: _PendingInference,
    next_obs: Mapping[str, torch.Tensor],
    feedback: TransitionFeedback,
    codec: SonglingActionCodec,
) -> Trajectory:
    normalized_actions = torch.as_tensor(
        codec.encode(feedback.executed_actions, clip=True), dtype=torch.float32
    )
    rewards = torch.from_numpy(feedback.rewards).to(torch.float32)
    terminated = torch.from_numpy(feedback.terminated).bool()
    truncated = torch.from_numpy(feedback.truncated).bool()
    valid = torch.from_numpy(feedback.valid_step_mask).bool()
    intervene = torch.from_numpy(feedback.intervene_flags).bool()
    curr = _unbatch_replay_obs(pending.replay_obs)
    nxt = _unbatch_replay_obs(next_obs)
    return Trajectory(
        max_episode_length=1,
        model_weights_id=f"sseval_actor_{pending.actor_version}",
        actions=normalized_actions.reshape(1, 1, -1),
        intervene_flags=intervene.reshape(1, 1, -1),
        rewards=rewards.reshape(1, 1, -1),
        terminations=terminated.reshape(1, 1, -1),
        truncations=truncated.reshape(1, 1, -1),
        dones=(terminated | truncated).reshape(1, 1, -1),
        versions=torch.full((1, 1, 1), float(pending.actor_version)),
        forward_inputs={
            "action": normalized_actions.reshape(1, 1, -1),
            "valid_step_mask": valid.reshape(1, 1, -1),
            "record_transition": torch.ones((1, 1, 1), dtype=torch.bool),
            "actor_switch": torch.full(
                (1, 1, 1),
                feedback.selected_mode.value == "actor",
                dtype=torch.bool,
            ),
        },
        curr_obs={
            key: value.reshape(1, 1, *value.shape) for key, value in curr.items()
        },
        next_obs={key: value.reshape(1, 1, *value.shape) for key, value in nxt.items()},
    )


class SsEvalRLTRuntime:
    """Serial inference facade with a non-blocking asynchronous TD3 learner."""

    def __init__(
        self,
        *,
        feature_model: Any,
        active_policy_model: torch.nn.Module,
        action_codec: SonglingActionCodec,
        learner: InProcessRLTTD3Learner,
        reference_seed: int = 2026,
    ) -> None:
        self.feature_model = feature_model.eval().requires_grad_(False)
        self.active_policy_model = active_policy_model.eval().requires_grad_(False)
        self.action_codec = action_codec
        self.learner = learner
        self.reference_seed = int(reference_seed)
        self._seed_index = 0
        self._episode_id: str | None = None
        self._instruction = ""
        self._next_chunk_id = 0
        self._pending: _PendingInference | None = None
        self._seen_feedback: set[tuple[str, int]] = set()
        self._active_actor_version = 0
        self._episode_actor_ready = False
        self._lock = threading.RLock()
        self._closed = False

    @classmethod
    def from_config(cls, config_path: str | Path) -> "SsEvalRLTRuntime":
        cfg = OmegaConf.load(str(config_path))
        OmegaConf.resolve(cfg)
        validate_rlt_stage2_configs(cfg.actor_model, cfg.feature_model)
        device = torch.device(str(cfg.get("device", "cuda")))
        feature_model = get_model(copy.deepcopy(cfg.feature_model)).to(device).eval()
        active_model = get_model(copy.deepcopy(cfg.actor_model)).to(device).eval()
        actor_checkpoint = cfg.get("actor_checkpoint", None)
        if actor_checkpoint:
            state = torch.load(str(actor_checkpoint), map_location=device)
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            active_model.load_state_dict(state, strict=True)
        codec = SonglingActionCodec.from_config(cfg.action_codec)
        learner_model = copy.deepcopy(active_model).train().requires_grad_(True)
        learner_cfg = InProcessLearnerConfig.from_mapping(
            OmegaConf.to_container(cfg.get("learner", {}), resolve=True)
        )
        learner = InProcessRLTTD3Learner(learner_model, learner_cfg)
        return cls(
            feature_model=feature_model,
            active_policy_model=active_model,
            action_codec=codec,
            learner=learner,
            reference_seed=int(cfg.get("reference_seed", 2026)),
        )

    @property
    def active_actor_version(self) -> int:
        return self._active_actor_version

    def start_episode(
        self,
        session_id: str,
        instruction: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        with self._lock:
            self._require_open()
            if self._episode_id is not None:
                raise RuntimeError("An SsEval RLT episode is already active.")
            episode_id = str((metadata or {}).get("episode_id") or session_id).strip()
            if not episode_id:
                raise ValueError("session_id or metadata.episode_id must be non-empty.")
            instruction = str(instruction).strip()
            if not instruction:
                raise ValueError("SsEval RLT instruction must be non-empty.")
            self._activate_candidate_actor()
            self._episode_id = episode_id
            self._instruction = instruction
            self._next_chunk_id = 0
            self._pending = None
            self._seen_feedback.clear()
            self._episode_actor_ready = bool(self.learner.actor_ready)
            return episode_id

    def set_instruction(self, instruction: str) -> None:
        instruction = str(instruction).strip()
        if not instruction:
            raise ValueError("SsEval RLT instruction must be non-empty.")
        with self._lock:
            self._require_open()
            if self._episode_id is None:
                raise RuntimeError("No active episode for instruction update.")
            self._instruction = instruction

    @torch.no_grad()
    def infer_candidates(
        self,
        observation: Mapping[str, Any],
        transition_feedback: Mapping[str, Any] | None = None,
    ) -> DualActionCandidates:
        with self._lock:
            self._require_open()
            self.learner.raise_if_failed()
            if self._episode_id is None:
                raise RuntimeError("start_episode() must be called before inference.")
            env_obs = _single_observation(observation, self._instruction)
            chunk_id = self._next_chunk_id
            seed = self.reference_seed + self._seed_index
            candidates, replay_obs = predict_rlt_candidates(
                policy_model=self.active_policy_model,
                feature_model=self.feature_model,
                action_codec=self.action_codec,
                env_obs=env_obs,
                episode_id=self._episode_id,
                chunk_id=chunk_id,
                actor_ready=self._episode_actor_ready,
                actor_version=self._active_actor_version,
                reference_seed=seed,
            )
            feedback = (
                TransitionFeedback.from_mapping(transition_feedback)
                if transition_feedback is not None
                else None
            )
            if feedback is not None:
                self._ingest_feedback(feedback, replay_obs)
            if feedback is not None and feedback.done:
                self._episode_id = None
                self._instruction = ""
                self._pending = None
            else:
                self._pending = _PendingInference(
                    episode_id=self._episode_id,
                    chunk_id=chunk_id,
                    replay_obs=replay_obs,
                    actor_version=self._active_actor_version,
                )
            self._next_chunk_id += 1
            self._seed_index += 1
            return candidates

    def _ingest_feedback(
        self, feedback: TransitionFeedback, next_obs: Mapping[str, torch.Tensor]
    ) -> bool:
        if feedback.key in self._seen_feedback:
            return False
        pending = self._pending
        if pending is None:
            raise ValueError("Received transition feedback without a pending action.")
        if feedback.key != (pending.episode_id, pending.chunk_id):
            raise ValueError(
                "Out-of-order transition feedback: expected "
                f"{(pending.episode_id, pending.chunk_id)}, got {feedback.key}."
            )
        trajectory = _build_feedback_trajectory(
            pending, next_obs, feedback, self.action_codec
        )
        self.learner.submit(trajectory)
        self._seen_feedback.add(feedback.key)
        return True

    def end_episode(self, final_feedback: Mapping[str, Any]) -> bool:
        with self._lock:
            self._require_open()
            if self._episode_id is None or self._pending is None:
                raise RuntimeError("No active SsEval RLT episode to end.")
            feedback = TransitionFeedback.from_mapping(final_feedback)
            if not feedback.done:
                raise ValueError("end_episode requires terminal or truncated feedback.")
            ingested = self._ingest_feedback(feedback, self._pending.replay_obs)
            self._episode_id = None
            self._instruction = ""
            self._pending = None
            return ingested

    def reset(self) -> None:
        with self._lock:
            self._episode_id = None
            self._instruction = ""
            self._pending = None
            self._next_chunk_id = 0
            self._seen_feedback.clear()

    def get_wire_config(self) -> dict[str, Any]:
        return {
            "video_action_rate": 1,
            "video_length": 1,
            "image_format": "jpeg",
            "jpeg_quality": 95,
        }

    @torch.no_grad()
    def _activate_candidate_actor(self) -> bool:
        candidate = self.learner.take_candidate()
        if candidate is None:
            return False
        version, state = candidate
        if any(not torch.isfinite(value).all() for value in state.values()):
            raise RuntimeError("Candidate Actor contains non-finite parameters.")
        device_state = {
            key: value.to(next(self.active_policy_model.parameters()).device)
            for key, value in state.items()
        }
        self.active_policy_model.actor.load_state_dict(device_state, strict=True)
        self._active_actor_version = int(version)
        return True

    def save_checkpoint(self, path: str | Path) -> None:
        with self._lock:
            self._require_open()
            self.learner.save_checkpoint(path)

    def load_checkpoint(self, path: str | Path) -> None:
        with self._lock:
            self._require_open()
            if self._episode_id is not None:
                raise RuntimeError("Checkpoint restore requires an episode boundary.")
            self.learner.load_checkpoint(path)
            self._activate_candidate_actor()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("SsEval RLT runtime is closed.")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.learner.close()
