# Copyright 2026 The RLinf Authors.

"""Single-process asynchronous TD3 learner for the RWI Policy integration."""

from __future__ import annotations

import copy
import queue
import threading
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from rlinf.algorithms.rlt.transition import overwrite_rlt_ref_with_human
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.data.storage.replay import TrajectoryReplayBuffer
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.logging import get_logger

logger = get_logger()

# Reported on every log line even when the step skipped the delayed actor update.
_ACTOR_METRIC_KEYS = (
    "actor_loss",
    "bc_loss",
    "q_pi",
    "bc_weight",
    "q_weight",
    "residual_abs_mean",
    "residual_smoothness_loss",
)


def make_sseval_run_save_dir(save_dir: str | Path, *, stamp: str | None = None) -> Path:
    """Create a unique timestamped run directory under ``save_dir``.

    Autosave writes ``<save_dir>/<YYYYMMDD_HHMMSS>/step_N`` so a new Policy
    process cannot overwrite an earlier run's ``step_N``.
    """
    root = Path(save_dir).expanduser()
    stamp = stamp or time.strftime("%Y%m%d_%H%M%S")
    path = root / stamp
    suffix = 1
    while path.exists():
        path = root / f"{stamp}_{suffix}"
        suffix += 1
    path.mkdir(parents=True, exist_ok=False)
    return path


def replay_catch_up_updates(num_samples: int, *, min_buffer_size: int, utd: int) -> int:
    """Return critic updates implied by live ``min_buffer_size`` + ``utd`` collection.

    Live collection trains only after the buffer reaches ``min_buffer_size``.
    Each later transition runs ``utd`` critic updates. Loading replay files
    skips that loop, so restore uses the same count:
    ``(n - min_buffer_size + 1) * utd`` when ``n >= min_buffer_size``.
    """
    samples = int(num_samples)
    threshold = int(min_buffer_size)
    ratio = int(utd)
    if samples < threshold or threshold <= 0 or ratio <= 0:
        return 0
    return (samples - threshold + 1) * ratio


@dataclass(frozen=True)
class InProcessLearnerConfig:
    batch_size: int = 256
    min_buffer_size: int = 1000
    replay_capacity: int = 50000
    utd: int = 5
    gamma: float = 0.99
    tau: float = 0.005
    actor_update_interval: int = 2
    actor_lr: float = 1.0e-4
    critic_lr: float = 1.0e-4
    actor_clip_grad: float = 10.0
    critic_clip_grad: float = 10.0
    reference_dropout_prob: float = 0.5
    target_action_noise: bool = True
    actor_update_action_noise: bool = True
    residual_velocity_weight: float = 0.0
    residual_acceleration_weight: float = 0.0
    warmup_updates: int = 5000
    warmup_bc_weight: float = 7.0
    warmup_q_weight: float = 0.05
    online_bc_weight: float = 2.8
    online_q_weight: float = 0.5
    ramp_updates: int = 20000
    queue_size: int = 1024
    seed: int = 1234
    # Every N critic updates, emit one training metrics line. 0 disables it.
    log_interval: int = 100

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None):
        value = dict(value or {})
        known = cls.__dataclass_fields__
        unknown = sorted(set(value) - set(known))
        if unknown:
            raise ValueError(f"Unknown SsEval learner settings: {unknown}")
        result = cls(**value)
        for name in (
            "batch_size",
            "min_buffer_size",
            "replay_capacity",
            "utd",
            "actor_update_interval",
            "queue_size",
        ):
            if int(getattr(result, name)) <= 0:
                raise ValueError(f"{name} must be positive.")
        for name in (
            "residual_velocity_weight",
            "residual_acceleration_weight",
        ):
            if float(getattr(result, name)) < 0:
                raise ValueError(f"{name} must be non-negative.")
        if int(result.log_interval) < 0:
            raise ValueError("log_interval must be non-negative.")
        return result


def residual_temporal_losses(
    predicted_chunk: torch.Tensor,
    reference_chunk: torch.Tensor,
    valid_step_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return first/second-order smoothness losses on the effective residual."""
    if predicted_chunk.shape != reference_chunk.shape:
        raise ValueError(
            "predicted/reference chunks must match, got "
            f"{predicted_chunk.shape} and {reference_chunk.shape}."
        )
    if predicted_chunk.ndim != 3:
        raise ValueError(
            f"action chunks must be [B,C,A], got {predicted_chunk.shape}."
        )
    valid = valid_step_mask.to(
        device=predicted_chunk.device, dtype=torch.bool
    ).reshape(predicted_chunk.shape[0], predicted_chunk.shape[1])
    residual = predicted_chunk - reference_chunk
    zero = residual.sum() * 0.0

    if residual.shape[1] < 2:
        return zero, zero
    velocity_error = (residual[:, 1:] - residual[:, :-1]).square().mean(dim=-1)
    velocity_mask = valid[:, 1:] & valid[:, :-1]
    velocity_loss = (velocity_error * velocity_mask).sum() / velocity_mask.sum().clamp_min(
        1
    )

    if residual.shape[1] < 3:
        return velocity_loss, zero
    acceleration_error = (
        residual[:, 2:] - 2.0 * residual[:, 1:-1] + residual[:, :-2]
    ).square().mean(dim=-1)
    acceleration_mask = valid[:, 2:] & valid[:, 1:-1] & valid[:, :-2]
    acceleration_loss = (
        acceleration_error * acceleration_mask
    ).sum() / acceleration_mask.sum().clamp_min(1)
    return velocity_loss, acceleration_loss


def _to_device(value: Any, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _to_device(item, device) for key, item in value.items()}
    return value


class InProcessRLTTD3Learner:
    """Owns train/target models; publishes immutable Actor snapshots."""

    def __init__(
        self,
        model: torch.nn.Module,
        config: InProcessLearnerConfig | Mapping[str, Any] | None = None,
        *,
        start_background: bool = True,
        save_interval: int = 0,
        save_dir: str | Path | None = None,
    ) -> None:
        self.config = (
            config
            if isinstance(config, InProcessLearnerConfig)
            else InProcessLearnerConfig.from_mapping(config)
        )
        self.save_interval = int(save_interval)
        if self.save_interval < 0:
            raise ValueError(f"save_interval must be >= 0, got {self.save_interval}.")
        if self.save_interval > 0 and not save_dir:
            raise ValueError("save_dir is required when save_interval > 0.")
        if save_dir and self.save_interval > 0:
            self.save_dir = make_sseval_run_save_dir(save_dir)
            logger.info("SsEval checkpoints will be written under %s", self.save_dir)
        elif save_dir:
            self.save_dir = Path(save_dir).expanduser()
        else:
            self.save_dir = None
        self._last_saved_step = 0
        self.model = model
        self.device = next(model.parameters()).device
        self.target_model = copy.deepcopy(model).requires_grad_(False).eval()
        self.actor_optimizer = torch.optim.Adam(
            self.model.actor.parameters(), lr=self.config.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.model.q_head.parameters(), lr=self.config.critic_lr
        )
        self.replay = TrajectoryReplayBuffer(
            seed=self.config.seed,
            enable_cache=True,
            cache_size=min(self.config.replay_capacity, 5000),
            sample_window_size=self.config.replay_capacity,
            max_num_samples=self.config.replay_capacity,
            auto_save=False,
        )
        self.update_step = 0
        self.total_transitions = 0
        self._queue: queue.Queue[Trajectory | None] = queue.Queue(
            maxsize=self.config.queue_size
        )
        self._candidate_lock = threading.Lock()
        self._train_lock = threading.RLock()
        self._background_error: BaseException | None = None
        self._candidate_actor_state: dict[str, torch.Tensor] | None = None
        self._candidate_version: int | None = None
        self._last_metrics: dict[str, float] = {}
        self._last_actor_metrics: dict[str, float] = {}
        self._closed = False
        self._thread: threading.Thread | None = None
        if start_background:
            self._thread = threading.Thread(
                target=self._run, name="sseval-rlt-learner", daemon=True
            )
            self._thread.start()

    @property
    def actor_ready(self) -> bool:
        return self.update_step >= self.config.warmup_updates

    @property
    def last_metrics(self) -> dict[str, float]:
        return dict(self._last_metrics)

    def raise_if_failed(self) -> None:
        if self._background_error is not None:
            raise RuntimeError("SsEval RLT learner failed.") from self._background_error

    def submit(self, trajectory: Trajectory) -> None:
        self.raise_if_failed()
        if self._closed:
            raise RuntimeError("SsEval RLT learner is closed.")
        try:
            self._queue.put_nowait(trajectory)
        except queue.Full as exc:
            raise RuntimeError(
                "SsEval RLT learner queue is full; refusing to drop transition."
            ) from exc

    def _run(self) -> None:
        while True:
            trajectory = self._queue.get()
            try:
                if trajectory is None:
                    return
                self.add_and_train(trajectory)
            except BaseException as exc:
                self._background_error = exc
                return
            finally:
                self._queue.task_done()

    def add_and_train(self, trajectory: Trajectory) -> None:
        with self._train_lock:
            self._write_human_actions_into_ref(trajectory)
            self.replay.add_trajectories([trajectory])
            self.total_transitions += int(trajectory.rewards.shape[0])
            if self.replay.total_samples < self.config.min_buffer_size:
                return
            for _ in range(self.config.utd):
                self.train_once()
                self._maybe_save()

    def _objective_weights(self) -> tuple[float, float]:
        cfg = self.config
        if self.update_step < cfg.warmup_updates:
            return cfg.warmup_bc_weight, cfg.warmup_q_weight
        if cfg.ramp_updates <= 0:
            return cfg.online_bc_weight, cfg.online_q_weight
        progress = min(
            max((self.update_step - cfg.warmup_updates + 1) / cfg.ramp_updates, 0.0),
            1.0,
        )
        bc = cfg.warmup_bc_weight + progress * (
            cfg.online_bc_weight - cfg.warmup_bc_weight
        )
        q = cfg.warmup_q_weight + progress * (cfg.online_q_weight - cfg.warmup_q_weight)
        return bc, q

    @staticmethod
    def _write_human_actions_into_ref(trajectory: Trajectory) -> None:
        obs = trajectory.curr_obs
        if obs is None or "ref_chunk" not in obs:
            return
        obs["ref_chunk"] = overwrite_rlt_ref_with_human(
            obs["ref_chunk"],
            trajectory.actions,
            trajectory.intervene_flags,
        )

    def train_once(self) -> dict[str, float]:
        batch = _to_device(self.replay.sample(self.config.batch_size), self.device)
        batch["curr_obs"]["ref_chunk"] = overwrite_rlt_ref_with_human(
            batch["curr_obs"]["ref_chunk"],
            batch["actions"],
            batch.get("intervene_flags"),
        )
        # Sampling clamps to the replay size, so this can be below batch_size
        # while the buffer is still filling up.
        sampled_batch_size = int(batch["rewards"].shape[0])
        rewards = batch["rewards"].reshape(sampled_batch_size, -1).float()
        valid = batch["forward_inputs"]["valid_step_mask"].reshape_as(rewards).bool()
        steps = torch.arange(rewards.shape[-1], device=self.device, dtype=rewards.dtype)
        discounts = torch.pow(
            torch.as_tensor(self.config.gamma, device=self.device), steps
        )
        discounted_reward = (rewards * valid * discounts).sum(dim=-1, keepdim=True)
        done = (
            (batch["terminations"].reshape_as(valid).bool())
            | (batch["truncations"].reshape_as(valid).bool())
        ) & valid
        bootstrap = (~done.any(dim=-1, keepdim=True)) & valid.all(dim=-1, keepdim=True)

        with torch.no_grad():
            next_action, _, _ = self.model(
                forward_type=ForwardType.SAC,
                obs=batch["next_obs"],
                apply_reference_dropout=False,
                apply_action_noise=self.config.target_action_noise,
            )
            next_q = (
                self.target_model(
                    forward_type=ForwardType.SAC_Q,
                    obs=batch["next_obs"],
                    actions=next_action,
                )
                .min(dim=-1, keepdim=True)
                .values
            )
            target_q = (
                discounted_reward
                + (self.config.gamma ** rewards.shape[-1]) * bootstrap.float() * next_q
            )

        q_values = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=batch["curr_obs"],
            actions=batch["actions"],
        )
        critic_loss = F.mse_loss(q_values, target_q.expand_as(q_values))
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        critic_norm = torch.nn.utils.clip_grad_norm_(
            self.model.q_head.parameters(), self.config.critic_clip_grad
        )
        self.critic_optimizer.step()

        actor_loss_value = float("nan")
        bc_loss_value = float("nan")
        q_pi_value = float("nan")
        bc_weight_value = float("nan")
        q_weight_value = float("nan")
        residual_abs_value = float("nan")
        residual_velocity_value = float("nan")
        residual_acceleration_value = float("nan")
        residual_smoothness_value = float("nan")
        actor_updated = (self.update_step + 1) % self.config.actor_update_interval == 0
        if actor_updated:
            predicted, _, _ = self.model(
                forward_type=ForwardType.SAC,
                obs=batch["curr_obs"],
                apply_reference_dropout=self.config.reference_dropout_prob > 0,
                reference_dropout_prob=self.config.reference_dropout_prob,
                apply_action_noise=self.config.actor_update_action_noise,
            )
            q1 = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=batch["curr_obs"],
                actions=predicted,
                detach_encoder=True,
            )[..., :1]
            chunk_len = rewards.shape[-1]
            action_dim = predicted.shape[-1] // chunk_len
            predicted_chunk = predicted.reshape(-1, chunk_len, action_dim)
            executed_chunk = batch["actions"].reshape_as(predicted_chunk)
            ref_chunk = batch["curr_obs"]["ref_chunk"].reshape(
                -1, chunk_len, action_dim
            )
            intervene = batch["intervene_flags"].reshape(-1, chunk_len).bool()
            target = torch.where(intervene[..., None], executed_chunk, ref_chunk)
            bc_error = (predicted_chunk - target).square().mean(dim=-1)
            bc_loss = (bc_error * valid).sum() / valid.sum().clamp_min(1)
            bc_weight, q_weight = self._objective_weights()

            # Smooth the deterministic deployment policy, not sampled training noise.
            smooth_action, _, _ = self.model(
                forward_type=ForwardType.SAC,
                obs=batch["curr_obs"],
                apply_reference_dropout=False,
                apply_action_noise=False,
            )
            smooth_chunk = smooth_action.reshape_as(predicted_chunk)
            residual_velocity_loss, residual_acceleration_loss = (
                residual_temporal_losses(smooth_chunk, ref_chunk, valid)
            )
            residual_smoothness_loss = (
                self.config.residual_velocity_weight * residual_velocity_loss
                + self.config.residual_acceleration_weight
                * residual_acceleration_loss
            )
            actor_loss = (
                bc_weight * bc_loss
                - q_weight * q1.mean()
                + residual_smoothness_loss
            )
            self.actor_optimizer.zero_grad(set_to_none=True)
            self.critic_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            actor_norm = torch.nn.utils.clip_grad_norm_(
                self.model.actor.parameters(), self.config.actor_clip_grad
            )
            self.actor_optimizer.step()
            actor_loss_value = float(actor_loss.detach().cpu())
            bc_loss_value = float(bc_loss.detach().cpu())
            q_pi_value = float(q1.mean().detach().cpu())
            bc_weight_value = float(bc_weight)
            q_weight_value = float(q_weight)
            # How far the deployed chunk departs from the VLA reference. Stays
            # near zero while the residual actor is still copying Stage1.
            residual_abs_value = float(
                (smooth_chunk - ref_chunk).abs().mean().detach().cpu()
            )
            residual_velocity_value = float(
                residual_velocity_loss.detach().cpu()
            )
            residual_acceleration_value = float(
                residual_acceleration_loss.detach().cpu()
            )
            residual_smoothness_value = float(
                residual_smoothness_loss.detach().cpu()
            )
            self._publish_candidate(self.update_step + 1)
        else:
            actor_norm = torch.zeros(())

        self._soft_update_target_critics()
        self.update_step += 1
        self._last_metrics = {
            "critic_loss": float(critic_loss.detach().cpu()),
            "critic_grad_norm": float(critic_norm.detach().cpu()),
            "target_q": float(target_q.mean().detach().cpu()),
            "actor_loss": actor_loss_value,
            "actor_grad_norm": float(actor_norm.detach().cpu()),
            "bc_loss": bc_loss_value,
            "q_pi": q_pi_value,
            "bc_weight": bc_weight_value,
            "q_weight": q_weight_value,
            "residual_abs_mean": residual_abs_value,
            "residual_velocity_loss": residual_velocity_value,
            "residual_acceleration_loss": residual_acceleration_value,
            "residual_smoothness_loss": residual_smoothness_value,
            "actor_updated": float(actor_updated),
            "update_step": float(self.update_step),
            "replay_size": float(self.replay.total_samples),
        }
        if actor_updated:
            self._last_actor_metrics = {
                key: self._last_metrics[key] for key in _ACTOR_METRIC_KEYS
            }
        self._maybe_log_metrics()
        return dict(self._last_metrics)

    def _maybe_log_metrics(self) -> None:
        interval = int(self.config.log_interval)
        if interval <= 0 or self.update_step % interval != 0:
            return
        # The actor only updates every ``actor_update_interval`` steps, so reuse
        # the most recent actor values instead of logging NaN.
        metrics = {**self._last_metrics, **self._last_actor_metrics}
        logger.info(
            "SsEval TD3 step=%d replay=%d actor_ready=%s | critic_loss=%.4f "
            "target_q=%.4f | actor_loss=%.4f bc=%.5f q_pi=%.4f "
            "(bc_w=%.2f q_w=%.3f) | residual_abs=%.5f smooth=%.5f",
            int(metrics["update_step"]),
            int(metrics["replay_size"]),
            self.actor_ready,
            metrics["critic_loss"],
            metrics["target_q"],
            metrics["actor_loss"],
            metrics["bc_loss"],
            metrics["q_pi"],
            metrics["bc_weight"],
            metrics["q_weight"],
            metrics["residual_abs_mean"],
            metrics["residual_smoothness_loss"],
        )

    @torch.no_grad()
    def _soft_update_target_critics(self) -> None:
        for target, source in zip(
            self.target_model.q_head.parameters(), self.model.q_head.parameters()
        ):
            target.mul_(1.0 - self.config.tau).add_(source, alpha=self.config.tau)

    def _publish_candidate(self, version: int) -> None:
        snapshot = {
            key: value.detach().cpu().clone()
            for key, value in self.model.actor.state_dict().items()
        }
        with self._candidate_lock:
            self._candidate_actor_state = snapshot
            self._candidate_version = int(version)

    def take_candidate(self) -> tuple[int, dict[str, torch.Tensor]] | None:
        with self._candidate_lock:
            if self._candidate_actor_state is None or self._candidate_version is None:
                return None
            result = self._candidate_version, self._candidate_actor_state
            self._candidate_actor_state = None
            self._candidate_version = None
            return result

    def wait_idle(self) -> None:
        self.raise_if_failed()
        self._queue.join()
        self.raise_if_failed()

    def _checkpoint_path(self, step: int) -> Path:
        if self.save_dir is None:
            raise RuntimeError("save_dir is not configured.")
        return self.save_dir / f"step_{step}"

    def _maybe_save(self, *, force: bool = False) -> None:
        if self.save_interval <= 0 or self.save_dir is None:
            return
        step = int(self.update_step)
        if step <= 0:
            return
        if not force and step % self.save_interval != 0:
            return
        if step == self._last_saved_step:
            return
        path = self._checkpoint_path(step)
        logger.info("Saving SsEval checkpoint at update_step %s to %s", step, path)
        self._write_checkpoint(path)
        self._last_saved_step = step

    def _write_checkpoint(self, path: str | Path) -> None:
        root = Path(path)
        root.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "target_model": self.target_model.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "update_step": self.update_step,
                "total_transitions": self.total_transitions,
                "torch_rng_state": torch.random.get_rng_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            },
            root / "learner.pt",
        )
        self.replay.save_checkpoint(str(root / "replay"))

    def save_checkpoint(self, path: str | Path) -> None:
        self.wait_idle()
        with self._train_lock:
            self._write_checkpoint(path)

    def load_checkpoint(self, path: str | Path) -> None:
        self.wait_idle()
        root = Path(path)
        state = torch.load(
            root / "learner.pt", map_location=self.device, weights_only=False
        )
        with self._train_lock:
            self.model.load_state_dict(state["model"], strict=True)
            self.target_model.load_state_dict(state["target_model"], strict=True)
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
            self.update_step = int(state["update_step"])
            self.total_transitions = int(state["total_transitions"])
            torch.random.set_rng_state(state["torch_rng_state"].cpu())
            cuda_state = state.get("cuda_rng_state_all")
            if torch.cuda.is_available() and cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
            self.replay.clear()
            self.replay.load_checkpoint(str(root / "replay"))
            self._last_saved_step = self.update_step
            self._publish_candidate(self.update_step)

    def load_replay(self, path: str | Path) -> None:
        """Load replay files and catch up the live ``min_buffer_size`` + ``utd`` count.

        Does not read ``learner.pt``. Actor, critic, and optimizer stay at
        their current initialization; ``update_step`` starts at 0, then
        catch-up trains the live collection count. Checkpoints are written
        under the timestamped run directory at ``save_interval``, same as
        live collection.
        """
        self.wait_idle()
        root = Path(path)
        if not (root / "metadata.json").is_file():
            raise ValueError(f"replay checkpoint {root} is missing metadata.json.")
        with self._train_lock:
            self.replay.clear()
            self.replay.load_checkpoint(str(root))
            self.total_transitions = int(self.replay.total_samples)
            self.update_step = 0
            self._last_saved_step = 0
            self._candidate_actor_state = None
            self._candidate_version = None
            self._catch_up_from_replay()

    def _catch_up_from_replay(self) -> None:
        samples = int(self.replay.total_samples)
        updates = replay_catch_up_updates(
            samples,
            min_buffer_size=self.config.min_buffer_size,
            utd=self.config.utd,
        )
        if updates <= 0:
            logger.info(
                "Replay has %s samples < min_buffer_size %s; skipping catch-up.",
                samples,
                self.config.min_buffer_size,
            )
            return
        logger.info(
            "Catching up %s TD3 updates from %s replay samples "
            "(min_buffer_size=%s, utd=%s).",
            updates,
            samples,
            self.config.min_buffer_size,
            self.config.utd,
        )
        for _ in range(updates):
            self.train_once()
            self._maybe_save()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._thread is not None and self._thread.is_alive():
            self._queue.put(None)
            self._thread.join(timeout=30.0)
            if self._thread.is_alive():
                raise RuntimeError("SsEval RLT learner did not stop within 30 seconds.")
        with self._train_lock:
            self._maybe_save(force=True)
        self.replay.close()
