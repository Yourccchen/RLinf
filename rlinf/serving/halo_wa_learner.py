"""HALO-WA training on the existing SsEval replay and publication infrastructure.

The legacy MLP learner is unchanged. This learner retains WA references as input,
uses executed actions only as supervised targets, and maintains a separate demo
buffer. Collection never makes an untrained Actor eligible for execution.
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from rlinf.data.storage.replay import TrajectoryReplayBuffer
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.halo_wa import losses
from rlinf.serving.sseval_learner import (
    InProcessLearnerConfig,
    InProcessRLTTD3Learner,
    _to_device,
    make_sseval_run_save_dir,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HaloWALearnerConfig(InProcessLearnerConfig):
    training_stage: str = "online"
    batch_size: int = 64
    actor_lr: float = 2.5e-5
    critic_lr: float = 3e-5
    bc_warmup_steps: int = 1000
    bc_coef: float = 50.0
    demo_ratio: float = 0.5
    target_action_noise: bool = False
    actor_update_action_noise: bool = False
    reference_dropout_prob: float = 0.0
    q_upper_bound: float = 1.0
    enable_q_upper_bound: bool = True
    overshoot_penalty_coef: float = 1.0

    @classmethod
    def from_mapping(cls, value):
        cfg = super().from_mapping(value)
        if cfg.training_stage not in {"collect", "online"}:
            raise ValueError("training_stage must be collect or online")
        if cfg.bc_warmup_steps < 1 or cfg.bc_coef < 0:
            raise ValueError("Require positive bc_warmup_steps and nonnegative bc_coef")
        if cfg.demo_ratio != 0.5 or cfg.batch_size % 2:
            raise ValueError("HALO-WA requires demo_ratio=0.5 and an even batch_size")
        if not 0 < cfg.gamma <= 1 or not 0 < cfg.tau <= 1:
            raise ValueError("gamma and tau must be in (0,1]")
        if (
            cfg.target_action_noise
            or cfg.actor_update_action_noise
            or cfg.reference_dropout_prob
        ):
            raise ValueError(
                "Official HALO-WA configuration uses no action noise or reference dropout"
            )
        return cfg


def _cat_batches(batches):
    first = batches[0]
    if torch.is_tensor(first):
        return torch.cat(batches, dim=0)
    if isinstance(first, dict):
        if any(set(b) != set(first) for b in batches):
            raise ValueError("Demo and replay schemas differ")
        return {key: _cat_batches([b[key] for b in batches]) for key in first}
    if first is None and all(b is None for b in batches):
        return None
    raise ValueError("Replay batch must contain tensors, dictionaries or None")


def _sample_exact(buffer, count):
    if buffer.total_samples == 0:
        raise ValueError("Cannot sample an empty demo or online buffer")
    batches = []
    remaining = count
    while remaining:
        batch = buffer.sample(remaining)
        size = batch["rewards"].shape[0]
        batches.append(batch)
        remaining -= size
    return _cat_batches(batches)


class HaloWALearner(InProcessRLTTD3Learner):
    """Train the official attention heads with BC warmup and balanced replay."""

    def __init__(self, model, config=None, *, feature_identity: str, **kwargs):
        cfg = HaloWALearnerConfig.from_mapping(config)
        self.feature_identity = feature_identity
        self.bc_step = 0
        self.demo = TrajectoryReplayBuffer(
            seed=cfg.seed + 1,
            enable_cache=True,
            cache_size=min(cfg.replay_capacity, 5000),
            sample_window_size=cfg.replay_capacity,
            max_num_samples=cfg.replay_capacity,
            auto_save=False,
        )
        super().__init__(model, cfg, **kwargs)
        if (
            cfg.training_stage == "collect"
            and self.save_dir is not None
            and self.save_interval == 0
        ):
            self.save_dir = make_sseval_run_save_dir(self.save_dir)

    @property
    def actor_ready(self):
        return (
            self.config.training_stage == "online"
            and self.bc_step >= self.config.bc_warmup_steps
        )

    def add_demo_trajectories(self, trajectories):
        """Add normalized demonstrations without changing the frozen WA reference."""
        with self._train_lock:
            self.demo.add_trajectories(trajectories)

    def add_and_train(self, trajectory):
        with self._train_lock:
            self.replay.add_trajectories([trajectory])
            count = int(trajectory.rewards.shape[0] * trajectory.rewards.shape[1])
            self.total_transitions += count
            if self.config.training_stage == "collect":
                if (
                    self.save_interval > 0
                    and self.total_transitions % self.save_interval == 0
                ):
                    self._write_checkpoint(
                        self.save_dir / f"collect_{self.total_transitions}"
                    )
                return
            if self.replay.total_samples < self.config.min_buffer_size:
                return
            while self.bc_step < self.config.bc_warmup_steps:
                self.train_once()
            for _ in range(self.config.utd * count):
                self.train_once()
                self._maybe_save()

    def _bc_target(self, batch, reference):
        executed = batch["actions"].reshape_as(reference)
        human = batch["intervene_flags"].reshape(reference.shape[:2]).bool()
        return torch.where(human[..., None], executed, reference)

    def train_once(self):
        """Perform one BC warmup step or one mixed TD3 step."""
        if self.config.training_stage == "collect":
            raise ValueError("Collection does not train the Actor")
        if self.demo.total_samples == 0:
            raise ValueError(
                "HALO-WA needs demo data before BC warmup or online training"
            )
        cfg = self.config
        warming = self.bc_step < cfg.bc_warmup_steps
        if warming:
            batch = _sample_exact(self.demo, cfg.batch_size)
            demo_count, online_count = cfg.batch_size, 0
        else:
            demo_count = online_count = cfg.batch_size // 2
            batch = _cat_batches(
                [
                    _sample_exact(self.demo, demo_count),
                    _sample_exact(self.replay, online_count),
                ]
            )
        batch = _to_device(batch, self.device)
        obs = batch["curr_obs"]
        rewards = batch["rewards"].reshape(cfg.batch_size, -1).float()
        valid = batch["forward_inputs"]["valid_step_mask"].reshape_as(rewards).bool()
        reference = obs["ref_chunk"][:, : self.model.chunk_len]
        if not valid.any():
            raise ValueError("HALO-WA training batch has no valid executed steps")
        metrics = dict(
            stage="bc" if warming else "online",
            demo_samples=demo_count,
            online_samples=online_count,
            actor_updated=False,
            actor_loss=0.0,
            bc_loss=0.0,
            q_pi=0.0,
            critic_loss=0.0,
        )
        if not warming:
            with torch.no_grad():
                # Upstream targets share the current visual encoder; only the
                # actor and critic attention heads have Polyak target weights.
                next_visual = self.model.encode_visual(batch["next_obs"])
                next_action = self.target_model(
                    forward_type=ForwardType.SAC,
                    obs=batch["next_obs"],
                    visual_features=next_visual,
                )[0]
                next_q = (
                    self.target_model(
                        forward_type=ForwardType.SAC_Q,
                        obs=batch["next_obs"],
                        actions=next_action,
                        visual_features=next_visual,
                    )
                    .min(-1, keepdim=True)
                    .values
                )
                reward = (rewards * valid).sum(-1, keepdim=True)
                done = (
                    batch["terminations"].reshape_as(valid).bool()
                    | batch["truncations"].reshape_as(valid).bool()
                ) & valid
                terminal = done.any(-1, keepdim=True) | ~valid.all(-1, keepdim=True)
            q = self.model(
                forward_type=ForwardType.SAC_Q,
                obs=obs,
                actions=batch["actions"],
                detach_encoder=True,
            )
            critic_loss, _, extra = losses.compute_td3_critic_loss(
                q[:, :1],
                q[:, 1:],
                reward,
                terminal.float(),
                next_q,
                losses.CriticLossConfig(
                    discount=cfg.gamma ** rewards.shape[1],
                    q_upper_bound=cfg.q_upper_bound,
                    enable_q_upper_bound=cfg.enable_q_upper_bound,
                    overshoot_penalty_coef=cfg.overshoot_penalty_coef,
                ),
            )
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.q_head.parameters(),
                cfg.critic_clip_grad,
                error_if_nonfinite=True,
            )
            self.critic_optimizer.step()
            metrics.update(critic_loss=float(critic_loss.detach()), **extra)
        actor_updated = (
            warming or (self.update_step + 1) % cfg.actor_update_interval == 0
        )
        if actor_updated:
            self.model.set_critic_requires_grad(False)
            try:
                pred = self.model(forward_type=ForwardType.SAC, obs=obs)[0].reshape_as(
                    reference
                )
                bc = losses.compute_bc_loss(
                    pred, self._bc_target(batch, reference), valid
                )
                q_pi = (
                    None
                    if warming
                    else self.model(
                        forward_type=ForwardType.SAC_Q, obs=obs, actions=pred
                    )
                    .min(-1, keepdim=True)
                    .values
                )
                loss, _ = losses.compute_bc_regularized_actor_loss(
                    q_pi,
                    bc,
                    losses.ActorLossConfig(
                        bc_coef=1.0 if warming else cfg.bc_coef,
                        q_upper_bound=cfg.q_upper_bound,
                        enable_q_upper_bound=cfg.enable_q_upper_bound,
                    ),
                )
                self.actor_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.actor.parameters(),
                    cfg.actor_clip_grad,
                    error_if_nonfinite=True,
                )
                self.actor_optimizer.step()
            finally:
                self.model.set_critic_requires_grad(True)
            metrics.update(
                actor_updated=True,
                actor_loss=float(loss.detach()),
                bc_loss=float(bc.detach()),
                q_pi=0.0 if q_pi is None else float(q_pi.detach().mean()),
            )
        if warming:
            self.bc_step += 1
            self.target_model.load_state_dict(self.model.state_dict())
        else:
            self.update_step += 1
            with torch.no_grad():
                for name in ("actor.head", "q_head"):
                    for target, source in zip(
                        self.target_model.get_submodule(name).parameters(),
                        self.model.get_submodule(name).parameters(),
                    ):
                        target.mul_(1 - cfg.tau).add_(source, alpha=cfg.tau)
        if actor_updated and self.actor_ready:
            self._publish_candidate(self.update_step)
        self._last_metrics = metrics
        step = self.bc_step if warming else self.update_step
        if cfg.log_interval > 0 and step % cfg.log_interval == 0:
            logger.info(
                "HALO-WA stage=%s step=%d demo=%d online=%d bc=%.5f critic=%.5f",
                metrics["stage"],
                step,
                demo_count,
                online_count,
                metrics["bc_loss"],
                metrics["critic_loss"],
            )
        return metrics

    def _write_checkpoint(self, path):
        super()._write_checkpoint(path)
        root = Path(path)
        self.demo.save_checkpoint(str(root / "demo"))
        metadata = dict(
            format="halo-wa-sseval-v1",
            feature_identity=self.feature_identity,
            config=asdict(self.config),
            bc_step=self.bc_step,
        )
        (root / "halo_wa.json").write_text(json.dumps(metadata, indent=2))
        torch.save(
            dict(
                replay_rng=self.replay.random_generator.get_state(),
                demo_rng=self.demo.random_generator.get_state(),
            ),
            root / "samplers.pt",
        )

    def _metadata(self, path):
        metadata = json.loads((Path(path) / "halo_wa.json").read_text())
        if (
            metadata["format"] != "halo-wa-sseval-v1"
            or metadata["feature_identity"] != self.feature_identity
        ):
            raise ValueError("HALO-WA checkpoint feature identity mismatch")
        return metadata

    def load_demo_checkpoint(self, path):
        """Load a HALO collection checkpoint as demonstrations, without its Actor."""
        self._metadata(path)
        with self._train_lock:
            self.demo.clear()
            self.demo.load_checkpoint(str(Path(path) / "replay"))
            if self.demo.total_samples == 0:
                raise ValueError("Demo checkpoint is empty")

    def load_checkpoint(self, path):
        metadata = self._metadata(path)
        if metadata["config"] != asdict(self.config):
            raise ValueError("HALO-WA checkpoint learner configuration mismatch")
        super().load_checkpoint(path)
        with self._train_lock:
            self.demo.clear()
            self.demo.load_checkpoint(str(Path(path) / "demo"))
            state = torch.load(
                Path(path) / "samplers.pt", map_location="cpu", weights_only=True
            )
            self.replay.random_generator.set_state(state["replay_rng"])
            self.demo.random_generator.set_state(state["demo_rng"])
            self.bc_step = int(metadata["bc_step"])

    def load_replay(self, path):
        raise ValueError(
            "HALO-WA requires demo_checkpoint or a complete actor_checkpoint; replay-only restore lacks feature identity"
        )

    def close(self):
        if self._closed:
            return
        # Replay persistence uses its executor, so finish it before base close.
        try:
            self.wait_idle()
            with self._train_lock:
                if (
                    self.config.training_stage == "collect"
                    and self.save_dir
                    and self.total_transitions
                ):
                    self._write_checkpoint(
                        self.save_dir / f"collect_{self.total_transitions}"
                    )
        finally:
            try:
                super().close()
            finally:
                self.demo.close()
