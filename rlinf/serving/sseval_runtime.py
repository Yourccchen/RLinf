# Copyright 2026 The RLinf Authors.

"""In-process RLT runtime called by RealWorldInference's RlinfPolicy."""

from __future__ import annotations

import copy
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from rlinf.algorithms.rlt.rollout import (
    predict_rlt_candidates,
    validate_rlt_stage2_configs,
)
from rlinf.algorithms.rlt.transition import overwrite_rlt_ref_with_human
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.envs.remote_songling import SonglingActionCodec
from rlinf.models import get_model
from rlinf.serving.sseval_contract import (
    ACTION_DIM,
    DualActionCandidates,
    TransitionFeedback,
    songling_chunk_shape,
)
from rlinf.serving.sseval_learner import (
    InProcessLearnerConfig,
    InProcessRLTTD3Learner,
)

logger = logging.getLogger(__name__)

_FEATURE_KEYS_FROM_SFT = (
    "model_type",
    "num_action_chunks",
    "action_dim",
    "num_steps",
    "add_value_head",
    "is_lora",
)


def _repo_root_from_runtime(runtime_path: Path) -> Path:
    for parent in (runtime_path.parent, *runtime_path.parents):
        if (parent / "rlinf" / "serving").is_dir() and (parent / "examples").is_dir():
            return parent
    return runtime_path.parents[3]


def resolve_stage1_sft_path(sft_ref: str, runtime_path: Path) -> Path:
    """Resolve ``stage1_sft_config`` against the runtime YAML and repo root."""
    candidate = Path(sft_ref).expanduser()
    search = []
    if candidate.is_absolute():
        search.append(candidate)
    else:
        search.extend(
            (
                runtime_path.parent / candidate,
                _repo_root_from_runtime(runtime_path) / candidate,
                Path.cwd() / candidate,
            )
        )
    for path in search:
        resolved = path.resolve()
        if resolved.is_file():
            return resolved
    raise FileNotFoundError(
        f"stage1_sft_config {sft_ref!r} was not found relative to "
        f"{runtime_path} or the RLinf repo root."
    )


def _resolve_sseval_path(
    path_ref: str | Path, runtime_config_path: str | Path | None = None
) -> Path:
    path = Path(path_ref).expanduser()
    if not path.is_absolute() and runtime_config_path is not None:
        path = _repo_root_from_runtime(Path(runtime_config_path)) / path
    return path.resolve()


def resolve_sseval_checkpoint_dir(
    path_ref: str | Path, runtime_config_path: str | Path | None = None
) -> Path:
    """Resolve ``actor_checkpoint`` to a ``step_N`` directory with ``learner.pt``."""
    path = _resolve_sseval_path(path_ref, runtime_config_path)
    if not path.is_dir():
        raise ValueError(
            "actor_checkpoint must be a step_N directory containing learner.pt, "
            f"got {path}."
        )
    if not (path / "learner.pt").is_file():
        raise ValueError(f"actor_checkpoint {path} is missing learner.pt.")
    return path


def resolve_sseval_replay_dir(
    path_ref: str | Path, runtime_config_path: str | Path | None = None
) -> Path:
    """Resolve ``replay_checkpoint`` to a directory with replay ``metadata.json``.

    Accepts a ``step_N`` directory that contains ``replay/``, or the ``replay/``
    directory itself.
    """
    path = _resolve_sseval_path(path_ref, runtime_config_path)
    if not path.is_dir():
        raise ValueError(
            "replay_checkpoint must be a step_N directory containing replay/, "
            f"or a replay directory with metadata.json, got {path}."
        )
    nested = path / "replay" / "metadata.json"
    if nested.is_file():
        return (path / "replay").resolve()
    if (path / "metadata.json").is_file():
        return path
    raise ValueError(
        "replay_checkpoint must be a step_N directory containing replay/, "
        f"or a replay directory with metadata.json, got {path}."
    )


def apply_sseval_restore(
    runtime: "SsEvalRLTRuntime",
    cfg: DictConfig,
    config_path: str | Path,
) -> None:
    """Restore from ``actor_checkpoint`` or ``replay_checkpoint``, not both."""
    actor_checkpoint = cfg.get("actor_checkpoint", None)
    replay_checkpoint = cfg.get("replay_checkpoint", None)
    if actor_checkpoint and replay_checkpoint:
        raise ValueError(
            "actor_checkpoint and replay_checkpoint cannot both be set. "
            "Use actor_checkpoint to restore learner weights and replay, "
            "or replay_checkpoint to reuse replay only."
        )
    if actor_checkpoint:
        runtime.load_checkpoint(
            resolve_sseval_checkpoint_dir(str(actor_checkpoint), config_path)
        )
    elif replay_checkpoint:
        runtime.load_replay_checkpoint(
            resolve_sseval_replay_dir(str(replay_checkpoint), config_path)
        )


def _copy_openpi_from_sft(src_model: DictConfig) -> dict[str, Any]:
    raw = OmegaConf.to_container(src_model.openpi, resolve=False)
    if not isinstance(raw, dict):
        raise ValueError("actor.model.openpi must be a mapping in the Stage1 SFT YAML.")
    chunks = int(src_model.num_action_chunks)
    action_dim = int(src_model.action_dim)
    num_steps = int(src_model.num_steps)
    raw["task"] = "eval"
    raw["action_chunk"] = chunks
    raw["action_horizon"] = chunks
    raw["action_env_dim"] = action_dim
    raw["num_steps"] = num_steps
    return raw


def align_feature_model_from_sft_config(
    cfg: DictConfig, runtime_config_path: str | Path
) -> None:
    """Copy frozen Stage1 fields from an SFT YAML into ``feature_model``.

    ``stage1_sft_config`` points at a Stage1 Hydra experiment. ``actor.model``
    supplies VLA/RLT shape, OpenPI transforms, and default ``openpi_data``.
    Eval-side ``feature_model.openpi_data`` keys override the SFT copy so
    ``norm_stats_path`` can differ from the training host. The deploy YAML
    keeps the shorter ``actor_model.num_action_chunks`` execute and replay
    reference horizons, along with ``feature_model.model_path``,
    ``feature_model.precision``, and ``openpi.task=eval``.
    """
    sft_ref = OmegaConf.select(cfg, "stage1_sft_config", default=None)
    if not sft_ref:
        return
    sft_path = resolve_stage1_sft_path(str(sft_ref), Path(runtime_config_path))
    sft = OmegaConf.load(str(sft_path))
    src = OmegaConf.select(sft, "actor.model")
    if src is None:
        raise ValueError(
            f"{sft_path} has no actor.model; it is not a Stage1 SFT config."
        )
    if OmegaConf.select(src, "openpi", default=None) is None:
        raise ValueError(f"{sft_path} actor.model.openpi is required for Stage1 align.")

    with open_dict(cfg):
        if OmegaConf.select(cfg, "feature_model", default=None) is None:
            cfg.feature_model = {}
        feature = cfg.feature_model
        with open_dict(feature):
            for key in _FEATURE_KEYS_FROM_SFT:
                value = OmegaConf.select(src, key, default=None)
                if value is not None:
                    feature[key] = copy.deepcopy(value)
            feature.openpi = _copy_openpi_from_sft(src)
            sft_data = OmegaConf.select(src, "openpi_data", default=None)
            eval_data = OmegaConf.select(feature, "openpi_data", default=None)
            if sft_data is not None or eval_data is not None:
                merged: dict[str, Any] = {}
                if sft_data is not None:
                    merged.update(OmegaConf.to_container(sft_data, resolve=False) or {})
                if eval_data is not None:
                    merged.update(
                        OmegaConf.to_container(eval_data, resolve=False) or {}
                    )
                feature.openpi_data = merged
        actor = OmegaConf.select(cfg, "actor_model", default=None)
        if actor is not None:
            execute_chunks = int(
                OmegaConf.select(
                    actor,
                    "num_action_chunks",
                    default=int(src.num_action_chunks),
                )
            )
            reference_chunks = int(
                OmegaConf.select(
                    actor,
                    "ref_num_action_chunks",
                    default=execute_chunks,
                )
            )
            with open_dict(actor):
                actor.num_action_chunks = execute_chunks
                actor.ref_num_action_chunks = reference_chunks


_REPLAY_ID_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def sseval_replay_weights_id(episode_id: str, chunk_id: int, selected_mode: str) -> str:
    """Return the replay file suffix for one SsEval transition.

    Replay files are ``trajectory_{id}_{model_weights_id}.pt``. The suffix
    names the collected chunk and who executed it. Actor snapshot versions
    stay in ``Trajectory.versions``; putting them here looks like a worker
    rank or checkpoint step.
    """
    safe = _REPLAY_ID_UNSAFE.sub("-", str(episode_id)).strip("-._")
    if not safe:
        safe = "episode"
    return f"ep-{safe[:80]}_chunk-{int(chunk_id)}_{selected_mode}"


@dataclass
class _PendingInference:
    episode_id: str
    chunk_id: int
    replay_obs: dict[str, torch.Tensor]
    actor_version: int


def _single_observation(
    observation: Mapping[str, Any], instruction: str, action_dim: int = ACTION_DIM
) -> dict:
    state = torch.as_tensor(observation["states"], dtype=torch.float32)
    main = torch.as_tensor(observation["main_images"])
    wrist = torch.as_tensor(observation["wrist_images"])
    if state.shape == (action_dim,):
        state = state.unsqueeze(0)
    if main.ndim == 3:
        main = main.unsqueeze(0)
    if wrist.ndim == 4:
        wrist = wrist.unsqueeze(0)
    if state.shape != (1, action_dim):
        raise ValueError(f"SsEval state must be [1,{action_dim}], got {state.shape}.")
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
    *,
    preserve_reference_actions: bool = False,
) -> Trajectory:
    normalized_actions = torch.as_tensor(
        codec.encode(feedback.executed_actions, clip=True), dtype=torch.float32
    )
    rewards = torch.from_numpy(feedback.rewards).to(torch.float32)
    terminated = torch.from_numpy(feedback.terminated).bool()
    truncated = torch.from_numpy(feedback.truncated).bool()
    valid = torch.from_numpy(feedback.valid_step_mask).bool()
    intervene = torch.from_numpy(feedback.intervene_flags).bool()
    record_transition = bool(
        (feedback.rlt_switch_flags & feedback.valid_step_mask).any()
    )
    curr = _unbatch_replay_obs(pending.replay_obs)
    if not preserve_reference_actions:
        curr["ref_chunk"] = overwrite_rlt_ref_with_human(
            curr["ref_chunk"].unsqueeze(0),
            normalized_actions.reshape(1, -1),
            intervene.unsqueeze(0),
        ).squeeze(0)
    nxt = _unbatch_replay_obs(next_obs)
    return Trajectory(
        max_episode_length=1,
        model_weights_id=sseval_replay_weights_id(
            pending.episode_id,
            pending.chunk_id,
            feedback.selected_mode.value,
        ),
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
            "record_transition": torch.full(
                (1, 1, 1), record_transition, dtype=torch.bool
            ),
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


def _build_strided_trajectory(
    *,
    episode_id: str,
    start_step: int,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    valid: torch.Tensor,
    intervene: torch.Tensor,
    rlt_switch: torch.Tensor,
    modes: list[str],
    actor_versions: list[int],
    curr_obs: Mapping[str, torch.Tensor],
    next_obs: Mapping[str, torch.Tensor],
    preserve_reference_actions: bool = False,
) -> Trajectory:
    """Build one overlapping C-step transition from a continuous episode buffer."""
    mode = modes[0] if len(set(modes)) == 1 else "mixed"
    curr = {key: value.detach().clone() for key, value in curr_obs.items()}
    if not preserve_reference_actions:
        curr["ref_chunk"] = overwrite_rlt_ref_with_human(
            curr["ref_chunk"].unsqueeze(0),
            actions.reshape(1, -1),
            intervene.unsqueeze(0),
        ).squeeze(0)
    nxt = {key: value.detach().clone() for key, value in next_obs.items()}
    return Trajectory(
        max_episode_length=1,
        model_weights_id=sseval_replay_weights_id(
            episode_id,
            start_step,
            mode,
        ),
        actions=actions.reshape(1, 1, -1),
        intervene_flags=intervene.reshape(1, 1, -1),
        rewards=rewards.reshape(1, 1, -1),
        terminations=terminated.reshape(1, 1, -1),
        truncations=truncated.reshape(1, 1, -1),
        dones=(terminated | truncated).reshape(1, 1, -1),
        versions=torch.full(
            (1, 1, 1),
            float(actor_versions[0] if actor_versions else 0),
        ),
        forward_inputs={
            "action": actions.reshape(1, 1, -1),
            "valid_step_mask": valid.reshape(1, 1, -1),
            "record_transition": torch.full(
                (1, 1, 1),
                bool((rlt_switch & valid).any()),
                dtype=torch.bool,
            ),
            "actor_switch": torch.full(
                (1, 1, 1),
                any(item == "actor" for item in modes),
                dtype=torch.bool,
            ),
        },
        curr_obs={
            key: value.reshape(1, 1, *value.shape) for key, value in curr.items()
        },
        next_obs={key: value.reshape(1, 1, *value.shape) for key, value in nxt.items()},
    )


class SsEvalRLTRuntime:
    """Serial inference facade with a non-blocking asynchronous TD3 learner.

    Executed chunks stay in an episode buffer until a labeled outcome arrives.
    Success and failure flush the buffer into replay; abort/truncate drops it.
    """

    def __init__(
        self,
        *,
        feature_model: Any,
        active_policy_model: torch.nn.Module,
        action_codec: SonglingActionCodec,
        learner: InProcessRLTTD3Learner,
        reference_seed: int = 2026,
        chunk_len: int | None = None,
        transition_stride: int | None = None,
    ) -> None:
        self.feature_model = feature_model.eval().requires_grad_(False)
        self.active_policy_model = active_policy_model.eval().requires_grad_(False)
        self.action_codec = action_codec
        self.action_dim = action_codec.action_dim
        self.learner = learner
        self.reference_seed = int(reference_seed)
        model_chunk_len = int(
            getattr(active_policy_model, "chunk_len", 0)
            or getattr(active_policy_model, "num_action_chunks", 0)
            or 0
        )
        resolved = int(chunk_len) if chunk_len is not None else model_chunk_len
        if resolved < 1:
            raise ValueError(f"SsEval RLT chunk_len must be positive, got {resolved}.")
        self.chunk_len = resolved
        self.transition_stride = (
            resolved if transition_stride is None else int(transition_stride)
        )
        if (
            not 0 < self.transition_stride <= resolved
            or resolved % self.transition_stride
        ):
            raise ValueError(
                "transition_stride must be a positive divisor of chunk_len"
            )
        if self.transition_stride < resolved and (
            self.action_dim != 25 or hasattr(feature_model, "prepare_observation")
        ):
            raise ValueError(
                "Online stride replay requires Xingchen OpenPI intermediate observations"
            )
        self._ario_receipts = None
        self._ario_key = None
        self._ario_ranges = {}
        self._ario_stride_steps = []
        self._stride_actions = []
        self._stride_rewards = []
        self._stride_terminated = []
        self._stride_truncated = []
        self._stride_valid = []
        self._stride_intervene = []
        self._stride_switch = []
        self._stride_modes = []
        self._stride_versions = []
        self._stride_features = {}
        self._stride_chunk_starts = {}
        self._stride_windows = []
        self._next_stride_start = 0
        self._seed_index = 0
        self._episode_id: str | None = None
        self._instruction = ""
        self._next_chunk_id = 0
        self._pending: _PendingInference | None = None
        self._episode_buffer: list[Trajectory] = []
        self._seen_feedback: set[tuple[str, int]] = set()
        self._active_actor_version = 0
        self._episode_actor_ready = False
        self._lock = threading.RLock()
        self._closed = False

    @classmethod
    def from_config(
        cls,
        config_path: str | Path,
        *,
        feature_model: Any = None,
        feature_model_adapter: Any = None,
    ) -> "SsEvalRLTRuntime":
        if feature_model is not None and feature_model_adapter is not None:
            raise ValueError("Pass a feature_model or an adapter, not both")
        cfg = OmegaConf.load(str(config_path))
        if feature_model is None:
            align_feature_model_from_sft_config(cfg, config_path)
        OmegaConf.resolve(cfg)
        device = torch.device(str(cfg.get("device", "cuda")))
        embodiment = str(cfg.get("embodiment", "songling"))
        if embodiment not in {"songling", "xingchen"}:
            raise ValueError(f"Unsupported RLT embodiment: {embodiment!r}")
        action_dim = 25 if embodiment == "xingchen" else ACTION_DIM
        if int(cfg.actor_model.action_dim) != action_dim:
            raise ValueError("Actor action_dim does not match embodiment")
        codec = SonglingActionCodec.from_config(cfg.action_codec, action_dim=action_dim)
        if feature_model is None:
            feature_action_dim = None
            if feature_model_adapter is not None:
                if (
                    embodiment != "xingchen"
                    or int(cfg.feature_model.action_dim)
                    != feature_model_adapter.model_action_dim
                ):
                    raise ValueError(
                        "Xingchen OpenPI adapter requires the trained 31D Stage1"
                    )
                feature_action_dim = feature_model_adapter.action_dim
            validate_rlt_stage2_configs(
                cfg.actor_model,
                cfg.feature_model,
                feature_action_dim=feature_action_dim,
            )
            feature_model = (
                get_model(copy.deepcopy(cfg.feature_model)).to(device).eval()
            )
            if feature_model_adapter is not None:
                feature_model = feature_model_adapter(feature_model)
        else:
            if (
                cfg.get("actor_checkpoint") or cfg.get("replay_checkpoint")
            ) and not getattr(feature_model, "supports_replay_restore", True):
                raise ValueError(
                    "This feature provider does not support actor/replay restore yet"
                )
            feature_model.validate_config(cfg.actor_model, codec)
            feature_model = feature_model.to(device).eval()
        active_model = get_model(copy.deepcopy(cfg.actor_model)).to(device).eval()
        learner_model = copy.deepcopy(active_model).train().requires_grad_(True)
        learner_type = str(cfg.get("learner_type", "rlt_td3"))
        learner_class = InProcessRLTTD3Learner
        learner_kwargs = {}
        raw_learner_cfg = OmegaConf.to_container(cfg.get("learner", {}), resolve=True)
        if learner_type == "halo_wa":
            from rlinf.serving.halo_wa_learner import HaloWALearner, HaloWALearnerConfig

            if cfg.get("replay_checkpoint"):
                raise ValueError(
                    "HALO-WA requires demo_checkpoint or a complete actor_checkpoint"
                )
            if str(cfg.actor_model.model_type) != "halo_wa_policy":
                raise ValueError("HALO-WA learner requires halo_wa_policy")
            learner_class = HaloWALearner
            learner_cfg = HaloWALearnerConfig.from_mapping(raw_learner_cfg)
            learner_kwargs["feature_identity"] = feature_model.rlt_checkpoint_version
            if learner_cfg.training_stage == "online" and not (
                cfg.get("demo_checkpoint") or cfg.get("actor_checkpoint")
            ):
                raise ValueError(
                    "HALO-WA online training requires a demo_checkpoint or complete resume"
                )
        elif learner_type == "rlt_td3":
            if str(cfg.actor_model.model_type) == "halo_wa_policy":
                raise ValueError("halo_wa_policy requires learner_type=halo_wa")
            learner_cfg = InProcessLearnerConfig.from_mapping(raw_learner_cfg)
        else:
            raise ValueError(f"Unknown learner_type: {learner_type}")
        save_interval = int(cfg.get("save_interval", 0) or 0)
        save_dir = cfg.get("save_dir", None)
        if save_interval > 0:
            if not save_dir:
                raise ValueError("save_dir is required when save_interval > 0.")
            save_path = Path(str(save_dir)).expanduser()
            if not save_path.is_absolute():
                save_path = _repo_root_from_runtime(Path(config_path)) / save_path
            save_dir = save_path.resolve()
        learner = learner_class(
            learner_model,
            learner_cfg if learner_type == "rlt_td3" else raw_learner_cfg,
            **learner_kwargs,
            save_interval=save_interval,
            save_dir=save_dir,
        )
        if learner_type == "halo_wa" and cfg.get("demo_checkpoint"):
            try:
                demo_path = Path(str(cfg.demo_checkpoint)).expanduser()
                if not demo_path.is_absolute():
                    demo_path = _repo_root_from_runtime(Path(config_path)) / demo_path
                learner.load_demo_checkpoint(demo_path)
            except BaseException:
                learner.close()
                raise
        runtime = cls(
            feature_model=feature_model,
            active_policy_model=active_model,
            action_codec=codec,
            learner=learner,
            reference_seed=int(cfg.get("reference_seed", 2026)),
            chunk_len=int(cfg.actor_model.num_action_chunks),
            transition_stride=cfg.get("transition_stride")
            if embodiment == "xingchen"
            else None,
        )
        try:
            apply_sseval_restore(runtime, cfg, config_path)
        except BaseException:
            runtime.close()
            raise
        return runtime

    def enable_ario_recording(self, run_id: str) -> None:
        """Enable receipts for the active episode; raw data stays on the robot."""
        from rlinf.serving.replay_receipts import ReplayReceipts

        if self._episode_id is None:
            raise RuntimeError("Start an episode before enabling ARIO recording")
        if self._ario_receipts is None:
            if self.learner.save_dir is None:
                raise ValueError("ARIO receipts require the existing learner save_dir")
            self._ario_receipts = ReplayReceipts(
                Path(self.learner.save_dir) / "ario_receipts"
            )
        self._ario_key = f"{run_id}:{self._episode_id}"
        if not self._ario_receipts.contains(self._ario_key):
            self._ario_receipts.start(
                self._ario_key,
                self._instruction,
                self.transition_stride,
                self._episode_id,
            )

    def ario_status(self, episode_id: str, run_id: str) -> dict[str, Any]:
        """Return persisted acceptance state, including after episode reset."""
        from rlinf.serving.replay_receipts import ReplayReceipts

        if self._ario_receipts is not None:
            return self._ario_receipts.status(f"{run_id}:{episode_id}")
        if self.learner.save_dir is not None:
            return ReplayReceipts(Path(self.learner.save_dir) / "ario_receipts").status(
                f"{run_id}:{episode_id}"
            )
        return {"episode_id": episode_id, "complete": False}

    def _ario_recording(self) -> bool:
        return self._ario_receipts is not None and self._ario_receipts.contains(
            self._ario_key
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
            if hasattr(self.feature_model, "start_episode"):
                self.feature_model.start_episode(episode_id)
            self._activate_candidate_actor()
            self._episode_id = episode_id
            self._instruction = instruction
            self._next_chunk_id = 0
            self._pending = None
            self._episode_buffer.clear()
            self._reset_stride_buffers()
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
            if hasattr(self.feature_model, "prepare_observation"):
                env_obs = self.feature_model.prepare_observation(
                    observation, self._instruction
                )
            else:
                env_obs = _single_observation(
                    observation, self._instruction, self.action_dim
                )
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
            actor_expected = songling_chunk_shape(self.chunk_len, self.action_dim)
            if candidates.actor_action.shape != actor_expected:
                raise ValueError(
                    f"RLT Actor candidate must be {actor_expected}, got "
                    f"{candidates.actor_action.shape}."
                )
            if (
                candidates.vla_action.ndim != 2
                or candidates.vla_action.shape[1] != self.action_dim
                or candidates.vla_action.shape[0] < self.chunk_len
            ):
                raise ValueError(
                    f"RLT VLA candidate must have shape (H, {self.action_dim}) with "
                    f"H >= {self.chunk_len}, got {candidates.vla_action.shape}."
                )
            feedback = (
                TransitionFeedback.from_mapping(
                    transition_feedback, action_dim=self.action_dim
                )
                if transition_feedback is not None
                else None
            )
            if feedback is not None:
                self._ingest_feedback(feedback, replay_obs)
            if feedback is not None and feedback.done:
                self._close_episode()
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

    @torch.no_grad()
    def _extract_intermediate_features(
        self,
        feedback: TransitionFeedback,
    ) -> dict[int, dict[str, torch.Tensor]]:
        observations = sorted(
            feedback.intermediate_observations,
            key=lambda item: int(item["step_index"]),
        )
        expected = set(
            range(self.transition_stride, self.chunk_len, self.transition_stride)
        )
        received = {int(item["step_index"]) for item in observations}
        if not received.issubset(expected):
            raise ValueError(
                "Stride-subsampled feedback contains unexpected observation offsets; "
                f"expected a subset of {sorted(expected)}, got {sorted(received)}."
            )
        if not observations:
            return {}
        singles = [
            _single_observation(item, self._instruction, self.action_dim)
            for item in observations
        ]
        env_obs = {
            "states": torch.cat([item["states"] for item in singles], dim=0),
            "main_images": torch.cat([item["main_images"] for item in singles], dim=0),
            "wrist_images": torch.cat(
                [item["wrist_images"] for item in singles], dim=0
            ),
            "task_descriptions": [item["task_descriptions"][0] for item in singles],
        }
        device = next(self.feature_model.parameters()).device
        generator = torch.Generator(device=device)
        generator.manual_seed(self.reference_seed + self._seed_index)
        extracted = self.feature_model.extract_rlt_obs(env_obs, rng=generator)
        reference_horizon = int(
            getattr(self.active_policy_model, "ref_chunk_len", self.chunk_len)
        )
        normalized = dict(extracted)
        normalized["ref_chunk"] = self.action_codec.encode(
            extracted["ref_chunk"][..., :reference_horizon, :],
            clip=True,
        )
        return {
            int(observation["step_index"]): {
                key: value[index].detach().to(torch.float32).cpu().contiguous()
                for key, value in normalized.items()
                if key in ("z_rl", "proprio", "ref_chunk")
            }
            for index, observation in enumerate(observations)
        }

    def _append_strided_feedback(
        self,
        pending: _PendingInference,
        next_obs: Mapping[str, torch.Tensor],
        feedback: TransitionFeedback,
    ) -> None:
        base_step = len(self._stride_actions)
        if self._ario_recording():
            self._ario_stride_steps.extend(
                [feedback.chunk_id, offset] for offset in range(self.chunk_len)
            )
        self._stride_chunk_starts[feedback.key] = base_step
        self._stride_features[base_step] = _unbatch_replay_obs(pending.replay_obs)
        for offset, feature in self._extract_intermediate_features(feedback).items():
            self._stride_features[base_step + offset] = feature
        self._stride_features[base_step + self.chunk_len] = _unbatch_replay_obs(
            next_obs
        )

        normalized_actions = torch.as_tensor(
            self.action_codec.encode(feedback.executed_actions, clip=True),
            dtype=torch.float32,
        )
        vectors = (
            (self._stride_actions, normalized_actions),
            (self._stride_rewards, torch.from_numpy(feedback.rewards).float()),
            (
                self._stride_terminated,
                torch.from_numpy(feedback.terminated).bool(),
            ),
            (self._stride_truncated, torch.from_numpy(feedback.truncated).bool()),
            (self._stride_valid, torch.from_numpy(feedback.valid_step_mask).bool()),
            (
                self._stride_intervene,
                torch.from_numpy(feedback.intervene_flags).bool(),
            ),
            (
                self._stride_switch,
                torch.from_numpy(feedback.rlt_switch_flags).bool(),
            ),
        )
        for target, values in vectors:
            target.extend(value.detach().cpu() for value in values)
        self._stride_modes.extend([feedback.selected_mode.value] * self.chunk_len)
        self._stride_versions.extend([pending.actor_version] * self.chunk_len)

        total_steps = len(self._stride_actions)
        while self._next_stride_start + self.chunk_len <= total_steps:
            start = self._next_stride_start
            stop = start + self.chunk_len
            self._next_stride_start += self.transition_stride
            curr = self._stride_features.get(start)
            nxt = self._stride_features.get(stop)
            if curr is None or nxt is None:
                continue
            valid = torch.stack(self._stride_valid[start:stop])
            rlt_switch = torch.stack(self._stride_switch[start:stop])
            if not bool((valid & rlt_switch).any()):
                continue
            self._episode_buffer.append(
                _build_strided_trajectory(
                    episode_id=pending.episode_id,
                    start_step=pending.chunk_id * self.chunk_len - base_step + start,
                    actions=torch.stack(self._stride_actions[start:stop]),
                    rewards=torch.stack(self._stride_rewards[start:stop]),
                    terminated=torch.stack(self._stride_terminated[start:stop]),
                    truncated=torch.stack(self._stride_truncated[start:stop]),
                    valid=valid,
                    intervene=torch.stack(self._stride_intervene[start:stop]),
                    rlt_switch=rlt_switch,
                    modes=self._stride_modes[start:stop],
                    actor_versions=self._stride_versions[start:stop],
                    curr_obs=curr,
                    next_obs=nxt,
                    preserve_reference_actions=bool(
                        getattr(self.feature_model, "preserve_reference_actions", False)
                    ),
                )
            )
            self._stride_windows.append((start, self._episode_buffer[-1]))
            if self._ario_recording():
                trajectory = self._episode_buffer[-1]
                self._ario_ranges[trajectory.model_weights_id] = [
                    step
                    for step, keep in zip(
                        self._ario_stride_steps[start:stop], valid.tolist()
                    )
                    if keep
                ]

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
        should_record = bool(
            (feedback.rlt_switch_flags & feedback.valid_step_mask).any()
        )
        if not should_record:
            if self.transition_stride < self.chunk_len:
                self._reset_stride_buffers()
            self._seen_feedback.add(feedback.key)
            if feedback.done:
                self._resolve_episode_buffer(feedback)
            return True
        if feedback.executed_actions.shape != songling_chunk_shape(
            self.chunk_len, self.action_dim
        ):
            raise ValueError(
                "Recorded RLT feedback must match the Actor execute horizon "
                f"{songling_chunk_shape(self.chunk_len, self.action_dim)}, got "
                f"{feedback.executed_actions.shape}."
            )
        if self.transition_stride < self.chunk_len:
            self._append_strided_feedback(pending, next_obs, feedback)
        else:
            trajectory = _build_feedback_trajectory(
                pending,
                next_obs,
                feedback,
                self.action_codec,
                preserve_reference_actions=bool(
                    getattr(self.feature_model, "preserve_reference_actions", False)
                ),
            )
            if bool(
                trajectory.forward_inputs["record_transition"]
                .detach()
                .to(torch.bool)
                .all()
            ):
                self._episode_buffer.append(trajectory)
                if self._ario_recording():
                    self._ario_ranges[trajectory.model_weights_id] = [
                        [feedback.chunk_id, index]
                        for index, keep in enumerate(feedback.valid_step_mask)
                        if keep
                    ]
        self._seen_feedback.add(feedback.key)
        if feedback.done:
            self._resolve_episode_buffer(feedback)
        return True

    def _resolve_episode_buffer(self, feedback: TransitionFeedback) -> None:
        trajectories = self._episode_buffer
        self._episode_buffer = []
        recording = self._ario_recording()
        if recording:
            records = (
                [
                    {
                        "weights_id": item.model_weights_id,
                        "steps": self._ario_ranges[item.model_weights_id],
                        "accepted": False,
                    }
                    for item in trajectories
                ]
                if feedback.has_labeled_outcome
                else []
            )
            try:
                self._ario_receipts.expect(self._ario_key, records)
            except Exception:
                logger.exception("ARIO receipt preparation failed")
                recording = False
        if not feedback.has_labeled_outcome:
            if recording:
                try:
                    self._ario_receipts.finish(self._ario_key)
                except Exception:
                    logger.exception("ARIO receipt finalization failed")
            return
        for trajectory in trajectories:
            if recording:
                from functools import partial

                self.learner.observe_replay(
                    trajectory.model_weights_id,
                    partial(
                        self._ario_receipts.accept,
                        self._ario_key,
                        trajectory.model_weights_id,
                    ),
                )
            self.learner.submit(trajectory)
        if recording:
            try:
                self._ario_receipts.finish(self._ario_key)
            except Exception:
                logger.exception("ARIO receipt finalization failed")

    def _relabel_buffered_terminal(self, feedback: TransitionFeedback) -> bool:
        if self.transition_stride < self.chunk_len:
            base = self._stride_chunk_starts.get(feedback.key)
            if base is None:
                return False
            updated = False
            for start, trajectory in self._stride_windows:
                left = max(start, base)
                right = min(start + self.chunk_len, base + self.chunk_len)
                if left >= right:
                    continue
                source = slice(left - base, right - base)
                target = slice(left - start, right - start)
                for name, values in (
                    ("rewards", feedback.rewards),
                    ("terminations", feedback.terminated),
                    ("truncations", feedback.truncated),
                ):
                    tensor = getattr(trajectory, name)
                    tensor[..., target] = torch.as_tensor(
                        values[source], dtype=tensor.dtype, device=tensor.device
                    )
                trajectory.dones = trajectory.terminations | trajectory.truncations
                updated = True
            return updated
        weights_id = sseval_replay_weights_id(
            feedback.episode_id,
            feedback.chunk_id,
            feedback.selected_mode.value,
        )
        for trajectory in reversed(self._episode_buffer):
            if trajectory.model_weights_id != weights_id:
                continue
            shape = trajectory.rewards.shape
            device = trajectory.rewards.device
            trajectory.rewards = torch.as_tensor(
                feedback.rewards, dtype=torch.float32, device=device
            ).reshape(shape)
            trajectory.terminations = torch.as_tensor(
                feedback.terminated, dtype=torch.bool, device=device
            ).reshape(shape)
            trajectory.truncations = torch.as_tensor(
                feedback.truncated, dtype=torch.bool, device=device
            ).reshape(shape)
            trajectory.dones = trajectory.terminations | trajectory.truncations
            return True
        return False

    def _close_episode(self) -> None:
        self._ario_key = None
        self._ario_ranges.clear()
        self._episode_id = None
        self._instruction = ""
        self._pending = None
        self._episode_buffer.clear()
        self._reset_stride_buffers()

    def _reset_stride_buffers(self) -> None:
        self._ario_stride_steps.clear()
        self._stride_actions.clear()
        self._stride_rewards.clear()
        self._stride_terminated.clear()
        self._stride_truncated.clear()
        self._stride_valid.clear()
        self._stride_intervene.clear()
        self._stride_switch.clear()
        self._stride_modes.clear()
        self._stride_versions.clear()
        self._stride_features.clear()
        self._stride_chunk_starts.clear()
        self._stride_windows.clear()
        self._next_stride_start = 0

    def end_episode(self, final_feedback: Mapping[str, Any]) -> bool:
        with self._lock:
            self._require_open()
            if self._episode_id is None:
                raise RuntimeError("No active SsEval RLT episode to end.")
            feedback = TransitionFeedback.from_mapping(
                final_feedback, action_dim=self.action_dim
            )
            if not feedback.done:
                raise ValueError("end_episode requires terminal or truncated feedback.")
            if feedback.key in self._seen_feedback:
                should_record = bool(
                    (feedback.rlt_switch_flags & feedback.valid_step_mask).any()
                )
                if (
                    should_record
                    and feedback.has_labeled_outcome
                    and not self._relabel_buffered_terminal(feedback)
                ):
                    raise ValueError(
                        "Terminal feedback refers to a recorded chunk that is "
                        "missing from the episode buffer."
                    )
                self._resolve_episode_buffer(feedback)
                self._close_episode()
                return False
            if self._pending is None:
                raise RuntimeError("No pending SsEval RLT action to end.")
            ingested = self._ingest_feedback(feedback, self._pending.replay_obs)
            self._close_episode()
            return ingested

    def reset(self) -> None:
        with self._lock:
            self._close_episode()
            self._next_chunk_id = 0
            self._seen_feedback.clear()
            if hasattr(self.feature_model, "reset"):
                self.feature_model.reset()

    def get_wire_config(self) -> dict[str, Any]:
        if hasattr(self.feature_model, "get_wire_config"):
            return self.feature_model.get_wire_config()
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
            if not getattr(self.feature_model, "supports_replay_restore", True):
                raise ValueError(
                    "This feature provider does not support actor/replay restore yet"
                )
            checkpoint_dir = resolve_sseval_checkpoint_dir(path)
            self.learner.load_checkpoint(checkpoint_dir)
            self.active_policy_model.load_state_dict(
                self.learner.model.state_dict(), strict=True
            )
            self.active_policy_model.eval().requires_grad_(False)
            self._activate_candidate_actor()

    def load_replay_checkpoint(self, path: str | Path) -> None:
        """Load replay only, catch up TD3 updates, and keep a fresh Stage2 actor."""
        with self._lock:
            self._require_open()
            if self._episode_id is not None:
                raise RuntimeError("Checkpoint restore requires an episode boundary.")
            if not getattr(self.feature_model, "supports_replay_restore", True):
                raise ValueError(
                    "This feature provider does not support actor/replay restore yet"
                )
            self.learner.load_replay(resolve_sseval_replay_dir(path))
            self._activate_candidate_actor()

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("SsEval RLT runtime is closed.")

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._close_episode()
            self._seen_feedback.clear()
            try:
                self.learner.close()
            finally:
                try:
                    if self._ario_receipts is not None:
                        self._ario_receipts.close()
                finally:
                    if hasattr(self.feature_model, "close"):
                        self.feature_model.close()
