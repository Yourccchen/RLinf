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

import hashlib
from typing import Any, Literal

import numpy as np
import torch

from rlinf.algorithms.rlt.route import RLTRoute, RLTRouteContext
from rlinf.algorithms.rlt.transition import RLT_OBS_KEYS, RLT_TRANSITION_PREFIX
from rlinf.serving.sseval_contract import ACTION_DIM, DualActionCandidates


def _cfg_get(config: Any, key: str, default: Any = None) -> Any:
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def validate_online_transition_stride(
    transition_stride: int, *, chunk_len: int
) -> None:
    """Reject unsupported online overlap instead of silently relabeling actions.

    Online Songling currently computes the next RLT observation only after the
    whole action chunk executes. Overlap is available in offline preprocessing;
    online collection must therefore use one transition per executed chunk.
    """
    transition_stride = int(transition_stride)
    chunk_len = int(chunk_len)
    if transition_stride <= 0:
        raise ValueError("rollout.transition_stride must be positive.")
    if transition_stride != chunk_len:
        raise ValueError(
            "Online RLT overlap requires intermediate observations to pass through "
            "the Stage1 feature worker. This pipeline only observes post-chunk state, "
            f"so transition_stride must equal chunk_len ({chunk_len}), got "
            f"{transition_stride}. Use build_songling_rlt_replay.py for stride-2 "
            "overlapping offline transitions."
        )


def validate_rlt_stage2_configs(policy_cfg: Any, feature_cfg: Any) -> None:
    """Fail before loading Stage 1 when Stage 1/Stage 2 shapes disagree."""
    feature_openpi = _cfg_get(feature_cfg, "openpi", {})
    checks = (
        (
            "action_dim",
            int(_cfg_get(policy_cfg, "action_dim")),
            int(_cfg_get(feature_cfg, "action_dim")),
        ),
        (
            "reference horizon",
            int(
                _cfg_get(
                    policy_cfg,
                    "ref_num_action_chunks",
                    _cfg_get(policy_cfg, "num_action_chunks"),
                )
            ),
            int(_cfg_get(feature_cfg, "num_action_chunks")),
        ),
        (
            "RLT embedding dim",
            int(_cfg_get(policy_cfg, "z_dim")),
            int(_cfg_get(feature_openpi, "rlt_embed_dim", 2048)),
        ),
    )
    mismatches = [
        f"{name}: Stage2={stage2}, Stage1={stage1}"
        for name, stage2, stage1 in checks
        if stage2 != stage1
    ]
    if mismatches:
        raise ValueError(
            "Incompatible RLT Stage1/Stage2 configuration: " + "; ".join(mismatches)
        )
    config_name = str(_cfg_get(feature_openpi, "config_name", "")).lower()
    if "songling" in config_name:
        horizon = int(_cfg_get(policy_cfg, "num_action_chunks"))
        if horizon < 1:
            raise ValueError(
                f"Invalid Songling RLT configuration: Stage2 chunk length "
                f"must be positive, got {horizon}."
            )
        songling_checks = (
            ("Stage2 action_dim", int(_cfg_get(policy_cfg, "action_dim")), ACTION_DIM),
            (
                "Stage2 proprio_dim",
                int(_cfg_get(policy_cfg, "proprio_dim")),
                ACTION_DIM,
            ),
            (
                "Stage2 reference horizon",
                int(_cfg_get(policy_cfg, "ref_num_action_chunks")),
                horizon,
            ),
            (
                "Stage1 reference horizon",
                int(_cfg_get(feature_cfg, "num_action_chunks")),
                horizon,
            ),
            (
                "Stage1 openpi action_horizon",
                int(_cfg_get(feature_openpi, "action_horizon", horizon)),
                horizon,
            ),
            (
                "Stage1 openpi action_chunk",
                int(_cfg_get(feature_openpi, "action_chunk", horizon)),
                horizon,
            ),
            (
                "Stage1 model_action_dim",
                int(_cfg_get(feature_openpi, "model_action_dim", 0)),
                32,
            ),
            (
                "Stage1 num_images_in_input",
                int(_cfg_get(feature_openpi, "num_images_in_input", 0)),
                3,
            ),
            (
                "Stage1 rlt_image_only",
                bool(_cfg_get(feature_openpi, "rlt_image_only", True)),
                False,
            ),
            (
                "Stage1 rlt_use_mask",
                bool(_cfg_get(feature_openpi, "rlt_use_mask", False)),
                True,
            ),
            (
                "Stage1 rlt_prefix_seq_len",
                int(_cfg_get(feature_openpi, "rlt_prefix_seq_len", 0)),
                1024,
            ),
            (
                "Stage1 rlt_num_layers",
                int(_cfg_get(feature_openpi, "rlt_num_layers", 0)),
                2,
            ),
            (
                "Stage1 rlt_num_heads",
                int(_cfg_get(feature_openpi, "rlt_num_heads", 0)),
                8,
            ),
            (
                "Stage1 rlt_encoder_type",
                str(_cfg_get(feature_openpi, "rlt_encoder_type", "")),
                "append_self_attention",
            ),
        )
        invalid = [
            f"{name}: expected {expected}, got {actual}"
            for name, actual, expected in songling_checks
            if actual != expected
        ]
        expected_repos = {
            "pi05_rlt_songling_all": "songling/all_tasks",
            "pi05_rlt_songling_joint": "songling/garment_folding",
        }
        expected_repo = expected_repos.get(config_name)
        feature_data = _cfg_get(feature_cfg, "openpi_data", {})
        actual_repo = str(_cfg_get(feature_data, "repo_id", "")).strip()
        raw_norm_stats_path = _cfg_get(feature_data, "norm_stats_path", None)
        has_explicit_norm_stats = isinstance(raw_norm_stats_path, str) and bool(
            raw_norm_stats_path.strip()
        )
        if expected_repo is None:
            invalid.append(f"unsupported Stage1 config_name {config_name!r}")
        elif not actual_repo:
            invalid.append("Stage1 openpi_data.repo_id must be non-empty")
        elif actual_repo != expected_repo and not has_explicit_norm_stats:
            invalid.append(
                "Stage1 custom norm-stats repo requires an explicit "
                f"openpi_data.norm_stats_path: expected {expected_repo!r}, "
                f"got {actual_repo!r}"
            )
        if invalid:
            raise ValueError(
                "Invalid Songling RLT configuration: " + "; ".join(invalid)
            )
    if not bool(_cfg_get(feature_openpi, "use_rlt", False)):
        raise ValueError("rollout.rlt_feature_model.openpi.use_rlt must be true.")
    task = str(_cfg_get(feature_openpi, "task", "")).lower()
    if task != "eval":
        raise ValueError(
            "rollout.rlt_feature_model.openpi.task must be 'eval' for frozen "
            f"Stage1 extraction, got {task!r}."
        )


@torch.no_grad()
def predict_rlt_candidates(
    *,
    policy_model: Any,
    feature_model: Any,
    action_codec: Any,
    env_obs: dict[str, Any],
    episode_id: str,
    chunk_id: int,
    actor_ready: bool,
    actor_version: int,
    reference_seed: int,
) -> tuple[DualActionCandidates, dict[str, torch.Tensor]]:
    """Return both physical VLA and Actor chunks without choosing execution mode."""
    device = next(feature_model.parameters()).device
    rng = torch.Generator(device=device)
    rng.manual_seed(int(reference_seed))
    extracted = feature_model.extract_rlt_obs(env_obs, rng=rng)
    vla_action = extracted["ref_chunk"]
    normalized_obs = dict(extracted)
    normalized_obs["ref_chunk"] = action_codec.encode(vla_action, clip=True)
    actor_action, _ = policy_model.predict_action_batch(
        env_obs=normalized_obs, mode="eval", return_obs=True
    )
    if isinstance(actor_action, np.ndarray):
        actor_action = torch.from_numpy(actor_action).to(normalized_obs["z_rl"].device)
    physical_actor = action_codec.decode(actor_action, clip=True)
    checkpoint_version = str(
        getattr(feature_model, "rlt_checkpoint_version", "unknown")
    )
    checkpoint_hash = int.from_bytes(
        hashlib.sha256(checkpoint_version.encode("utf-8")).digest()[:8],
        byteorder="big",
        signed=False,
    ) & ((1 << 63) - 1)

    def first_numpy(value: Any) -> np.ndarray:
        if torch.is_tensor(value):
            value = value.detach().to(torch.float32).cpu().numpy()
        value = np.asarray(value, dtype=np.float32)
        if value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        return value

    candidates = DualActionCandidates(
        episode_id=episode_id,
        chunk_id=chunk_id,
        vla_action=first_numpy(vla_action),
        actor_action=first_numpy(physical_actor),
        actor_ready=actor_ready,
        actor_version=actor_version,
        feature_checkpoint_hash=checkpoint_hash,
        reference_seed=reference_seed,
    )
    replay_obs = {
        key: value.detach().to(torch.float32).cpu()
        for key, value in normalized_obs.items()
        if key in RLT_OBS_KEYS
    }
    return candidates, replay_obs


def _append_rlt_transition_obs(
    *,
    feature_model: Any,
    result: dict[str, Any],
    rlt_obs: dict[str, torch.Tensor],
    final_obs: dict[str, Any] | None,
    action_codec: Any | None = None,
    reference_seed: int | None = None,
) -> None:
    transition_obs = rlt_obs
    if final_obs is not None:
        extract_kwargs = {}
        if reference_seed is not None:
            device = next(feature_model.parameters()).device
            rng = torch.Generator(device=device)
            rng.manual_seed(int(reference_seed))
            extract_kwargs["rng"] = rng
        transition_obs = feature_model.extract_rlt_obs(final_obs, **extract_kwargs)
        if action_codec is not None:
            transition_obs = dict(transition_obs)
            transition_obs["ref_chunk"] = action_codec.encode(
                transition_obs["ref_chunk"], clip=True
            )
    for key in RLT_OBS_KEYS:
        result["forward_inputs"][f"{RLT_TRANSITION_PREFIX}{key}"] = transition_obs[key]


def predict_rlt_actions(
    *,
    policy_model: Any,
    feature_model: Any,
    rlt_route: RLTRoute,
    env_obs: dict[str, Any],
    final_obs: dict[str, Any] | None,
    mode: Literal["train", "eval"],
    version: int = 0,
    rlt_switch_flags: torch.Tensor | None = None,
    intervene_requested: torch.Tensor | None = None,
    expert_model: Any | None = None,
    action_codec: Any | None = None,
    reference_seed: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    with torch.no_grad():
        extract_kwargs = {}
        if reference_seed is not None:
            device = next(feature_model.parameters()).device
            rng = torch.Generator(device=device)
            rng.manual_seed(int(reference_seed))
            extract_kwargs["rng"] = rng
        rlt_obs = feature_model.extract_rlt_obs(env_obs, **extract_kwargs)
        if action_codec is not None:
            rlt_obs = dict(rlt_obs)
            rlt_obs["ref_chunk"] = action_codec.encode(rlt_obs["ref_chunk"], clip=True)
        actions, result = policy_model.predict_action_batch(
            env_obs=rlt_obs,
            mode=mode,
            return_obs=True,
        )
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions)

        route_output = rlt_route.route(
            RLTRouteContext(
                env_obs=env_obs,
                rlt_obs=rlt_obs,
                student_actions=actions,
                result=result,
                mode=mode,
                rlt_switch_flags=rlt_switch_flags,
                intervene_requested=intervene_requested,
                expert_model=expert_model,
                version=version,
            )
        )
        actions = route_output.actions
        result = route_output.result
        batch_size = int(actions.shape[0])
        if reference_seed is not None:
            result["forward_inputs"]["rlt_reference_seed"] = torch.full(
                (batch_size, 1),
                int(reference_seed),
                dtype=torch.int64,
                device=actions.device,
            )
        checkpoint_version = str(
            getattr(feature_model, "rlt_checkpoint_version", "unknown")
        )
        checkpoint_hash = int.from_bytes(
            hashlib.sha256(checkpoint_version.encode("utf-8")).digest()[:8],
            byteorder="big",
            signed=False,
        ) & ((1 << 63) - 1)
        result["forward_inputs"]["rlt_feature_version_hash"] = torch.full(
            (batch_size, 1),
            checkpoint_hash,
            dtype=torch.int64,
            device=actions.device,
        )

        _append_rlt_transition_obs(
            feature_model=feature_model,
            result=result,
            rlt_obs=rlt_obs,
            final_obs=final_obs,
            action_codec=action_codec,
            reference_seed=(
                None if reference_seed is None else int(reference_seed) + 1
            ),
        )

    return actions, result
