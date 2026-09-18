#!/usr/bin/env python
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build an offline Songling RLT replay checkpoint from labeled episodes.

Input is a torch file containing either a list of episodes or
``{"episodes": [...]}``. Each episode contains ``observations`` (T+1 RLinf or
RPC observation mappings), physical ``executed_actions[T,14]``, ``rewards[T]``,
``terminated[T]`` and ``truncated[T]``. Pure demonstrations without explicit
terminal labels are rejected.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict

from rlinf.data.rlt import build_offline_rlt_trajectories
from rlinf.data.storage.replay import TrajectoryReplayBuffer
from rlinf.envs.remote_songling import SonglingActionCodec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", required=True)
    parser.add_argument("--stage2-config", required=True)
    parser.add_argument("--stage1-checkpoint", required=True)
    parser.add_argument("--action-low", required=True, help="JSON array[14]")
    parser.add_argument("--action-high", required=True, help="JSON array[14]")
    parser.add_argument("--output", required=True)
    parser.add_argument("--prompt", default="fold clothes")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--transition-stride", type=int, default=2)
    parser.add_argument("--chunk-len", type=int, default=50)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _standard_observation(obs: dict[str, Any], prompt: str) -> dict[str, Any]:
    if "states" in obs:
        return {
            "states": torch.as_tensor(obs["states"]).reshape(14),
            "main_images": torch.as_tensor(obs["main_images"]),
            "wrist_images": torch.as_tensor(obs["wrist_images"]),
            "task_description": str(
                obs.get("task_description", obs.get("instruction", prompt))
            ),
        }
    return {
        "states": torch.as_tensor(obs["state"]).reshape(14),
        "main_images": torch.as_tensor(obs["head_camera"]),
        "wrist_images": torch.stack(
            [torch.as_tensor(obs["left_camera"]), torch.as_tensor(obs["right_camera"])],
            dim=0,
        ),
        "task_description": str(obs.get("instruction", prompt)),
    }


def _batch_observations(observations: list[dict[str, Any]], prompt: str) -> dict:
    standardized = [_standard_observation(obs, prompt) for obs in observations]
    return {
        "states": torch.stack([obs["states"] for obs in standardized]),
        "main_images": torch.stack([obs["main_images"] for obs in standardized]),
        "wrist_images": torch.stack([obs["wrist_images"] for obs in standardized]),
        "task_descriptions": [obs["task_description"] for obs in standardized],
    }


def _extract_features(model, observations, *, batch_size: int, seed: int, prompt: str):
    features = []
    device = next(model.parameters()).device
    for start in range(0, len(observations), batch_size):
        batch = _batch_observations(observations[start : start + batch_size], prompt)
        generator = torch.Generator(device=device).manual_seed(seed + start)
        output = model.extract_rlt_obs(batch, rng=generator)
        for index in range(len(batch["task_descriptions"])):
            features.append(
                {
                    key: value[index].detach().cpu()
                    for key, value in output.items()
                    if key in ("z_rl", "proprio", "ref_chunk")
                }
            )
    return features


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    output = pathlib.Path(args.output).expanduser()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {output}")
        shutil.rmtree(output)

    payload = torch.load(args.episodes, map_location="cpu", weights_only=False)
    episodes = payload.get("episodes") if isinstance(payload, dict) else payload
    if not isinstance(episodes, list) or not episodes:
        raise ValueError("Episode file must contain a non-empty episode list.")

    cfg = OmegaConf.load(args.stage2_config)
    if OmegaConf.select(cfg, "rollout.rlt_feature_model", default=None) is None:
        raise ValueError(
            "--stage2-config must contain rollout.rlt_feature_model; use the online "
            "Songling Stage2 config, not the offline-only learner config."
        )
    with open_dict(cfg):
        cfg.rollout.rlt_feature_model.model_path = str(
            pathlib.Path(args.stage1_checkpoint).expanduser()
        )
        cfg.rollout.rlt_feature_model.openpi_data.default_prompt = args.prompt
    feature_cfg = OmegaConf.create(
        OmegaConf.to_container(cfg.rollout.rlt_feature_model, resolve=True)
    )

    from rlinf.models import get_model

    model = get_model(feature_cfg).to(args.device).eval()
    model.requires_grad_(False)
    codec = SonglingActionCodec(
        json.loads(args.action_low), json.loads(args.action_high)
    )
    output.mkdir(parents=True, exist_ok=False)
    replay = TrajectoryReplayBuffer(
        seed=args.seed,
        enable_cache=False,
        sample_window_size=0,
        auto_save=True,
        auto_save_path=str(output),
    )
    total = 0
    for episode_index, episode in enumerate(episodes):
        observations = episode.get("observations")
        if not isinstance(observations, list):
            raise ValueError(f"Episode {episode_index} lacks observations list.")
        features = _extract_features(
            model,
            observations,
            batch_size=args.batch_size,
            seed=args.seed + episode_index * 1_000_000,
            prompt=args.prompt,
        )
        for feature in features:
            feature["ref_chunk"] = codec.encode(feature["ref_chunk"], clip=True)
        converted = dict(episode)
        converted["executed_actions"] = codec.encode(
            np.asarray(episode["executed_actions"], dtype=np.float32), clip=True
        )
        trajectories = build_offline_rlt_trajectories(
            converted,
            features,
            chunk_len=args.chunk_len,
            transition_stride=args.transition_stride,
            model_weights_id=f"offline_ep_{episode_index}",
        )
        replay.add_trajectories(trajectories)
        total += len(trajectories)
        print(f"episode {episode_index}: {len(trajectories)} transitions", flush=True)

    replay.close()
    print(f"saved {total} transitions to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
