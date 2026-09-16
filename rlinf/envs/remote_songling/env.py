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

from __future__ import annotations

import uuid
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from .client import SonglingRPCClient, SonglingRPCError
from .codec import SonglingActionCodec, decode_config_sequence


class RemoteSonglingEnv(gym.Env):
    """RLinf environment backed by an external Songling hardware RPC service."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        cfg,
        num_envs: int,
        seed_offset: int,
        total_num_processes: int,
        worker_info=None,
    ) -> None:
        del seed_offset, total_num_processes, worker_info
        if int(num_envs) != 1:
            raise ValueError(
                f"RemoteSonglingEnv supports exactly one robot, got {num_envs}."
            )
        if bool(cfg.get("auto_reset", False)):
            raise ValueError(
                "RemoteSonglingEnv requires auto_reset=False for operator-gated reset."
            )
        self.cfg = cfg
        self.num_envs = 1
        self.auto_reset = False
        self.ignore_terminations = bool(cfg.get("ignore_terminations", False))
        self._is_start = True
        self._elapsed_steps = np.zeros(1, dtype=np.int32)
        self._chunk_id = 0
        self._episode_id = ""
        self._policy_version = str(cfg.get("policy_version", "unknown"))
        self._session_id = str(uuid.uuid4())
        self._last_obs: dict[str, Any] | None = None

        rpc_cfg = cfg.rpc
        self.protocol_version = str(rpc_cfg.protocol_version)
        self.client = SonglingRPCClient(
            str(rpc_cfg.endpoint),
            protocol_version=self.protocol_version,
            connect_timeout_s=float(rpc_cfg.get("connect_timeout_s", 10.0)),
            request_timeout_s=float(rpc_cfg.get("request_timeout_s", 30.0)),
            max_message_bytes=int(rpc_cfg.get("max_message_bytes", 64 * 1024 * 1024)),
        )
        self.codec = SonglingActionCodec.from_config(cfg.action_codec)
        self.action_frequency_hz = float(cfg.get("action_frequency_hz", 50.0))
        self.max_chunk_len = int(cfg.get("max_chunk_len", 10))
        self.action_units = decode_config_sequence(
            cfg.get("action_units", []), "action_units"
        )
        if len(self.action_units) != 14:
            raise ValueError("RemoteSonglingEnv action_units must contain 14 entries.")
        self.default_critical_phase = bool(cfg.get("critical_phase_default", True))
        self.task_description = str(cfg.get("task_description", "")).strip()
        if not self.task_description:
            raise ValueError("RemoteSonglingEnv task_description must be non-empty.")

        image_shape = tuple(
            int(v)
            for v in decode_config_sequence(
                cfg.get("image_shape", [240, 320, 3]), "image_shape"
            )
        )
        if len(image_shape) != 3 or image_shape[-1] != 3:
            raise ValueError(f"image_shape must be [H, W, 3], got {image_shape}.")
        self.image_shape = image_shape
        self.camera_keys = {
            "head": str(cfg.get("head_camera_key", "head_camera")),
            "left": str(cfg.get("left_camera_key", "left_camera")),
            "right": str(cfg.get("right_camera_key", "right_camera")),
        }
        self.joint_order = list(
            cfg.get(
                "joint_order",
                [
                    *[f"left_joint_{i}" for i in range(1, 7)],
                    "left_gripper",
                    *[f"right_joint_{i}" for i in range(1, 7)],
                    "right_gripper",
                ],
            )
        )
        if len(self.joint_order) != 14:
            raise ValueError("RemoteSonglingEnv joint_order must contain 14 names.")

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(14,), dtype=np.float32
        )
        image_space = lambda: gym.spaces.Box(
            low=0, high=255, shape=self.image_shape, dtype=np.uint8
        )
        self.observation_space = gym.spaces.Dict(
            {
                "states": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
                ),
                "main_images": image_space(),
                "wrist_images": gym.spaces.Box(
                    low=0,
                    high=255,
                    shape=(2, *self.image_shape),
                    dtype=np.uint8,
                ),
            }
        )
        self._validate_capabilities(self.client.call("health"))

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = bool(value)

    @property
    def elapsed_steps(self) -> np.ndarray:
        return self._elapsed_steps

    @property
    def total_num_group_envs(self) -> int:
        return 1

    def _validate_capabilities(self, capabilities: dict[str, Any]) -> None:
        expected = {
            "protocol_version": self.protocol_version,
            "action_type": "absolute_qpos",
            "action_dim": 14,
        }
        mismatches = [
            f"{key}: expected {value!r}, got {capabilities.get(key)!r}"
            for key, value in expected.items()
            if capabilities.get(key) != value
        ]
        if list(capabilities.get("joint_order", [])) != self.joint_order:
            mismatches.append("joint_order differs from the configured 14-D order")
        if list(capabilities.get("action_units", [])) != self.action_units:
            mismatches.append("action_units differ from the configured 14-D units")
        server_frequency = float(capabilities.get("action_frequency_hz", 0.0))
        if not np.isclose(server_frequency, self.action_frequency_hz):
            mismatches.append(
                f"action_frequency_hz: expected {self.action_frequency_hz}, "
                f"got {server_frequency}"
            )
        if int(capabilities.get("max_chunk_len", 0)) < self.max_chunk_len:
            mismatches.append(
                "server max_chunk_len is smaller than configured max_chunk_len"
            )
        camera_names = set(capabilities.get("camera_names", []))
        missing_cameras = set(self.camera_keys.values()) - camera_names
        if missing_cameras:
            mismatches.append(f"missing cameras {sorted(missing_cameras)}")
        if not bool(capabilities.get("healthy", False)):
            mismatches.append("server reports unhealthy")
        if mismatches:
            raise ValueError(
                "Incompatible Songling RPC capabilities: " + "; ".join(mismatches)
            )

    def _observation(self, payload: dict[str, Any]) -> dict[str, Any]:
        freshness = payload.get("freshness", {})
        required_fresh = ["state", *self.camera_keys.values()]
        stale = [name for name in required_fresh if freshness.get(name) is not True]
        if stale:
            raise ValueError(f"Songling RPC returned stale/missing streams: {stale}.")
        timestamps = payload.get("timestamps", {})
        missing_timestamps = [name for name in required_fresh if name not in timestamps]
        if missing_timestamps:
            raise ValueError(
                f"Songling RPC observation lacks timestamps for {missing_timestamps}."
            )

        state = np.asarray(payload.get("state"), dtype=np.float32)
        if state.shape != (14,) or not np.isfinite(state).all():
            raise ValueError(
                f"Songling state must be finite float32[14], got {state.shape}."
            )
        images = {}
        for logical_name, wire_name in self.camera_keys.items():
            image = np.asarray(payload.get(wire_name))
            if image.shape != self.image_shape or image.dtype != np.uint8:
                raise ValueError(
                    f"Songling camera {wire_name!r} must be uint8{self.image_shape}, "
                    f"got dtype={image.dtype}, shape={image.shape}."
                )
            images[logical_name] = image
        instruction = str(payload.get("instruction", self.task_description)).strip()
        if not instruction:
            raise ValueError("Songling observation instruction must be non-empty.")
        return {
            "states": torch.from_numpy(state[None].copy()),
            "main_images": torch.from_numpy(images["head"][None].copy()),
            "wrist_images": torch.from_numpy(
                np.stack([images["left"], images["right"]], axis=0)[None]
            ),
            "task_descriptions": [instruction],
        }

    def reset(self, *, seed=None, options=None, reset_state_ids=None, env_idx=None):
        del seed, reset_state_ids, env_idx
        options = options or {}
        result = self.client.call(
            "reset",
            {
                "session_id": self._session_id,
                "instruction": str(options.get("instruction", self.task_description)),
            },
        )
        self._episode_id = str(result.get("episode_id", ""))
        if not self._episode_id:
            raise ValueError("Songling reset response must include episode_id.")
        observation_payload = result.get("observation")
        if observation_payload is None:
            observation_payload = self.client.call(
                "observe", {"session_id": self._session_id}
            ).get("observation")
        if not isinstance(observation_payload, dict):
            raise ValueError("Songling reset/observe response lacks observation.")
        self._last_obs = self._observation(observation_payload)
        self._elapsed_steps[:] = 0
        self._chunk_id = 0
        return self._last_obs, {
            "episode_id": self._episode_id,
            "rlt_switch_flags": torch.tensor(
                [[self.default_critical_phase]], dtype=torch.bool
            ),
        }

    @staticmethod
    def _pad(values: np.ndarray, length: int, fill_value=0) -> np.ndarray:
        if values.shape[0] == length:
            return values
        pad_shape = (length - values.shape[0], *values.shape[1:])
        padding = np.full(pad_shape, fill_value, dtype=values.dtype)
        return np.concatenate([values, padding], axis=0)

    def _failure_chunk(self, chunk_len: int, exc: Exception):
        if self._last_obs is None:
            raise exc
        rewards = torch.zeros((1, chunk_len), dtype=torch.float32)
        terminations = torch.zeros((1, chunk_len), dtype=torch.bool)
        truncations = torch.zeros((1, chunk_len), dtype=torch.bool)
        truncations[0, 0] = True
        info = {
            "reason": type(exc).__name__,
            "message": str(exc),
            "valid_step_mask": torch.zeros((1, chunk_len), dtype=torch.bool),
            "executed_actions": torch.zeros((1, chunk_len * 14)),
            "intervene_action": torch.zeros((1, chunk_len * 14)),
            "intervene_flag": torch.zeros((1, chunk_len), dtype=torch.bool),
            "rlt_switch_flags": torch.zeros((1, chunk_len), dtype=torch.bool),
            "final_observation": self._last_obs,
        }
        return (
            [self._last_obs for _ in range(chunk_len)],
            rewards,
            terminations,
            truncations,
            [info for _ in range(chunk_len)],
        )

    def chunk_step(self, chunk_actions):
        normalized = np.asarray(
            (
                chunk_actions.detach().cpu()
                if torch.is_tensor(chunk_actions)
                else chunk_actions
            ),
            dtype=np.float32,
        )
        if (
            normalized.ndim != 3
            or normalized.shape[0] != 1
            or normalized.shape[2] != 14
        ):
            raise ValueError(
                "RemoteSonglingEnv expects actions [1, K, 14], got "
                f"{normalized.shape}."
            )
        chunk_len = int(normalized.shape[1])
        if not 1 <= chunk_len <= self.max_chunk_len:
            raise ValueError(
                f"Songling chunk length must be in [1, {self.max_chunk_len}], "
                f"got {chunk_len}."
            )
        physical = self.codec.decode(normalized[0], clip=True).astype(np.float32)
        self._chunk_id += 1
        try:
            result = self.client.call(
                "chunk_step",
                {
                    "session_id": self._session_id,
                    "episode_id": self._episode_id,
                    "chunk_id": self._chunk_id,
                    "actions": physical,
                    "action_frequency_hz": self.action_frequency_hz,
                    "policy_version": self._policy_version,
                },
            )
            observation_payloads = result.get("observations", [])
            if not isinstance(observation_payloads, list) or not observation_payloads:
                raise ValueError(
                    "Songling chunk_step must return non-empty observations."
                )
            executed_physical = np.asarray(
                result.get("executed_actions"), dtype=np.float32
            )
            actual_len = len(observation_payloads)
            if not 1 <= actual_len <= chunk_len:
                raise ValueError(
                    f"Songling response length must be in [1, {chunk_len}], got {actual_len}."
                )
            if executed_physical.shape != (actual_len, 14):
                raise ValueError(
                    "executed_actions must match observations as [K,14], got "
                    f"{executed_physical.shape}."
                )
            rewards = np.asarray(result.get("rewards"), dtype=np.float32).reshape(-1)
            terminated = np.asarray(result.get("terminated"), dtype=bool).reshape(-1)
            truncated = np.asarray(result.get("truncated"), dtype=bool).reshape(-1)
            for name, values in (
                ("rewards", rewards),
                ("terminated", terminated),
                ("truncated", truncated),
            ):
                if values.shape != (actual_len,):
                    raise ValueError(f"{name} must have shape ({actual_len},).")
            terminal = terminated | truncated
            if terminal[:-1].any():
                raise ValueError("Songling RPC returned steps after a terminal step.")
            if actual_len < chunk_len and not terminal[-1]:
                raise ValueError(
                    "A short Songling chunk must terminate or truncate at its last step."
                )
            if terminal[-1] and not isinstance(result.get("final_observation"), dict):
                raise ValueError(
                    "Terminal Songling responses require final_observation."
                )
            valid = np.asarray(
                result.get("valid_step_mask", np.ones(actual_len, dtype=bool)),
                dtype=bool,
            ).reshape(-1)
            if valid.shape != (actual_len,) or not valid.all():
                raise ValueError(
                    "Returned observations must describe exactly the valid prefix; "
                    "valid_step_mask entries for that prefix must all be true."
                )
            observations = [self._observation(item) for item in observation_payloads]
        except (SonglingRPCError, ValueError, TypeError) as exc:
            return self._failure_chunk(chunk_len, exc)

        executed = self.codec.encode(executed_physical, clip=True).astype(np.float32)
        human_physical = result.get("human_actions")
        if human_physical is None:
            human = np.zeros((actual_len, 14), dtype=np.float32)
        else:
            human_physical = np.asarray(human_physical, dtype=np.float32)
            if human_physical.shape != (actual_len, 14):
                return self._failure_chunk(
                    chunk_len,
                    ValueError(
                        f"human_actions must have shape ({actual_len},14), "
                        f"got {human_physical.shape}."
                    ),
                )
            human = self.codec.encode(human_physical, clip=True).astype(np.float32)
        intervene = np.asarray(
            result.get("intervene_flags", np.zeros(actual_len, dtype=bool)), dtype=bool
        ).reshape(-1)
        if intervene.shape != (actual_len,):
            return self._failure_chunk(
                chunk_len, ValueError("intervene_flags length mismatch.")
            )
        critical = np.asarray(
            result.get(
                "rlt_switch_flags",
                np.full(actual_len, self.default_critical_phase, dtype=bool),
            ),
            dtype=bool,
        ).reshape(-1)
        if critical.shape != (actual_len,):
            return self._failure_chunk(
                chunk_len, ValueError("rlt_switch_flags length mismatch.")
            )

        valid_padded = self._pad(valid, chunk_len, False)
        rewards_padded = self._pad(rewards, chunk_len, 0.0)
        terminated_padded = self._pad(terminated, chunk_len, False)
        truncated_padded = self._pad(truncated, chunk_len, False)
        executed_padded = self._pad(executed, chunk_len, 0.0)
        human_padded = self._pad(human, chunk_len, 0.0)
        intervene_padded = self._pad(intervene, chunk_len, False)
        critical_padded = self._pad(critical, chunk_len, False)

        final_payload = result.get("final_observation")
        final_obs = (
            self._observation(final_payload)
            if isinstance(final_payload, dict)
            else observations[-1]
        )
        self._last_obs = final_obs
        self._elapsed_steps += actual_len
        if actual_len < chunk_len:
            observations.extend([final_obs] * (chunk_len - actual_len))

        info = {
            "episode_id": self._episode_id,
            "chunk_id": self._chunk_id,
            "outcome": result.get("outcome", "continue"),
            "reason": result.get("reason", ""),
            "valid_step_mask": torch.from_numpy(valid_padded[None]),
            "executed_actions": torch.from_numpy(executed_padded.reshape(1, -1)),
            "intervene_action": torch.from_numpy(human_padded.reshape(1, -1)),
            "intervene_flag": torch.from_numpy(intervene_padded[None]),
            "rlt_switch_flags": torch.from_numpy(critical_padded[None]),
            "final_observation": final_obs,
        }
        return (
            observations,
            torch.from_numpy(rewards_padded[None]),
            torch.from_numpy(terminated_padded[None]),
            torch.from_numpy(truncated_padded[None]),
            [info for _ in range(chunk_len)],
        )

    def set_policy_version(self, version: Any) -> None:
        self._policy_version = str(version)

    def get_hold_actions(self, fallback_actions=None) -> np.ndarray:
        if self._last_obs is not None:
            physical = self._last_obs["states"].numpy()
            return self.codec.encode(physical, clip=True).astype(np.float32)
        if fallback_actions is not None:
            return np.asarray(fallback_actions, dtype=np.float32)
        return np.zeros((1, 14), dtype=np.float32)

    def offload(self) -> None:
        return None

    def onload(self) -> None:
        return None

    def close(self) -> None:
        try:
            self.client.call(
                "close",
                {"session_id": self._session_id, "episode_id": self._episode_id},
            )
        except SonglingRPCError:
            pass
        finally:
            self.client.close()
