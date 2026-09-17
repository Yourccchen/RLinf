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

"""SFT data loader for Xingchen (Astribot-S1) ARIO-format data.

Reads episodes directly from S3/OSS using ArioStreamingDataset (from
openpi_Ario), applies XingchenInputs → openpi model transforms, and yields
(Observation, actions) batches compatible with the openpi_rlinf SFT worker.

The loader bypasses the LeRobot format requirement entirely — no pre-conversion
needed.  S3 credentials are read from the environment
(AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY).
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing
import pathlib
import typing
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from openpi.transforms import compose

logger = logging.getLogger(__name__)

__all__ = [
    "XingchenSftDataConfig",
    "XingchenSftDataLoader",
    "build_xingchen_sft_dataloader",
]

_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
_OPTIONAL_ARIO_CONFIG_DEFAULTS = {
    "index_cache_dir": "",
    "discover_from_data_lake": False,
    "index_workers": 32,
}


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------


def _xingchen_collate(items: list[dict]) -> tuple[Any, torch.Tensor]:
    """Collate XingchenInputs-transformed dicts into (Observation, actions).

    Mirrors the layout that the openpi_rlinf SFT worker expects: an Observation
    namedtuple (or compatible object) plus an [B, action_horizon, action_dim]
    actions tensor.
    """
    from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation

    if not items:
        raise ValueError("Cannot collate an empty Xingchen SFT batch.")

    def _stack(key, dtype=None):
        return torch.from_numpy(
            np.stack([np.asarray(item[key], dtype=dtype) for item in items])
        )

    images = {
        key: torch.from_numpy(
            np.stack([np.asarray(item["image"][key]) for item in items])
        )
        for key in _IMAGE_KEYS
    }
    image_masks = {
        key: torch.from_numpy(
            np.stack(
                [np.asarray(item["image_mask"][key], dtype=np.bool_) for item in items]
            )
        )
        for key in _IMAGE_KEYS
    }
    observation = Observation.from_dict(
        {
            "image": images,
            "image_mask": image_masks,
            "state": _stack("state", np.float32),
            "tokenized_prompt": _stack("tokenized_prompt", np.int64).long(),
            "tokenized_prompt_mask": _stack("tokenized_prompt_mask", np.bool_),
        }
    )
    actions = _stack("actions", np.float32)
    # Ensure [B, action_horizon, action_dim]
    if actions.dim() == 2:
        actions = actions.unsqueeze(1)
    return observation, actions


# ---------------------------------------------------------------------------
# Transformed dataset wrapper
# ---------------------------------------------------------------------------


class _TransformedArioDataset(torch.utils.data.Dataset):
    """Apply the composed openpi model transform to each ARIO sample."""

    def __init__(self, ario_dataset, transform):
        self._dataset = ario_dataset
        self._transform = transform

    def __getitem__(self, idx: int) -> dict:
        return self._transform(self._dataset[idx])

    def __len__(self) -> int:
        return len(self._dataset)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class XingchenSftDataConfig:
    """Resolved data-pipeline metadata for the Xingchen SFT loader."""

    s3_prefixes: str
    action_dim: int
    action_horizon: int
    max_token_len: int


def _create_ario_config(config_cls: type, config_kwargs: dict[str, Any]) -> Any:
    """Create an ARIO config across supported openpi_Ario versions."""
    supported_fields = {field.name for field in dataclasses.fields(config_cls)}
    unsupported_fields = set(config_kwargs) - supported_fields
    unsupported_required = unsupported_fields - _OPTIONAL_ARIO_CONFIG_DEFAULTS.keys()
    unsupported_enabled = {
        name
        for name in unsupported_fields & _OPTIONAL_ARIO_CONFIG_DEFAULTS.keys()
        if config_kwargs[name] != _OPTIONAL_ARIO_CONFIG_DEFAULTS[name]
    }
    if unsupported_required or unsupported_enabled:
        names = ", ".join(sorted(unsupported_required | unsupported_enabled))
        raise RuntimeError(
            "The installed openpi_Ario ArioConfig does not support configured "
            f"options: {names}. Update openpi_Ario or disable these options."
        )
    return config_cls(
        **{
            name: value
            for name, value in config_kwargs.items()
            if name in supported_fields
        }
    )


def _resolve_openpi_data_kwargs(model_cfg: Any) -> dict[str, Any] | None:
    """Resolve data overrides nested under the actor model config."""
    data_kwargs = OmegaConf.select(model_cfg, "openpi_data", default=None)
    if data_kwargs is None:
        return None
    return typing.cast(
        dict[str, Any], OmegaConf.to_container(data_kwargs, resolve=True)
    )


def build_xingchen_sft_dataloader(
    cfg: Any,
    world_size: int,
    rank: int,
    data_paths: Any,
    eval_dataset: bool = False,
) -> tuple["XingchenSftDataLoader", XingchenSftDataConfig]:
    """Build the Xingchen/ARIO SFT data loader.

    Reads ``data.s3_prefixes`` (and optionally ``data.s3_endpoint``) from the
    Hydra config.  Falls back to ``data.train_data_paths[0].dataset_path`` as
    an S3 URI if ``data.s3_prefixes`` is not set.
    """
    from rlinf.models.embodiment.openpi_rlinf.transforms_pipeline import (
        build_openpi_transforms,
    )

    model_cfg = cfg.actor.model
    data_cfg = cfg.data

    # Resolve S3 prefix: explicit config key > prefix file > data_paths entry.
    s3_prefixes: str = str(OmegaConf.select(data_cfg, "s3_prefixes", default="") or "")
    if not s3_prefixes:
        # A newline-delimited file keeps multi-task runs (hundreds of prefixes)
        # out of the YAML.
        prefixes_file = str(
            OmegaConf.select(data_cfg, "s3_prefixes_file", default="") or ""
        )
        if prefixes_file:
            lines = [
                line.strip()
                for line in pathlib.Path(prefixes_file)
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip() and not line.startswith("#")
            ]
            if not lines:
                raise ValueError(
                    f"s3_prefixes_file {prefixes_file!r} contained no prefixes"
                )
            s3_prefixes = ",".join(lines)
            logger.info(
                "Xingchen/Songling SFT: loaded %d prefixes from %s",
                len(lines),
                prefixes_file,
            )
    if not s3_prefixes:
        # Allow dataset_path to be used as an S3 prefix directly.
        from rlinf.data.storage.lerobot import resolve_lerobot_repo_id

        fallback = resolve_lerobot_repo_id(data_paths)
        if fallback and fallback.startswith("s3://"):
            s3_prefixes = fallback
    if not s3_prefixes:
        raise ValueError(
            "Xingchen SFT loader requires data.s3_prefixes (e.g. "
            "'s3://bucket/prefix/') or data.train_data_paths[0].dataset_path "
            "set to an S3 URI."
        )

    s3_endpoint: str = str(OmegaConf.select(data_cfg, "s3_endpoint", default="") or "")
    action_horizon: int = int(model_cfg.num_action_chunks)
    action_dim: int = int(model_cfg.action_dim)
    max_token_len: int = int(model_cfg.openpi.max_token_len)
    task_prompt: str = str(
        OmegaConf.select(
            model_cfg, "openpi_data.default_prompt", default="fold clothes"
        )
        or "fold clothes"
    )
    video_downsample_rate: int = int(
        OmegaConf.select(data_cfg, "video_downsample_rate", default=1)
    )
    min_frames: int = int(OmegaConf.select(data_cfg, "min_frames", default=51))
    max_episodes = OmegaConf.select(data_cfg, "max_episodes", default=None)
    cache_size: int = int(OmegaConf.select(data_cfg, "cache_size", default=32))
    # "songling_canonical55" pulls the 14-D qpos/gripper vector out of
    # state.pt's __canonical55__; "xingchen" expects the older split .pt files.
    data_format: str = str(
        OmegaConf.select(data_cfg, "data_format", default="songling_canonical55")
    )
    action_start_offset: int = int(
        OmegaConf.select(data_cfg, "action_start_offset", default=1)
    )
    filter_episodes_by_state: bool = bool(
        OmegaConf.select(data_cfg, "filter_episodes_by_state", default=True)
    )
    load_instructions: bool = bool(
        OmegaConf.select(data_cfg, "load_instructions", default=True)
    )
    instruction_field: str = str(
        OmegaConf.select(data_cfg, "instruction_field", default="sub_instructions")
    )
    video_reader_cache_size: int = int(
        OmegaConf.select(data_cfg, "video_reader_cache_size", default=16)
    )
    # Decoded-episode disk cache. The ArioStreamingDataset default is /tmp,
    # which is small inside a k8s pod; point it at shared storage instead.
    disk_cache_dir: str = str(
        OmegaConf.select(data_cfg, "disk_cache_dir", default="/tmp/ario_disk_cache")
    )
    disk_cache_max_gb: float = float(
        OmegaConf.select(data_cfg, "disk_cache_max_gb", default=200.0)
    )
    # Frame index location, when it should not sit next to the cached objects.
    # Over the full corpus the index costs ~30min of S3 probes, so it is worth
    # keeping on shared storage even when objects cache on node-local disk.
    index_cache_dir: str = str(
        OmegaConf.select(data_cfg, "index_cache_dir", default="") or ""
    )
    # Skip the S3 listing that confirms all three camera views exist. Over 547
    # tasks that listing walks ~9.5M objects; the data lake already knows which
    # episodes are usable.
    discover_from_data_lake: bool = bool(
        OmegaConf.select(data_cfg, "discover_from_data_lake", default=False)
    )
    index_workers: int = int(OmegaConf.select(data_cfg, "index_workers", default=32))
    batch_size: int = (
        int(cfg.actor.get("eval_batch_size", cfg.actor.micro_batch_size))
        if eval_dataset
        else int(cfg.actor.micro_batch_size)
    )
    num_workers: int = int(OmegaConf.select(data_cfg, "num_workers", default=4))
    seed: int = int(OmegaConf.select(cfg.actor, "seed", default=0))

    # Build the ARIO streaming dataset
    from openpi.datasets.ario_dataset import ArioConfig, ArioStreamingDataset

    ario_cfg = _create_ario_config(
        ArioConfig,
        {
            "s3_prefixes": s3_prefixes,
            "s3_endpoint": s3_endpoint,
            "video_downsample_rate": video_downsample_rate,
            "min_frames": min_frames,
            "task": task_prompt,
            "multi_view": True,
            "max_episodes": max_episodes,
            "cache_size": cache_size,
            "disk_cache_dir": disk_cache_dir,
            "disk_cache_max_gb": disk_cache_max_gb,
            "index_cache_dir": index_cache_dir,
            "discover_from_data_lake": discover_from_data_lake,
            "index_workers": index_workers,
            "data_format": data_format,
            "action_start_offset": action_start_offset,
            "filter_episodes_by_state": filter_episodes_by_state,
            "load_instructions": load_instructions,
            "instruction_field": instruction_field,
            "video_reader_cache_size": video_reader_cache_size,
        },
    )
    # Build the openpi transform pipeline FIRST. It loads norm stats, which is
    # the cheapest thing that can fail; discovering and indexing the episodes
    # below takes ~an hour on the full corpus, so a missing norm_stats.json must
    # not be discovered only after that work is thrown away.
    config_name = str(model_cfg.openpi.config_name)
    model_path = str(model_cfg.model_path)
    data_kwargs = _resolve_openpi_data_kwargs(model_cfg)

    input_transforms, _ = build_openpi_transforms(
        model_path,
        config_name,
        data_kwargs=data_kwargs,
    )

    ario_dataset = ArioStreamingDataset(config=ario_cfg, action_horizon=action_horizon)
    transformed = _TransformedArioDataset(ario_dataset, compose(input_transforms))

    mp_context = multiprocessing.get_context("spawn") if num_workers > 0 else None
    generator = torch.Generator()
    generator.manual_seed(seed + rank)

    torch_loader = torch.utils.data.DataLoader(
        typing.cast(torch.utils.data.Dataset, transformed),
        batch_size=batch_size,
        shuffle=not eval_dataset,
        num_workers=num_workers,
        multiprocessing_context=mp_context,
        persistent_workers=num_workers > 0,
        collate_fn=_xingchen_collate,
        drop_last=True,
        generator=generator,
    )

    data_config = XingchenSftDataConfig(
        s3_prefixes=s3_prefixes,
        action_dim=action_dim,
        action_horizon=action_horizon,
        max_token_len=max_token_len,
    )

    logger.info(
        "Xingchen SFT data loader: s3_prefixes=%s, batch_size=%d, "
        "num_workers=%d, action_horizon=%d, episodes=%d",
        s3_prefixes,
        batch_size,
        num_workers,
        action_horizon,
        len(ario_dataset),
    )

    return XingchenSftDataLoader(torch_loader, data_config), data_config


class XingchenSftDataLoader:
    """Infinite (Observation, actions) loop over the Xingchen/ARIO SFT dataset."""

    def __init__(
        self,
        torch_loader: torch.utils.data.DataLoader,
        data_config: XingchenSftDataConfig,
    ):
        self._torch_loader = torch_loader
        self._data_config = data_config

    def data_config(self) -> XingchenSftDataConfig:
        return self._data_config

    @property
    def torch_loader(self) -> torch.utils.data.DataLoader:
        return self._torch_loader

    def __iter__(self):
        while True:
            yield from self._torch_loader

    def __len__(self) -> int:
        return len(self._torch_loader)
