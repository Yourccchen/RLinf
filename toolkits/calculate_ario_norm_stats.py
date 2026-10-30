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

"""Compute raw ARIO state/action statistics using an RLinf SFT configuration.

Run with ``python -m toolkits.calculate_ario_norm_stats --config-name NAME``.
The default visits every training frame without decoding video. ``--max-frames``
selects a reproducible uniform subset using the actor seed. AWS credentials
must be available in the environment or the default boto3 credential chain.
The output is ``actor.model.openpi_data.norm_stats_path``; no model weights or
existing normalization statistics are loaded.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from openpi.shared import normalize
from tqdm import tqdm

from rlinf.utils.logging import get_logger

logger = get_logger()


def compute_statistics(
    samples: Iterable[dict], action_dim: int
) -> dict[str, normalize.NormStats]:
    """Accumulate unpadded raw states and complete action chunks.

    Each sample contains ``observation/state`` and ``actions``. Every future
    action in a chunk contributes, including repeated episode-end targets,
    matching the weighting used by the SFT objective.
    """
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    pending = {key: [] for key in stats}

    def flush() -> None:
        for key in stats:
            if pending[key]:
                values = np.concatenate(pending[key], axis=0)
                stats[key].update(values)
                pending[key].clear()

    for sample in samples:
        for key, source in (("state", "observation/state"), ("actions", "actions")):
            values = np.asarray(sample[source], dtype=np.float64)
            if values.ndim < 1 or values.shape[-1] != action_dim:
                raise ValueError(f"{key} must have {action_dim} dimensions.")
            if not np.isfinite(values).all():
                raise ValueError(f"{key} contains non-finite values.")
            pending[key].append(values.reshape(-1, action_dim))
        if len(pending["state"]) >= 128:
            flush()
    flush()
    return {key: value.get_statistics() for key, value in stats.items()}


def main() -> None:
    """Read the training config and atomically save task-specific statistics."""
    from openpi.datasets.ario_dataset import ArioConfig

    from rlinf.data.datasets.openpi_rlinf.xingchen.ario_state_dataset import (
        create_ario_dataset,
    )
    from rlinf.data.datasets.openpi_rlinf.xingchen.xingchen_sft_data_loader import (
        _create_ario_config,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames < 2:
        parser.error("--max-frames must be at least 2")

    config_dir = Path(__file__).resolve().parents[1] / "examples/sft/config"
    os.environ.setdefault("EMBODIED_PATH", str(config_dir.parent))
    with initialize_config_dir(version_base="1.1", config_dir=str(config_dir)):
        cfg = compose(config_name=args.config_name)
    output = Path(cfg.actor.model.openpi_data.norm_stats_path).expanduser()
    if output.name != "norm_stats.json":
        raise ValueError("norm_stats_path must point to a norm_stats.json file.")

    # Use the same data options as training, with only video decoding disabled.
    # Non-ARIO fields such as num_workers and train_data_paths belong to RLinf.
    ario_fields = {field.name for field in dataclasses.fields(ArioConfig)}
    kwargs = {
        key: OmegaConf.to_container(value, resolve=True)
        if OmegaConf.is_config(value)
        else value
        for key, value in cfg.data.items()
        if key in ario_fields
    }
    if not kwargs.get("s3_prefixes"):
        raise ValueError("Statistics require explicit data.s3_prefixes.")
    kwargs.update(
        task=str(cfg.actor.model.openpi_data.default_prompt),
        multi_view=True,
        skip_video=True,
    )
    ario_config = _create_ario_config(ArioConfig, kwargs)

    # Match the loader's environment-based credential handling without logging
    # credentials or replacing credentials already supplied by the caller.
    import boto3

    for aws_name, ali_name in (
        ("AWS_ACCESS_KEY_ID", "ALIBABA_ACCESS_KEY_ID"),
        ("AWS_SECRET_ACCESS_KEY", "ALIBABA_ACCESS_KEY_SECRET"),
    ):
        if os.environ.get(ali_name):
            os.environ.setdefault(aws_name, os.environ[ali_name])
    credentials = boto3.Session().get_credentials()
    if credentials is not None:
        frozen = credentials.get_frozen_credentials()
        os.environ.setdefault("AWS_ACCESS_KEY_ID", frozen.access_key)
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", frozen.secret_key)
        if frozen.token:
            os.environ.setdefault("AWS_SESSION_TOKEN", frozen.token)

    dataset = create_ario_dataset(ario_config, int(cfg.actor.model.num_action_chunks))
    if len(dataset) < 2:
        raise ValueError("At least two training frames are required for statistics.")
    indices = range(len(dataset))
    if args.max_frames is not None and args.max_frames < len(dataset):
        rng = np.random.default_rng(int(cfg.actor.seed))
        indices = np.sort(rng.choice(len(dataset), args.max_frames, replace=False))
    samples = (dataset[int(index)] for index in tqdm(indices, desc="ARIO norm stats"))
    stats = compute_statistics(samples, int(cfg.actor.model.action_dim))
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(normalize.serialize_json(stats), encoding="utf-8")
    temporary.replace(output)
    OmegaConf.save(cfg, output.parent / "norm_stats_config.yaml", resolve=True)
    logger.info("Saved statistics from %d frames to %s", len(indices), output)


if __name__ == "__main__":
    main()
