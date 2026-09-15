#!/usr/bin/env python
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

"""Extract RLT tokens (``z_rl``) from a Stage-1 RLT SFT checkpoint.

Stage-1 trains the RLT encoder-decoder to compress the VLA prefix -- the image
and language hidden states pi0.5 attends over -- into a single token. Stage-2 RL
consumes that token instead of the raw prefix, so this script produces exactly
what the Stage-2 head would see: it runs the deterministic eval prefix path
(``train=False``, no flow-matching noise), not the training forward.

Usage:

    python toolkits/extract_rlt_tokens.py \\
        --checkpoint /path/to/checkpoints/global_step_17500 \\
        --config-name songling_rlt_stage1_sft_openpi_pi05 \\
        --num-samples 16 \\
        --output /path/to/rlt_tokens.pt

The output holds ``z_rl`` of shape ``[num_samples, rlt_embed_dim]`` alongside
the prompts and dataset indices each token came from, so a token can be traced
back to the frame that produced it.
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="global_step_<N> directory (it must contain actor/model_state_dict/full_weights.pt)",
    )
    parser.add_argument(
        "--config-name",
        default="songling_rlt_stage1_sft_openpi_pi05",
        help="Hydra config the checkpoint was trained with",
    )
    parser.add_argument(
        "--config-path",
        default="/home/caimengchen/codebases/RLinf/examples/sft/config",
        help="Directory holding the Hydra config",
    )
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Frames per forward pass; keep small on CPU",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=32,
        help="Cap episode discovery so the dataset builds quickly",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        choices=("bfloat16", "float32"),
        help=(
            "Compute dtype. Default bfloat16 matches the fsdp_config "
            "mixed_precision.param_dtype the run trained under; the fp32 in "
            "actor.model.precision is the master-weight dtype, which only FSDP "
            "casts down at forward time."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    checkpoint = pathlib.Path(args.checkpoint).expanduser()
    weights = checkpoint / "actor" / "model_state_dict" / "full_weights.pt"
    if not weights.is_file():
        sys.exit(f"ERROR: no full_weights.pt under {checkpoint}/actor/model_state_dict/")

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    with initialize_config_dir(config_dir=args.config_path, version_base=None):
        cfg = compose(config_name=args.config_name)

    # The weights come from the checkpoint, but the norm stats live beside the
    # base model. Rather than override both, build the transforms first while
    # model_path still points at the base model, then repoint it: get_model uses
    # model_path only to resolve actor/model_state_dict/full_weights.pt.
    base_model_path = pathlib.Path(str(cfg.actor.model.model_path)).expanduser()
    asset_id = str(OmegaConf.select(cfg.actor.model, "openpi_data.repo_id"))
    norm_stats_dir = base_model_path / asset_id
    if not (norm_stats_dir / "norm_stats.json").is_file():
        sys.exit(f"ERROR: no norm_stats.json under {norm_stats_dir}")

    with open_dict(cfg):
        cfg.data.max_episodes = int(args.max_episodes)

    if not bool(OmegaConf.select(cfg.actor.model, "openpi.use_rlt", default=False)):
        sys.exit(f"ERROR: {args.config_name} was not trained with openpi.use_rlt=True")

    print(f"checkpoint : {checkpoint}", flush=True)
    print(f"config     : {args.config_name}", flush=True)
    print(f"norm stats : {norm_stats_dir}", flush=True)
    print(f"device     : {args.device}", flush=True)
    print(f"dtype      : {args.dtype}", flush=True)

    from rlinf.data.datasets.openpi_rlinf.xingchen import build_xingchen_sft_dataloader
    from rlinf.models.embodiment.openpi_rlinf import get_model
    from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as pi0_model_module

    print("building dataset ...", flush=True)
    loader, _ = build_xingchen_sft_dataloader(
        cfg, world_size=1, rank=0, data_paths=None, eval_dataset=True
    )
    dataset = loader.torch_loader.dataset

    with open_dict(cfg):
        cfg.actor.model.model_path = str(checkpoint)

    print("loading model ...", flush=True)
    t0 = time.time()
    wrapper = get_model(cfg.actor.model, torch_dtype=getattr(torch, args.dtype))
    wrapper = wrapper.to(args.device).eval()
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(dataset), size=args.num_samples, replace=False)

    from rlinf.data.datasets.openpi_rlinf.xingchen.xingchen_sft_data_loader import (
        _xingchen_collate,
    )

    tokens: list[torch.Tensor] = []
    prompts: list[str] = []
    t0 = time.time()
    for start in range(0, args.num_samples, args.batch_size):
        chunk = indices[start : start + args.batch_size]
        items = [dataset[int(i)] for i in chunk]
        observation, _ = _xingchen_collate(items)
        observation = wrapper._observation_to_device(observation)

        with torch.no_grad():
            # train=False: deterministic prefix, matching what Stage-2 reads.
            prepared = pi0_model_module.preprocess_observation(observation, train=False)
            prefix_output, prefix_mask, _ = wrapper.model.build_prefix_cache(prepared)
            rlt_output, rlt_mask = wrapper._select_rlt_prefix_embeddings(
                prefix_output, prefix_mask, prepared.tokenized_prompt
            )
            z_rl = wrapper._encode_rlt_flat(rlt_output, rlt_mask).to(torch.float32)

        tokens.append(z_rl.cpu())
        prompts.extend(str(item.get("prompt", "")) for item in items)
        done = min(start + args.batch_size, args.num_samples)
        print(
            f"  {done}/{args.num_samples} samples  "
            f"({(time.time() - t0) / done:.1f}s/sample)",
            flush=True,
        )

    z_rl = torch.cat(tokens, dim=0)

    output = pathlib.Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "z_rl": z_rl,
            "dataset_indices": torch.as_tensor(indices, dtype=torch.long),
            "prompts": prompts,
            "checkpoint": str(checkpoint),
            "config_name": args.config_name,
        },
        output,
    )

    print(f"\nz_rl shape : {tuple(z_rl.shape)}  dtype={z_rl.dtype}")
    print(f"  mean={z_rl.mean():.4f}  std={z_rl.std():.4f}")
    print(f"  min={z_rl.min():.4f}  max={z_rl.max():.4f}")
    norms = z_rl.norm(dim=-1)
    print(f"  L2 norm: mean={norms.mean():.3f}  min={norms.min():.3f}  max={norms.max():.3f}")

    # Tokens that are identical across different frames would mean the encoder
    # collapsed, which no loss curve would have revealed.
    if z_rl.shape[0] > 1:
        centered = z_rl - z_rl.mean(dim=0, keepdim=True)
        normalized = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        similarity = normalized @ normalized.T
        off_diagonal = similarity[~torch.eye(len(z_rl), dtype=torch.bool)]
        print(
            f"  pairwise cosine (mean-centered): mean={off_diagonal.mean():.3f}  "
            f"max={off_diagonal.max():.3f}"
        )

    print(f"\nsaved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
