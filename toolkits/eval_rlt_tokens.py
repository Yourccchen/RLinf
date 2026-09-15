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

"""Measure how well the RLT encoder compresses the VLA prefix.

``rlt_loss`` alone cannot answer that: it is an unnormalized MSE over prefix
embeddings, so its scale depends on the embedding variance rather than on how
much was preserved. This reports four things the training curve does not show.

Reconstruction is scored as explained variance, ``1 - MSE / Var(prefix)``, and
compared against two baselines a useless encoder would still match: predicting
the global mean (EV = 0 by construction) and predicting the per-position mean,
which knows token position but never looks at the frame. An encoder that does
not beat the per-position baseline has only learned a position prior.

Collapse is scored separately because the reconstruction loss cannot detect it:
the decoder also sees the target embeddings, so it can learn to ignore the token
entirely and ``rlt_loss`` still falls. Mean-centered pairwise cosine near zero
means different frames really do produce different tokens.

Temporal smoothness and task separation check that the token space is
structured the way a downstream RL head needs: adjacent frames close together,
different tasks apart.

Usage:

    python toolkits/eval_rlt_tokens.py \\
        --checkpoint /path/to/checkpoints/global_step_20000 \\
        --num-samples 64
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
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-name", default="songling_rlt_stage1_sft_openpi_pi05")
    parser.add_argument(
        "--config-path", default="/home/caimengchen/codebases/RLinf/examples/sft/config"
    )
    parser.add_argument("--num-samples", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-episodes", type=int, default=64)
    parser.add_argument(
        "--frames-per-episode",
        type=int,
        default=4,
        help="Consecutive frames sampled per episode, for the smoothness check",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float32"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def masked_stats(
    reconstructed: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None
) -> tuple[float, float, float]:
    """Return (mse, variance, explained_variance) over valid prefix positions."""
    reconstructed = reconstructed.to(torch.float32)
    target = target.to(torch.float32)
    if mask is None:
        weights = torch.ones(target.shape[:2], device=target.device, dtype=torch.float32)
    else:
        weights = mask.to(device=target.device, dtype=torch.float32)
    weights = weights[..., None]
    denominator = torch.clamp(weights.sum() * target.shape[-1], min=1.0)

    mse = (torch.square(reconstructed - target) * weights).sum() / denominator
    mean = (target * weights).sum() / denominator
    variance = (torch.square(target - mean) * weights).sum() / denominator
    return float(mse), float(variance), float(1.0 - mse / variance)


def main() -> int:
    args = parse_args()

    checkpoint = pathlib.Path(args.checkpoint).expanduser()
    if not (checkpoint / "actor" / "model_state_dict" / "full_weights.pt").is_file():
        sys.exit(f"ERROR: no full_weights.pt under {checkpoint}/actor/model_state_dict/")

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf, open_dict

    with initialize_config_dir(config_dir=args.config_path, version_base=None):
        cfg = compose(config_name=args.config_name)

    with open_dict(cfg):
        cfg.data.max_episodes = int(args.max_episodes)

    print(f"checkpoint : {checkpoint}", flush=True)
    print(f"device     : {args.device}  dtype={args.dtype}", flush=True)

    from rlinf.data.datasets.openpi_rlinf.xingchen import build_xingchen_sft_dataloader
    from rlinf.data.datasets.openpi_rlinf.xingchen.xingchen_sft_data_loader import (
        _xingchen_collate,
    )
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
    wrapper = get_model(cfg.actor.model, torch_dtype=getattr(torch, args.dtype))
    wrapper = wrapper.to(args.device).eval()

    # Sample runs of consecutive frames: the smoothness check needs neighbours,
    # and separate runs stand in for different tasks/episodes.
    rng = np.random.default_rng(args.seed)
    run_length = max(1, args.frames_per_episode)
    num_runs = max(1, args.num_samples // run_length)
    starts = rng.choice(
        len(dataset) - run_length, size=num_runs, replace=False
    )
    indices = np.concatenate([np.arange(s, s + run_length) for s in starts])
    run_id = np.repeat(np.arange(num_runs), run_length)

    tokens: list[torch.Tensor] = []
    recon_mse: list[float] = []
    recon_var: list[float] = []
    recon_ev: list[float] = []
    pos_baseline_ev: list[float] = []

    t0 = time.time()
    for start in range(0, len(indices), args.batch_size):
        chunk = indices[start : start + args.batch_size]
        items = [dataset[int(i)] for i in chunk]
        observation, _ = _xingchen_collate(items)
        observation = wrapper._observation_to_device(observation)

        with torch.no_grad():
            prepared = pi0_model_module.preprocess_observation(observation, train=False)
            prefix_output, prefix_mask, _ = wrapper.model.build_prefix_cache(prepared)
            rlt_output, rlt_mask = wrapper._select_rlt_prefix_embeddings(
                prefix_output, prefix_mask, prepared.tokenized_prompt
            )

            rlt_param = next(wrapper.rlt_module.parameters())
            prefix = rlt_output.to(device=rlt_param.device, dtype=rlt_param.dtype)
            mask = rlt_mask if wrapper.rlt_cfg.rlt_use_mask else None
            reconstructed, rl_tokens = wrapper.rlt_module.reconstruct(prefix, mask)

            mse, variance, ev = masked_stats(reconstructed, prefix, mask)

            # Baseline that never looks at the frame: predict each position by
            # its mean across the batch. Beating this is the real bar.
            per_position = prefix.to(torch.float32).mean(dim=0, keepdim=True)
            _, _, baseline_ev = masked_stats(
                per_position.expand_as(prefix), prefix, mask
            )

        tokens.append(rl_tokens.reshape(rl_tokens.shape[0], -1).float().cpu())
        recon_mse.append(mse)
        recon_var.append(variance)
        recon_ev.append(ev)
        pos_baseline_ev.append(baseline_ev)
        done = min(start + args.batch_size, len(indices))
        print(f"  {done}/{len(indices)} ({(time.time() - t0) / done:.1f}s/sample)", flush=True)

    z_rl = torch.cat(tokens, dim=0)
    run_id_t = torch.as_tensor(run_id[: len(z_rl)])

    print("\n" + "=" * 62)
    print("1. 重建质量  (RLT 的训练目标, 归一化后才有意义)")
    print("=" * 62)
    print(f"  masked MSE            : {np.mean(recon_mse):.4f}")
    print(f"  prefix 方差           : {np.mean(recon_var):.4f}")
    print(f"  解释方差 EV           : {np.mean(recon_ev):+.4f}   (1=无损, 0=等同常量基线)")
    print(f"  逐位置均值基线 EV     : {np.mean(pos_baseline_ev):+.4f}   (不看内容的下限)")
    margin = np.mean(recon_ev) - np.mean(pos_baseline_ev)
    print(f"  超出基线              : {margin:+.4f}")
    print(f"  压缩比                : {prefix.shape[1]}:1  ({prefix.shape[1]} tokens -> 1)")

    print("\n" + "=" * 62)
    print("2. 坍缩检测  (rlt_loss 看不出来的失败模式)")
    print("=" * 62)
    centered = z_rl - z_rl.mean(dim=0, keepdim=True)
    normalized = centered / centered.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    similarity = normalized @ normalized.T
    off_diagonal = similarity[~torch.eye(len(z_rl), dtype=torch.bool)]
    print(f"  两两余弦(去均值)      : 均值={off_diagonal.mean():+.4f}  最大={off_diagonal.max():.4f}")
    # An encoder that ignores its input would leave almost no variance across
    # samples relative to the variance within a token.
    across = z_rl.std(dim=0).mean()
    within = z_rl.std(dim=1).mean()
    print(f"  样本间 std / token 内 std : {across:.4f} / {within:.4f} = {across / within:.4f}")
    print(f"  判定                  : {'未坍缩' if off_diagonal.mean().abs() < 0.3 else '疑似坍缩'}")

    print("\n" + "=" * 62)
    print("3. 结构  (时序平滑 / 不同片段可分)")
    print("=" * 62)
    same = similarity[(run_id_t[:, None] == run_id_t[None, :]) & ~torch.eye(len(z_rl), dtype=torch.bool)]
    diff = similarity[run_id_t[:, None] != run_id_t[None, :]]
    print(f"  同片段相邻帧余弦      : {same.mean():+.4f}")
    print(f"  不同片段余弦          : {diff.mean():+.4f}")
    print(f"  可分度 (同 - 异)      : {same.mean() - diff.mean():+.4f}   (>0 说明有结构)")

    if args.output:
        output = pathlib.Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "z_rl": z_rl,
                "dataset_indices": torch.as_tensor(indices, dtype=torch.long),
                "run_id": run_id_t,
                "explained_variance": float(np.mean(recon_ev)),
                "position_baseline_ev": float(np.mean(pos_baseline_ev)),
                "checkpoint": str(checkpoint),
            },
            output,
        )
        print(f"\nsaved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
