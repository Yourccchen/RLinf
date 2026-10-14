# HALO-WA with frozen Soma VAE observations

This optional SsEval backend keeps the existing `rlt_td3_mlp_policy` and
`InProcessRLTTD3Learner` intact. The MLP configuration remains the default.
Use `learner_type: halo_wa` together with `actor_model.model_type: halo_wa_policy`
and RWI `soma.feature_mode: vae_tokens` to select the new path. Mixing these
model/learner pairs is rejected. MLP checkpoints and replay are not convertible
HALO-WA checkpoints.

## Architecture and source

The attention components and BC/TD3 loss functions are vendored from
[official HALO-WA](https://github.com/YeanRoot/HALO-WA/tree/611008193fd3438bf99cc8f8377858fbdfcb9555)
at commit `611008193fd3438bf99cc8f8377858fbdfcb9555` under Apache-2.0.
`rlinf/models/embodiment/halo_wa/attention.py` corresponds to upstream
`rlinf/rlinf/models/embodiment/adapter/gigawa_actor_critic.py`;
`losses.py` corresponds to upstream `rlinf/rlinf/algorithms/gigawa_losses.py`.
See repository [LICENSE](../LICENSE).

The default hidden width is 512 with 8 attention heads and a 12-step action
chunk. The Actor uses self-attention then visual cross-attention. The critic
uses cross-attention and twin Q heads, following the implementation (it does
not add the extra critic self-attention shown in the paper illustration).
The final action dimension is still 14, not 512.

The frozen Soma `video_condition` observation latent replaces GigaWA's visual
latent. RWI selects the last observation timestep, retains spatial tokens in
view/row/column order, and supplies `[B,N,C]` through `z_rl`. No learned RLT
compressor or future diffusion action latent is used. `C` is the actual VAE
channel count, independently projected to 512. The feature specification must declare each view’s VAE `spatial_shapes`
(`[H,W]`); these dimensions are included in checkpoint identity and checked on
every observation, including the first after resume. Only `temporal: last` is
supported. The configured token maximum is checked before replay insertion.

The visual projection and positional embeddings belong to the Actor optimizer
and are published atomically with Actor weights. Critic updates detach visual
features; Actor updates differentiate through the frozen critic into actions
and visual features. Target actor/critic heads use the current visual encoder,
as upstream does. The frozen Soma model/VAE receives no gradients.

## Training contract

`collect` stores feedback and always reports `actor_ready=false`. On a normal
close it saves the final `collect_N` checkpoint even between save intervals.
Collection saves beneath a timestamped run directory under `save_dir`.

`online` requires `demo_checkpoint` (a qualified HALO collection directory) or
a complete `actor_checkpoint` for resume. BC warmup uses demonstrations only;
then every batch contains exactly 50% demonstrations and 50% online replay.
Small buffers are sampled with replacement to retain this ratio. Choose
successful/useful collection runs as demos; loading a run does not judge its
quality or filter failures automatically.

The Actor minimizes `-min(Q1,Q2) + bc_coef * BC`; BC warmup uses BC alone.
The official Q upper-bound clamp and overshoot penalty are enabled. Online
updates use 5 critic updates per new transition, with the Actor updated every
2 critic steps. Target heads use Polyak averaging. These values are editable
in the [example](../examples/embodiment/config/soma_halo_wa_sseval.yaml).

SsEval-specific adaptations are explicit:

- Inputs retain the original WA reference during human takeover. The BC target
  uses the executed human action on intervention steps and the WA reference
  otherwise. Executed actions train the critic.
- Valid-step masks exclude padding from BC and reward sums. Like upstream,
  chunk rewards are summed and bootstrap uses `gamma**chunk_len`. Termination,
  truncation or a partial chunk disables bootstrap because this transport does
  not provide a reliable continuation for those boundaries.
- Output uses `tanh` in the existing Songling normalized `[-1,1]` codec, then
  decodes to physical absolute qpos: joint radians and gripper millimetres.
  This differs from GigaWA's dataset-derived action bound normalization.
- The existing single-process SsEval replay/transport replaces the official
  distributed workers. This is an integration of the heads and training rule,
  not a reproduction of the paper's task results or full distributed stack.

Checkpoints contain both heads/targets/optimizers, demo and online replay,
warmup progress, both sampler RNGs and feature identity. Full resume requires
identical learner configuration and feature identity. Feature identity combines
the declared frozen asset identity, camera/token spec, channel count, horizon
and codec bounds; the operator must update the declared identity when weights
or preprocessing change. It does not independently hash model files.
Replay-only restore is rejected because it lacks this identity contract.

## Configuration and prerequisites

Use RWI `policy/config_rlinf_soma_halo_wa.yaml` and
`configs/robots/piper_async_qpos_halo_wa.py`. The profile uses 12-step feedback
with stride 12 and strict sequential execution. Its inherited host, Leader
ports and execution rate must match the deployed Piper and Soma training
frequency before launch. The existing MLP profiles/configurations are unchanged.

From the `ss-embodied` workspace, select `envs/ali.yaml` through the existing
`python -m ssemb.cli` workflow. This integration does not create a replacement
launch command. Supply the frozen Soma config/checkpoint and required VAE,
text-encoder and camera assets as described by RWI's `docs/piper_pt3_soma.md`.
Replace the RWI template paths and feature identity; set `SOMA_VAE_CHANNELS` to
the inspected latent channel count. Set `feature_spec.spatial_shapes` from the measured per-view latent layout.
Confirm camera order, token count, 14D
physical units and at least 12 reference actions. Do not set the channel count
to 512 merely because the transformer hidden width is 512.

Start with `training_stage: collect`. After qualifying a collection checkpoint,
set `training_stage: online` and its `demo_checkpoint`; retain the same frozen
feature identity. Actor selection becomes available on the next episode after
BC warmup completes. To resume, use the saved online directory as
`actor_checkpoint` with its original learner configuration.

This template needs real assets and qualified demonstrations. CPU contract
checks with synthetic observations do not establish real-model latency,
robot execution quality, or learning performance.
