# Remote Songling environment contract

`RemoteSonglingEnv` is the RLinf client for an independently deployed
RealWorldInference service. RLinf does not import or modify that repository.

## Transport

- Persistent WebSocket connection.
- Every frame is binary MessagePack.
- Protocol version: `songling-env-v1`.
- Request envelope: `protocol_version`, unique `request_id`, `method`, `params`.
- Response envelope: matching `protocol_version` and `request_id`, `ok`, then
  either `result` or structured `error`.
- NumPy arrays use the recursive `{__ndarray__, dtype, shape, data}` encoding
  implemented in `client.py`.
- RLinf never retries `chunk_step`; a transport failure is an execution-uncertain
  truncation.

## Required methods

- `health`: reports `healthy`, protocol version, `action_type=absolute_qpos`,
  `action_dim=14`, exact `joint_order`, exact `action_units`,
  `action_frequency_hz`, `max_chunk_len`, and all three camera names.
- `reset`: accepts `session_id` and instruction; returns `episode_id` and the
  operator-gated initial observation.
- `observe`: returns an observation when `reset` does not embed one.
- `chunk_step`: accepts physical absolute qpos `[K,14]`; returns the valid prefix
  of post-action observations, actual executed actions, rewards, terminal flags,
  intervention metadata, and final observation when terminal.
- `close`: enters hold and closes the external session.

Each observation contains finite `float32[14]` state, `uint8[H,W,3]`
`head_camera`, `left_camera`, and `right_camera`, a non-empty instruction, plus
freshness and monotonic timestamp entries for every stream.

## Required launch environment

The Stage2 YAML intentionally fails fast unless these values are supplied:

- `SONGLING_ENV_RPC_ENDPOINT`
- `SONGLING_RLT_STAGE1_CHECKPOINT`
- `SONGLING_ACTION_LOW` and `SONGLING_ACTION_HIGH` as JSON lists of 14 numbers
- `SONGLING_ACTION_UNITS` as a JSON list of 14 unit strings
- `SONGLING_ACTION_FREQUENCY_HZ` as a number
- `SONGLING_IMAGE_SHAPE` as `[H,W,3]`
- optional `SONGLING_TASK_PROMPT`

## Offline preprocessing and training

The episode file must contain a non-empty list (or `{"episodes": [...]}`) with
T+1 observations and step-aligned physical `executed_actions`, `rewards`,
`terminated`, and `truncated` arrays. Unlabeled demonstrations are rejected
because TD3 cannot infer success/failure targets from actions alone.

The following command runs frozen Stage1 inference, normalizes reference and
executed actions, builds C=50 transitions at stride 2, and writes a replay
checkpoint consumable by RLinf:

```bash
python toolkits/build_songling_rlt_replay.py \
  --episodes /path/to/songling_episodes.pt \
  --stage2-config examples/embodiment/config/songling_rlt_stage2_td3_mlp.yaml \
  --stage1-checkpoint /path/to/stage1_checkpoint \
  --action-low '[...]' --action-high '[...]' \
  --output /path/to/songling_rlt_replay
```

The following command starts only the Actor/Critic learner and preloads that
replay checkpoint; it does not create Environment or Rollout workers:

```bash
export SONGLING_RLT_OFFLINE_REPLAY=/path/to/songling_rlt_replay
bash examples/embodiment/run_offline_rl.sh songling_rlt_stage2_td3_offline
```

## Online training

The following command starts the Environment, Rollout, and Actor workers after
the required launch environment above has been exported:

```bash
bash examples/embodiment/run_embodiment.sh songling_rlt_stage2_td3_mlp
```

Online Actor updates are first staged in a rollout-side candidate model. The
active Actor and its policy version change only on initial startup or after the
single Songling environment reports an episode boundary; critics are never
synchronized to rollout.

Online collection records exactly one transition per fully executed C=50 RPC
chunk (`transition_stride=50`), because intermediate observations do not pass
through the Stage1 feature worker. Stride-2 overlapping windows are implemented
in offline preprocessing. A non-C online stride fails at startup rather than
silently storing actions that were not actually executed.
