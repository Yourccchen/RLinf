# RLT Stage 2 Progress

Branch: `feat/rlt-stage2`
Base: `feat/rlt` at `dcbec901`

## Completed

- Reused existing `feat/rlt-stage2` worktree without discarding pre-existing changes.
- Paper-aligned TD3 bootstrap now uses the current actor and target critics; actor objective uses Q1.
- Added actor/critic parameter-group gradient clipping through the FSDP global-norm API; the other optimizer group is temporarily masked without losing its gradients.
- Added Replay Buffer `max_num_samples` plumbing and trajectory-granular eviction.
- Separated real-world critical-phase recording from actor activation so VLA warmup transitions can be recorded.
- Added Songling TD3 shape/reference-horizon unit-test scaffolding.
- Fixed an undefined `actions` binding in `RealworldRLTRoute.route`.
- Added strict Songling three-camera repacking: overhead + ordered left/right wrist views, with fail-fast shape checks.
- Added pre-load Stage1/Stage2 validation for action dimension, reference horizon, RLT embedding dimension, `use_rlt`, and frozen eval mode.
- Added unit-test coverage for Songling camera and cross-stage configuration contracts.
- Added `RemoteSonglingEnv`, a persistent MessagePack/WebSocket RPC client, and `SonglingActionCodec`.
- Added executed-action propagation, valid-step masking, and no-bootstrap behavior for short/uncertain chunks.
- Added filtered per-transition replay for real RLT rollouts so `record_transition` is enforced.
- Added `songling_rlt_stage2_td3_mlp.yaml` with C=10, H_ref=10, UTD=5, q-head-only target updates, RPC settings, and fail-fast environment inputs.
- Corrected the draft H_ref=50 assumption: the actual Songling Stage1 SFT configs/checkpoints use a 10-step horizon, so Stage2 now enforces H_ref=10 to avoid untrained positions and RL-token distribution shift.
- Documented the exact external RealWorldInference wire contract in `rlinf/envs/remote_songling/README.md`.
- Added focused Codec, MessagePack ndarray, request-ID, short-chunk, and transport-failure tests.
- Added policy-version propagation and startup capability validation for the remote environment.
- Follow-up review: Songling Stage2 proprio now uses the same normalized `observation.state[:14]` representation as Stage1 instead of raw qpos scale.
- Strengthened Songling startup validation for `rlt_image_only=False`, masking, prefix length, RLT layer/head/encoder settings, config name, and norm-stat repo identity.
- Fixed Hydra resolution for JSON-array environment settings by keeping them as strings through composition and decoding them at the RemoteSongling boundary; direct YAML/OmegaConf sequences remain supported.
- Added `_self_` to the Stage2 defaults list to make composition order explicit and remove the Hydra warning.
- Added explicit seeded Stage1 reference sampling, per-transition seed/checkpoint hash metadata, and a token-only `extract_rlt_token_obs()` path for offline preprocessing.
- Exposed the resolved Stage1 checkpoint source on the feature wrapper for transition traceability.
- Added a labeled Songling episode -> normalized RLT replay converter with deterministic Stage1 inference and configurable offline stride (default 2).
- Added an offline-only TD3 config/entrypoint that preloads replay without launching Environment or Rollout workers.
- Restricted online patch synchronization to Actor parameters, while checkpointing target critic, optimizers, replay, schedule counters, and RNG state.
- Made online `transition_stride=10` explicit and fail-fast: this is the documented C-step degradation because intermediate observations do not traverse Stage1 online.
- Focused verification now passes 32 tests across Stage2, replay round-trip, one TD3 optimizer step, RemoteSonglingEnv, and a real local WebSocket Fake RPC server.
- Both online three-worker and offline-only Hydra configs resolve successfully with representative environment values.
- Real Stage1 checkpoint inference remains an external acceptance step because `SONGLING_RLT_STAGE1_CHECKPOINT` is not configured on the host; no synthetic result is reported as a substitute.
- Follow-up gap review added replay sampling-generator checkpoint/resume and candidate Actor staging with single-environment episode-boundary activation.
- Focused verification now passes 35 tests, including boundary activation/payload and replay RNG sequence round-trip.

## Verification

- `py_compile` passed for every modified/new Python module.
- `git diff --check` passed.
- Focused Stage2 suite in `.venv`: 23 passed (`test_remote_songling_env.py` + `test_rlt_stage2.py`).
- The actual `validate_rlt_stage2_configs` implementation passed accepted-H_ref=10 and rejected-H_ref=50 smoke checks.
- The actual FSDP parameter-group clipping helper passed an isolation/restoration smoke test.
- Songling Stage2 YAML parses and its C=10, H_ref=10, UTD=5, and q-head-only target invariants pass.
- Black check passed for the new/updated focused implementation files.
- Strict Songling semantic-config smoke test passed and rejected an `rlt_image_only` distribution mismatch.
- Full Hydra `--cfg job --resolve` now exits 0 with representative Songling environment values; neither the list-type exception nor missing-`_self_` warning remains.
- The complete focused suite now runs in `.venv`; the login-shell system Python remains irrelevant to project verification. Real OpenPI checkpoint inference is still blocked because `openpi` is not importable and CUDA is not exposed in the current SSH session.

## RLinf-only plan status (8 items)

1. **Complete ? TD3 core:** current Actor for target actions, target Critics only, Q1 actor objective, UTD 5, delayed actor, FSDP-safe clipping, bounded Replay.
2. **Partial ? Stage1 feature API:** three cameras, normalized proprio, shapes and semantic config checks are done; explicit reference RNG/version tracing, token-only extraction, and real checkpoint inference remain.
3. **Complete ? Songling Stage2 config:** TD3/RPC config and Hydra resolution are verified.
4. **Complete ? Remote Environment Worker:** RPC client, codec, env registration, executed actions, masks, policy version and capability checks are implemented.
5. **Partial ? Rollout/chunk transitions:** warmup recording, actual-action replay and short-chunk handling are done; configurable `transition_stride` and overlapping intermediate-feature transitions remain.
6. **Not started ? Offline Stage2 runner:** episode conversion, reward-label validation, replay preload and offline-only entrypoint remain.
7. **Partial ? Online three-worker runner:** existing RLinf runner/config/schedule/weight sync are wired; Actor-only sync, candidate activation at episode boundaries and fake end-to-end execution remain.
8. **Partial ? Verification:** 23 focused tests and static/config checks pass; real checkpoint, fake three-worker, shadow-mode and hardware staged acceptance remain.

## In Progress

- Full integration execution in a prepared RLinf runtime.

## Remaining external prerequisites

- Use `/home/fanyiming/RLinf/.venv/bin/python`; it contains Ray, OmegaConf, Gymnasium, WebSockets, MessagePack, PyTorch, and pytest.
- Real Stage1 feature/checkpoint inference still needs an importable `openpi` package and a GPU-visible job environment.
- RealWorldInference must implement the documented `songling-env-v1` health/reset/observe/chunk_step/close service and report exact joint order, units, frequency, camera keys, actual executed actions, rewards, terminal flags, intervention flags, and critical-phase flags.
- Real action limits, units, frequency, image shape, RPC endpoint, and the compatible Songling Stage1 checkpoint must be supplied through the YAML environment variables.

## Next

1. Run the complete focused suite in the prepared RLinf project environment.
2. Exercise one fake end-to-end EnvWorker -> rollout -> replay transition.
3. Run a health/reset/hold-only hardware smoke test before permitting action chunks.


