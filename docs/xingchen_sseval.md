# Xingchen SsEval runtime

Set `embodiment: xingchen` for EEF25 observations, actions and feedback.
Omitting this field retains Songling QPOS14 behavior. RWI owns hardware and
geometry adaptation; RLinf owns candidate inference, replay and learner updates.

Use `examples/embodiment/config/xingchen_rlt_sseval_td3.yaml` for OpenPI.
Xingchen and Songling both use `rlt_td3_mlp_policy` for RLT Stage2.
Their observation and action layouts differ; HALO-WA uses its own Stage2 model.
Supply its Stage1 paths and physical 25D bounds through `XINGCHEN_*` variables.
RWI injects `XingchenFeatureAdapter` through
`SsEvalRLTRuntime.from_config(feature_model_adapter=...)` to convert the trained
31D rot6d interface into EEF25. The target branch TD3 model/objectives remain
in use; this does not import the source branch paper actor.

OpenPI `transition_stride` must divide the actor horizon. RWI supplies matching
intermediate observations. Windows span contiguous recorded feedback; missing
snapshots exclude windows without their required features. Delayed outcome
labels update overlapping windows. Success/failure commits the episode, while
abort drops it.

Use `examples/embodiment/config/xingchen_soma_halo_wa_sseval.yaml` with RWI's
`SomaVAEFeatureModel(embodiment="xingchen")` for HALO-WA. It validates visual
layout, action bounds, horizon and feature identity. Keep whole-chunk replay
for this stateful provider. Human actions remain BC targets without overwriting
the frozen WA reference. Collection, demonstrations, warmup and resume follow
[HALO-WA Soma](halowa_soma.md).

These templates require actual trained assets, bounds and VAE dimensions.
Wire order is torso pose, head, left pose/gripper, right pose/gripper; poses
use xyz metres and xyzw. Do not restore Songling or 31D OpenPI learner data
as EEF25 replay. RWI's `docs/xingchen_rlt.md` documents paired launch commands.
