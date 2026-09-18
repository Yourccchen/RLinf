# SsEvalPlatform in-process RLT runtime

`SsEvalRLTRuntime` is called directly by RealWorldInference
`policy/policies/rlinf/policy.py`. No HTTP or extra WebSocket is used between
RWI PolicyServer and RLinf. The current development host has no GPU; real model
loading is intentionally deferred to the Policy GPU host.

## Required deployment environment

- `SONGLING_ACTION_LOW` and `SONGLING_ACTION_HIGH` as JSON arrays of 14 values
- optional `SONGLING_RLT_DEVICE`, default `cuda`

The selected Policy interpreter must contain both RLinf and RWI PolicyServer
dependencies (`torch`, OpenPI, OmegaConf, PyYAML, websockets and OpenCV). The
GPU host must also provide OpenCV's system libraries, or use a compatible
headless OpenCV build. Do not mix site-packages from different Python minor
versions.

`examples/embodiment/config/songling_rlt_sseval_td3.yaml` takes a
`stage1_sft_config` path. At load time the runtime copies `actor.model` from
that Stage1 SFT YAML into `feature_model` (shape, OpenPI transforms, RLT
encoder, default `openpi_data`) and forces `openpi.task: eval`. The deploy YAML
still owns `feature_model.model_path`, `precision`,
`openpi_data.norm_stats_path`, the TD3 actor, and the learner. Eval hosts do
not share the Stage1 SFT assets path, so the Stage1 checkpoint and
`norm_stats.json` paths are written in this file instead of inherited from
training. Point `stage1_sft_config` at the YAML used for the Stage1 run so
train and serve stay aligned without duplicating the frozen Stage1 fields.

To change the execute chunk length later, change Stage1
``actor.model.num_action_chunks`` (and retrain that checkpoint). Align copies
it onto the serving VLA and actor; Policy hello reports it as
``policy_spec.chunk_len``. SEP must validate dual-action payloads against that
advertised length instead of a hardcoded constant, then restart Policy and
SEP. Slave uses the received action shape.

The following command starts RWI PolicyServer with the in-process RLinf adapter;
it loads Stage1 and starts the background TD3 learner, so run it only on the GPU
Policy host:

```bash
cd /home/fanyiming/RealWorldInference-fym
POLICY_PYTHON=/home/fanyiming/RLinf/.venv/bin/python ./run_policy.sh \
  rlinf 0 policy/config_rlinf_songling_rlt.yaml
```

The following command runs only CPU/Fake RLinf contract, Runtime and learner
tests; it does not load a Stage1 checkpoint:

```bash
cd /home/fanyiming/RLinf
.venv/bin/python -m pytest -q \
  tests/unit_tests/test_sseval_rlt_contract.py \
  tests/unit_tests/test_sseval_rlt_runtime.py
```

The following command runs the RWI adapter and cross-repository Fake Runtime
tests without starting PolicyServer:

```bash
cd /home/fanyiming/RealWorldInference-fym
python -m pytest -q tests/test_rlinf_policy.py
```
