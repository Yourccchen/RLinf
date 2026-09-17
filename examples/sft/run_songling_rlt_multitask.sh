#!/bin/bash
# Stage-1 RLT SFT over the full Songling ARIO corpus (547 tasks, ~712k
# episodes), on two nodes.
#
# Unlike the single-task script, this one has to build the Ray cluster itself:
# RLinf attaches to whatever Ray is already running, so every node must join
# before the entry script starts. PyTorchJob gives each pod a RANK, which this
# script turns into RLINF_NODE_RANK -- it has to be exported before `ray start`,
# because Ray captures the environment at that moment and RLinf reads the rank
# back off the node when it sorts the cluster.
#
# Submit with sslaunch (2 nodes x 8 GPUs):
#   cd /home/caimengchen/codebases/RLinf
#   sslaunch submit -j songling-rlt-all -n 2 --no-log \
#       -- examples/sft/run_songling_rlt_multitask.sh
#
# Credentials are never stored here: the cluster injects them, or you export
# AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY before launching.
#
# Checkpoints stay on VEPfs and are also uploaded (DCP included) to
# $OSS_RUN_ROOT/<experiment_name>/checkpoints/. Disable with OSS_CKPT_UPLOAD=0.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
if [[ -n "${REPO_PATH:-}" ]]; then
    PROJECT_ROOT="$REPO_PATH"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$PROJECT_ROOT"

export REPO_PATH="$PROJECT_ROOT"
export EMBODIED_PATH="$PROJECT_ROOT/examples/sft"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="$PROJECT_ROOT/.venv/bin/python"
RAY="$PROJECT_ROOT/.venv/bin/ray"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: missing venv interpreter at $PYTHON" >&2
    exit 1
fi

CONFIG_NAME="${CONFIG_NAME:-songling_rlt_stage1_multitask}"

# ---------------------------------------------------------------------------
# Cluster identity
#
# MASTER_ADDR is the master pod's hostname, so it is unique per job: using it
# as the rendezvous key keeps a stale file from an earlier job from pointing
# this one at a dead head.
# ---------------------------------------------------------------------------
NODE_RANK="${RANK:-0}"
NUM_NODES="${WORLD_SIZE:-1}"
# Ray needs a port of its own, and with hostNetwork the port is taken on the
# host: derive it from the torch port k8s already allocated for this job so two
# concurrent jobs on one node cannot collide.
RAY_PORT="${RAY_PORT:-$(( ${MASTER_PORT:-28500} + 1 ))}"

RENDEZVOUS_DIR="/mnt/vepfs/world/caimengchen/rlinf_rendezvous/${MASTER_ADDR:-local}"
HEAD_IP_FILE="$RENDEZVOUS_DIR/head_ip"
DONE_FILE="$RENDEZVOUS_DIR/done"

export RLINF_NODE_RANK="$NODE_RANK"

echo "=== node rank $NODE_RANK / $NUM_NODES, ray port $RAY_PORT ==="
echo "=== rendezvous: $RENDEZVOUS_DIR ==="

# ---------------------------------------------------------------------------
# Credentials: accept cluster-injected Alibaba names or AWS-compatible names,
# falling back to the [default] profile in ~/.aws/credentials.
# ---------------------------------------------------------------------------
if [[ -n "${ALIBABA_ACCESS_KEY_ID:-}" ]]; then
    export AWS_ACCESS_KEY_ID="$ALIBABA_ACCESS_KEY_ID"
fi
if [[ -n "${ALIBABA_ACCESS_KEY_SECRET:-}" ]]; then
    export AWS_SECRET_ACCESS_KEY="$ALIBABA_ACCESS_KEY_SECRET"
fi
if [[ -z "${AWS_ACCESS_KEY_ID:-}" && -r "$HOME/.aws/credentials" ]]; then
    # ArioStreamingDataset reads os.environ directly and ignores the boto3
    # profile chain, so lift the profile into the environment here.
    eval "$(
        awk -F' *= *' '
            /^\[/ { in_default = ($0 == "[default]") ; next }
            in_default && $1 == "aws_access_key_id"     { printf "export AWS_ACCESS_KEY_ID=%s\n", $2 }
            in_default && $1 == "aws_secret_access_key" { printf "export AWS_SECRET_ACCESS_KEY=%s\n", $2 }
        ' "$HOME/.aws/credentials"
    )"
fi
: "${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID (or ALIBABA_ACCESS_KEY_ID) for OSS reads}"
: "${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY (or ALIBABA_ACCESS_KEY_SECRET) for OSS reads}"

# ---------------------------------------------------------------------------
# Fail fast on CPU: pi0.5 Stage-1 SFT is not viable without a GPU.
# ---------------------------------------------------------------------------
echo "=== checking accelerator ==="
"$PYTHON" - <<'PY'
import sys
import torch

if not torch.cuda.is_available():
    sys.exit("ERROR: CUDA unavailable; refusing to train pi0.5 on CPU.")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU{i}: {p.name}  {p.total_memory / 1024**3:.1f}GB")
x = torch.ones((512, 512), dtype=torch.bfloat16, device="cuda")
torch.matmul(x, x).cpu()
print("bf16 matmul smoke test: PASSED")
PY

NODE_IP="$(hostname -I | awk '{print $1}')"

if [[ "$NODE_RANK" -eq 0 ]]; then
    # -----------------------------------------------------------------------
    # Head: start Ray, publish the address, then train.
    # -----------------------------------------------------------------------
    mkdir -p "$RENDEZVOUS_DIR"
    rm -f "$HEAD_IP_FILE" "$DONE_FILE"

    echo "=== starting ray head on $NODE_IP:$RAY_PORT ==="
    "$RAY" start --head --node-ip-address="$NODE_IP" --port="$RAY_PORT" --disable-usage-stats

    echo "$NODE_IP" > "$HEAD_IP_FILE.tmp" && mv "$HEAD_IP_FILE.tmp" "$HEAD_IP_FILE"
    echo "=== published head address ==="

    # Wait for every other node to register before handing the cluster to
    # RLinf: Cluster asserts that the node ranks it finds are 0..N-1, so a
    # worker that is still joining would be read as a smaller cluster.
    echo "=== waiting for $NUM_NODES nodes to join ray ==="
    for _ in $(seq 1 120); do
        joined="$("$PYTHON" -c "
import ray
ray.init(address='auto', logging_level='ERROR')
print(sum(1 for n in ray.nodes() if n['Alive']))
" 2>/dev/null || echo 0)"
        echo "  ray sees $joined/$NUM_NODES alive nodes"
        [[ "$joined" -ge "$NUM_NODES" ]] && break
        sleep 10
    done
    if [[ "${joined:-0}" -lt "$NUM_NODES" ]]; then
        echo "ERROR: only $joined/$NUM_NODES nodes joined the ray cluster" >&2
        touch "$DONE_FILE"
        exit 1
    fi

    echo "=== config:  $CONFIG_NAME ==="
    echo "=== repo:    $PROJECT_ROOT ==="

    # shellcheck source=oss_ckpt_sidecar.sh
    source "$PROJECT_ROOT/examples/sft/oss_ckpt_sidecar.sh"
    start_oss_ckpt_sidecar
    trap stop_oss_ckpt_sidecar EXIT

    export HYDRA_FULL_ERROR=1
    set +e
    "$PYTHON" "$PROJECT_ROOT/examples/sft/train_vla_sft.py" \
        --config-path "$PROJECT_ROOT/examples/sft/config" \
        --config-name "$CONFIG_NAME" 2>&1 | tee /tmp/rlt_multitask_train.log
    TRAIN_RC=${PIPESTATUS[0]}
    set -e

    # Release the worker pod whatever happened, so it does not outlive the run.
    touch "$DONE_FILE"

    trap - EXIT
    drain_oss_ckpt_sidecar

    if [[ "$TRAIN_RC" -ne 0 ]] || grep -q "Error executing job" /tmp/rlt_multitask_train.log; then
        echo "=== multitask Stage-1 RLT SFT FAILED (rc=$TRAIN_RC) ===" >&2
        exit 1
    fi
    echo "=== multitask Stage-1 RLT SFT complete ==="
else
    # -----------------------------------------------------------------------
    # Worker: join the head, then idle until the head reports it is finished.
    # -----------------------------------------------------------------------
    echo "=== waiting for head address ==="
    HEAD_IP=""
    for _ in $(seq 1 180); do
        if [[ -s "$HEAD_IP_FILE" ]]; then
            HEAD_IP="$(cat "$HEAD_IP_FILE")"
            [[ -n "$HEAD_IP" ]] && break
        fi
        sleep 5
    done
    if [[ -z "$HEAD_IP" ]]; then
        echo "ERROR: head never published its address to $HEAD_IP_FILE" >&2
        exit 1
    fi

    echo "=== joining ray head at $HEAD_IP:$RAY_PORT from $NODE_IP ==="
    "$RAY" start --address="$HEAD_IP:$RAY_PORT" --node-ip-address="$NODE_IP" --disable-usage-stats

    echo "=== joined; idling until the head finishes ==="
    while [[ ! -f "$DONE_FILE" ]]; do
        sleep 30
    done
    echo "=== head finished; shutting down ray worker ==="
    "$RAY" stop || true
fi
