#!/bin/bash
# Extract RLT tokens (z_rl) from a Stage-1 RLT SFT checkpoint, on a GPU.
#
# The extraction runs a full pi0.5 prefix forward (3.35B params) per batch,
# which is why it belongs on a GPU node rather than the devbox.
#
#   cd /home/caimengchen/codebases/RLinf
#   sslaunch submit -j rlt-tokens -n 1 --no-log \
#       -- examples/sft/run_extract_rlt_tokens.sh
#
# Override the defaults with env vars, e.g. to use the final checkpoint:
#   CKPT_STEP=20000 sslaunch submit ...

set -euo pipefail

PROJECT_ROOT="${REPO_PATH:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$PROJECT_ROOT"

export REPO_PATH="$PROJECT_ROOT"
export EMBODIED_PATH="$PROJECT_ROOT/examples/sft"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="$PROJECT_ROOT/.venv/bin/python"

RUN_DIR="${RUN_DIR:-/mnt/vepfs/world/caimengchen/rlinf_runs/songling_rlt_stage1_sft_openpi_pi05}"
CKPT_STEP="${CKPT_STEP:-17500}"
CHECKPOINT="${CHECKPOINT:-$RUN_DIR/checkpoints/global_step_$CKPT_STEP}"
CONFIG_NAME="${CONFIG_NAME:-songling_rlt_stage1_sft_openpi_pi05}"
NUM_SAMPLES="${NUM_SAMPLES:-64}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_EPISODES="${MAX_EPISODES:-64}"
OUTPUT="${OUTPUT:-/mnt/vepfs/world/caimengchen/artifacts/rlt_tokens/rlt_tokens_step${CKPT_STEP}.pt}"

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

if [[ ! -f "$CHECKPOINT/actor/model_state_dict/full_weights.pt" ]]; then
    echo "ERROR: no full_weights.pt under $CHECKPOINT/actor/model_state_dict/" >&2
    echo "Available checkpoints:" >&2
    ls "$RUN_DIR/checkpoints/" >&2 || true
    exit 1
fi

echo "=== checkpoint: $CHECKPOINT ==="
echo "=== samples:    $NUM_SAMPLES (batch $BATCH_SIZE) ==="
echo "=== output:     $OUTPUT ==="

# Pin to one GPU: the extraction is a single-process forward, and the job holds
# a whole node only because the queue cannot allocate a partial one.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$PYTHON" -u "$PROJECT_ROOT/toolkits/extract_rlt_tokens.py" \
    --checkpoint "$CHECKPOINT" \
    --config-name "$CONFIG_NAME" \
    --config-path "$PROJECT_ROOT/examples/sft/config" \
    --num-samples "$NUM_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --max-episodes "$MAX_EPISODES" \
    --output "$OUTPUT"

echo "=== RLT token extraction complete ==="
