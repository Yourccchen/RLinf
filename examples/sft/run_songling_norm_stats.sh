#!/bin/bash
# Compute normalization statistics over the full Songling ARIO corpus.
#
# The multi-task Stage-1 run needs songling/all_tasks/norm_stats.json to
# normalize state and actions; only the single-task (garment_folding) stats
# exist on disk. This samples 200k frames of state/actions with skip_video=True,
# so it downloads .pt files only and never decodes video.
#
# Submitted as a cluster job rather than run on the devbox: the pass takes 1-2
# hours, and a devbox shell does not outlive the session that started it.
#
#   cd /home/caimengchen/codebases/RLinf
#   sslaunch submit -j songling-normstats --gpu 1 --no-log \
#       -- examples/sft/run_songling_norm_stats.sh
#
# On success the stats are copied to the path the RLinf loader reads, which is
# assets_dir in the pi05_rlt_songling_all dataconfig:
#   $MODEL_PATH/songling/all_tasks/norm_stats.json

set -euo pipefail

OPENPI_ROOT="${OPENPI_ROOT:-/home/caimengchen/codebases/openpi_Ario}"
RLINF_ROOT="${RLINF_ROOT:-/home/caimengchen/codebases/RLinf}"
PYTHON="$RLINF_ROOT/.venv/bin/python"
CONFIG_NAME="${NORM_CONFIG_NAME:-pi05_songling_all_norm}"
MAX_FRAMES="${MAX_FRAMES:-200000}"

# Where the training run looks the stats up.
MODEL_PATH="${MODEL_PATH:-/mnt/vepfs/world/caimengchen/models/pi05_base}"
DEST_DIR="$MODEL_PATH/songling/all_tasks"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: missing venv interpreter at $PYTHON" >&2
    exit 1
fi

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

cd "$OPENPI_ROOT"

echo "=== config:     $CONFIG_NAME ==="
echo "=== max frames: $MAX_FRAMES ==="
echo "=== dest:       $DEST_DIR ==="

"$PYTHON" -u scripts/compute_norm_stats.py \
    --config-name "$CONFIG_NAME" \
    --max-frames "$MAX_FRAMES"

# compute_norm_stats.py writes to assets_base_dir/<config name>/<repo_id>.
SRC="$OPENPI_ROOT/assets/$CONFIG_NAME/songling/all_tasks/norm_stats.json"
if [[ ! -s "$SRC" ]]; then
    echo "ERROR: expected stats at $SRC, but it is missing or empty" >&2
    exit 1
fi

mkdir -p "$DEST_DIR"
cp "$SRC" "$DEST_DIR/norm_stats.json.tmp"
mv "$DEST_DIR/norm_stats.json.tmp" "$DEST_DIR/norm_stats.json"
echo "=== wrote $DEST_DIR/norm_stats.json ==="

# Fail loudly if the stats do not describe the 14-D Songling vector the run
# trains on: a silent dimension mismatch would only surface as bad actions.
"$PYTHON" - "$DEST_DIR/norm_stats.json" <<'PY'
import json
import sys

path = sys.argv[1]
stats = json.load(open(path))["norm_stats"]
for key in ("state", "actions"):
    for field in ("mean", "std", "q01", "q99"):
        n = len(stats[key][field])
        if n != 14:
            sys.exit(f"ERROR: {key}.{field} has {n} dims, expected 14")
print("norm stats validated: state and actions are 14-D")
PY

echo "=== norm stats complete ==="
