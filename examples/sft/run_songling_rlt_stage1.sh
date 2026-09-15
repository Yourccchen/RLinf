#!/bin/bash
# Stage-1 RLT SFT for Songling dual-arm ARIO data.
#
# Trains pi0.5 together with the RLT token encoder-decoder, reading ARIO
# episodes straight from OSS (no LeRobot conversion). The joint objective is
#   loss = rlt_loss + rlt_alpha * vla_loss
# so the run validates both the SFT fit and the RLT reconstruction quality.
#
# Submit with sslaunch (one full 8-GPU node; the embody pool has no partially
# free nodes, so a smaller request cannot be admitted):
#   cd /home/caimengchen/codebases/RLinf
#   sslaunch submit -j songling-rlt-s1 -n 1 --tail-log \
#       -- examples/sft/run_songling_rlt_stage1.sh
#
# Credentials are never stored here: the cluster injects them, or you export
# AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY before launching.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
if [[ -n "${REPO_PATH:-}" ]]; then
    PROJECT_ROOT="$REPO_PATH"
else
    # This script lives at <repo>/examples/sft/, so climb two levels.
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$PROJECT_ROOT"

export REPO_PATH="$PROJECT_ROOT"
# Hydra searchpath for the shared model/ and hybrid_engines/ config groups.
export EMBODIED_PATH="$PROJECT_ROOT/examples/sft"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: missing venv interpreter at $PYTHON" >&2
    echo "Build it with: bash requirements/install.sh embodied --model openpi --env maniskill_libero" >&2
    exit 1
fi

CONFIG_NAME="${CONFIG_NAME:-songling_rlt_stage1_sft_openpi_pi05}"

# ---------------------------------------------------------------------------
# Credentials: accept cluster-injected Alibaba names or AWS-compatible names,
# falling back to the [default] profile in ~/.aws/credentials. Never pass these
# on the sslaunch command line -- they would end up in the job spec.
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
    sys.exit(
        "ERROR: CUDA unavailable; refusing to train pi0.5 on CPU. "
        "Submit this script to a GPU queue via sslaunch."
    )
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  GPU{i}: {p.name}  {p.total_memory / 1024**3:.1f}GB")

# Compile a small bf16 GEMM so an unsupported toolchain fails now, not after
# the 14GB checkpoint has loaded.
x = torch.ones((512, 512), dtype=torch.bfloat16, device="cuda")
torch.matmul(x, x).cpu()
print("bf16 matmul smoke test: PASSED")
PY

echo "=== config:     $CONFIG_NAME ==="
echo "=== repo:       $PROJECT_ROOT ==="
echo "=== python:     $PYTHON ==="

# ---------------------------------------------------------------------------
# Train
#
# Hydra swallows job exceptions and can still exit 0, which would report a
# crashed run as Succeeded. Force a non-zero exit unless the run really
# finished, so the job state reflects reality.
# ---------------------------------------------------------------------------
export HYDRA_FULL_ERROR=1
set +e
"$PYTHON" "$PROJECT_ROOT/examples/sft/train_vla_sft.py" \
    --config-path "$PROJECT_ROOT/examples/sft/config" \
    --config-name "$CONFIG_NAME" 2>&1 | tee /tmp/rlt_train.log
TRAIN_RC=${PIPESTATUS[0]}
set -e

if [[ "$TRAIN_RC" -ne 0 ]] || grep -q "Error executing job" /tmp/rlt_train.log; then
    echo "=== Stage-1 RLT SFT FAILED (rc=$TRAIN_RC) ===" >&2
    exit 1
fi

echo "=== Stage-1 RLT SFT complete ==="
