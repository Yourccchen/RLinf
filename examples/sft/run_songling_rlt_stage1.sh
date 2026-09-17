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
    # This script lives at <repo>/examples/sft/, so climb two levels.
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$PROJECT_ROOT"

export REPO_PATH="$PROJECT_ROOT"
# Hydra searchpath for the shared model/ and hybrid_engines/ config groups.
export EMBODIED_PATH="$PROJECT_ROOT/examples/sft"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

VENV_DIR="$PROJECT_ROOT/.venv"
PYTHON="$VENV_DIR/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    # sslaunch copies the project under /mnt/vepfs/base2/<user>. The venv's
    # absolute /home/<user> Python symlink then breaks, although its
    # site-packages and the corresponding uv-managed Python remain available
    # through the mapped home. Resolve that runtime in the same way as
    # openpi_Ario's Songling launcher.
    PYVENV_CFG="$VENV_DIR/pyvenv.cfg"
    if [[ ! -r "$PYVENV_CFG" ]]; then
        echo "ERROR: missing venv configuration at $PYVENV_CFG" >&2
        exit 1
    fi

    PYTHON_VERSION="$(awk -F= '/^version_info =/ {gsub(/ /, "", $2); print $2}' "$PYVENV_CFG")"
    PYTHON_MAJOR_MINOR="$(awk -F. '{print $1 "." $2}' <<< "$PYTHON_VERSION")"
    VENV_PYTHON_HOME="$(awk -F= '/^home =/ {sub(/^[[:space:]]*/, "", $2); print $2}' "$PYVENV_CFG")"
    UV_PYTHON_DIST="$(basename "$(dirname "$VENV_PYTHON_HOME")")"
    SHARED_HOME="$(dirname "$PROJECT_ROOT")"
    PROJECT_PYTHON="$VENV_DIR/python-runtime/bin/python${PYTHON_VERSION}"
    SHARED_PYTHON="$SHARED_HOME/.local/share/uv/python/$UV_PYTHON_DIST/bin/python${PYTHON_MAJOR_MINOR}"

    if [[ -x "$PROJECT_PYTHON" ]]; then
        PYTHON="$PROJECT_PYTHON"
    else
        PYTHON="$SHARED_PYTHON"
    fi
    if [[ ! -x "$PYTHON" ]]; then
        echo "ERROR: missing sslaunch Python runtime at $PYTHON" >&2
        echo "Build the local .venv before submitting the job." >&2
        exit 1
    fi

    export PYTHONPATH="$VENV_DIR/lib/python${PYTHON_MAJOR_MINOR}/site-packages${PYTHONPATH:+:$PYTHONPATH}"
fi

# The ARIO extension is maintained in the sibling openpi_Ario checkout. Its
# local /home path is mapped to the parent of PROJECT_ROOT under sslaunch.
OPENPI_ARIO_ROOT="${OPENPI_ARIO_ROOT:-$(dirname "$PROJECT_ROOT")/openpi_Ario}"
if [[ ! -d "$OPENPI_ARIO_ROOT/src/openpi" && -d "/home/fanyiming/openpi_Ario/src/openpi" ]]; then
    OPENPI_ARIO_ROOT="/home/fanyiming/openpi_Ario"
fi
if [[ ! -d "$OPENPI_ARIO_ROOT/src/openpi" ]]; then
    echo "ERROR: missing openpi_Ario source at $OPENPI_ARIO_ROOT" >&2
    exit 1
fi
export PYTHONPATH="$OPENPI_ARIO_ROOT/src:$OPENPI_ARIO_ROOT/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"

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
# shellcheck source=oss_ckpt_sidecar.sh
source "$PROJECT_ROOT/examples/sft/oss_ckpt_sidecar.sh"
start_oss_ckpt_sidecar
trap stop_oss_ckpt_sidecar EXIT

export HYDRA_FULL_ERROR=1
set +e
"$PYTHON" "$PROJECT_ROOT/examples/sft/train_vla_sft.py" \
    --config-path "$PROJECT_ROOT/examples/sft/config" \
    --config-name "$CONFIG_NAME" 2>&1 | tee /tmp/rlt_train.log
TRAIN_RC=${PIPESTATUS[0]}
set -e

trap - EXIT
drain_oss_ckpt_sidecar

if [[ "$TRAIN_RC" -ne 0 ]] || grep -q "Error executing job" /tmp/rlt_train.log; then
    echo "=== Stage-1 RLT SFT FAILED (rc=$TRAIN_RC) ===" >&2
    exit 1
fi

echo "=== Stage-1 RLT SFT complete ==="
