# Shared OSS checkpoint sidecar for Songling SFT launch scripts.
#
# Source this file after PROJECT_ROOT, PYTHON, and CONFIG_NAME are set:
#   source "$PROJECT_ROOT/examples/sft/oss_ckpt_sidecar.sh"
#   start_oss_ckpt_sidecar
#   ... train ...
#   drain_oss_ckpt_sidecar
#
# The watcher uploads each committed checkpoints/global_step_* tree (DCP
# included) to $OSS_RUN_DIR and leaves the local copy in place.
#
# Disable with OSS_CKPT_UPLOAD=0. Override the destination with OSS_RUN_ROOT
# or OSS_RUN_DIR. The default OSS prefix is
# s3://shengshu-base2-test/<user>/rlinf_runs, using the owner directory from
# runner.logger.log_path.

OSS_CKPT_UPLOAD="${OSS_CKPT_UPLOAD:-1}"
OSS_CKPT_UPLOADER_PID=""

_oss_ckpt_read_local_run_dir() {
    local config_file="$PROJECT_ROOT/examples/sft/config/${CONFIG_NAME}.yaml"
    "$PYTHON" - "$config_file" <<'PY'
from omegaconf import OmegaConf
import os
import sys

cfg = OmegaConf.load(sys.argv[1])
log_path = str(cfg.runner.logger.log_path).rstrip("/")
experiment_name = str(cfg.runner.logger.experiment_name)
owner = os.path.basename(os.path.dirname(log_path)) or "rlinf"
print(f"{log_path}/{experiment_name}")
print(experiment_name)
print(f"s3://shengshu-base2-test/{owner}/rlinf_runs")
PY
}

start_oss_ckpt_sidecar() {
    if [[ "${OSS_CKPT_UPLOAD}" != "1" ]]; then
        echo "=== OSS checkpoint upload disabled (OSS_CKPT_UPLOAD=${OSS_CKPT_UPLOAD}) ==="
        return 0
    fi

    export ALI__AWS_ACCESS_KEY_ID="${ALI__AWS_ACCESS_KEY_ID:-${AWS_ACCESS_KEY_ID:-}}"
    export ALI__AWS_SECRET_ACCESS_KEY="${ALI__AWS_SECRET_ACCESS_KEY:-${AWS_SECRET_ACCESS_KEY:-}}"
    export ALI__OSS_ENDPOINT="${ALI__OSS_ENDPOINT:-https://oss-cn-wulanchabu-internal.aliyuncs.com}"
    export ALI__AWS_S3_ADDRESSING_STYLE="${ALI__AWS_S3_ADDRESSING_STYLE:-virtual}"

    : "${ALI__AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID (or ALI__AWS_ACCESS_KEY_ID) for OSS checkpoint upload}"
    : "${ALI__AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY (or ALI__AWS_SECRET_ACCESS_KEY) for OSS checkpoint upload}"

    local parsed_dir experiment_name suggested_oss_root
    {
        read -r parsed_dir
        read -r experiment_name
        read -r suggested_oss_root
    } < <(_oss_ckpt_read_local_run_dir)
    LOCAL_RUN_DIR="${LOCAL_RUN_DIR:-$parsed_dir}"
    OSS_RUN_ROOT="${OSS_RUN_ROOT:-$suggested_oss_root}"
    OSS_RUN_DIR="${OSS_RUN_DIR:-${OSS_RUN_ROOT%/}/${experiment_name}}"

    echo "=== local checkpoints: $LOCAL_RUN_DIR ==="
    echo "=== OSS checkpoints:   $OSS_RUN_DIR ==="

    "$PYTHON" "$PROJECT_ROOT/toolkits/oss_ckpt_uploader.py" \
        --local-run-dir "$LOCAL_RUN_DIR" \
        --oss-run-dir "$OSS_RUN_DIR" \
        --poll-secs 60 \
        --settle-secs 10 \
        --max-idle-polls 0 &
    OSS_CKPT_UPLOADER_PID=$!
}

stop_oss_ckpt_sidecar() {
    if [[ -n "${OSS_CKPT_UPLOADER_PID}" ]]; then
        kill "$OSS_CKPT_UPLOADER_PID" 2>/dev/null || true
        wait "$OSS_CKPT_UPLOADER_PID" 2>/dev/null || true
        OSS_CKPT_UPLOADER_PID=""
    fi
}

drain_oss_ckpt_sidecar() {
    stop_oss_ckpt_sidecar
    if [[ "${OSS_CKPT_UPLOAD}" != "1" ]]; then
        return 0
    fi
    echo "=== uploading remaining checkpoints ==="
    "$PYTHON" "$PROJECT_ROOT/toolkits/oss_ckpt_uploader.py" \
        --local-run-dir "$LOCAL_RUN_DIR" \
        --oss-run-dir "$OSS_RUN_DIR" \
        --poll-secs 30 \
        --settle-secs 10 \
        --max-idle-polls 3
    echo "=== checkpoint upload complete ==="
}
