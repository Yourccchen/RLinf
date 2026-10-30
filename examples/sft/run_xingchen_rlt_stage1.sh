#!/bin/bash
# Submit one full B200 node:
#   sslaunch submit -c para -q b200 -n 1 -j xingchen-plug --no-log \
#     -- examples/sft/run_xingchen_rlt_stage1.sh
# Credentials are read at runtime from the existing environment file, never
# embedded in this script, CLI arguments, or the training configuration.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

python3 - "$PROJECT_ROOT" <<'PY'
import os
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1])
environment_file = Path(os.environ.get(
    "RLINF_ENV_CONFIG",
    str(root.parent / "ss-embodied/envs/bingxing_with_ali_datalake.yaml"),
))
with environment_file.open() as handle:
    config = yaml.safe_load(handle)

env = os.environ.copy()
read = config["object_store"]["read"]
write = config["object_store"]["write"]
env["AWS_ACCESS_KEY_ID"] = read["access_key_id"]
env["AWS_SECRET_ACCESS_KEY"] = read["secret_access_key"]
env.pop("AWS_SESSION_TOKEN", None)
env["ALI__AWS_ACCESS_KEY_ID"] = write["access_key_id"]
env["ALI__AWS_SECRET_ACCESS_KEY"] = write["secret_access_key"]
endpoint = write["endpoint"]
env["ALI__OSS_ENDPOINT"] = endpoint if "://" in endpoint else "https://" + endpoint
env["ALI__AWS_S3_ADDRESSING_STYLE"] = config["object_store"]["addressing_style"]
env["OSS_RUN_ROOT"] = (
    write["checkpoint_bucket_root"].replace("s3+write://", "s3://").rstrip("/")
    + "/fanyiming/rlinf_runs"
)
for key, name in (
    ("host", "SSEMB_PG_HOST"), ("port", "SSEMB_PG_PORT"),
    ("database", "SSEMB_PG_DATABASE"), ("user", "SSEMB_PG_USER"),
    ("password", "SSEMB_PG_PASSWORD"),
):
    env[name] = str(config["datalake"][key])
for name in (
    "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
    "http_proxy", "https_proxy", "all_proxy",
    "ALIBABA_ACCESS_KEY_ID", "ALIBABA_ACCESS_KEY_SECRET",
):
    env.pop(name, None)
env["CONFIG_NAME"] = "xingchen_plug_xc22_rlt_stage1_sft_openpi_pi05"
os.execvpe("bash", ["bash", str(root / "examples/sft/run_songling_rlt_stage1.sh")], env)
PY
