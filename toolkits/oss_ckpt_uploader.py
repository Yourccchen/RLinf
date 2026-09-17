#!/usr/bin/env python
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Upload committed RLinf checkpoints to Aliyun OSS.

This sidecar watches a local experiment directory, uploads each fully
committed ``checkpoints/global_step_*`` (and ``best_model``) tree through
the S3-compatible API, and verifies every remote file by size. Local
files are kept. Repeated runs are idempotent: files already present with
the expected size are skipped.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from rlinf.utils.checkpoint import (
    CHECKPOINT_COMMIT_MARKER,
    is_committed_checkpoint,
)

_PROXY_ENV_VARS = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "all_proxy",
    "ALL_PROXY",
)


def log(message: str) -> None:
    print(f"[uploader] {time.strftime('%H:%M:%S')} {message}", flush=True)


def parse_oss_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/prefix`` (or ``oss://`` / ``s3+ali://``) into parts."""
    for scheme in ("s3+ali://", "oss://", "s3://"):
        if uri.startswith(scheme):
            path = uri[len(scheme) :]
            break
    else:
        raise ValueError(f"Unsupported OSS URI: {uri}")
    bucket, separator, prefix = path.partition("/")
    if not bucket or not separator:
        raise ValueError(f"OSS URI must include bucket and prefix: {uri}")
    return bucket, prefix.rstrip("/")


def object_key(remote_prefix: str, relative_path: str) -> str:
    """Build an object key from the run prefix and a local relative path."""
    return f"{remote_prefix.rstrip('/')}/{relative_path.replace(os.sep, '/')}"


def committed_checkpoint_dirs(local_run_dir: str) -> list[str]:
    """Return checkpoint paths relative to ``local_run_dir`` that are committed.

    Looks under ``{local_run_dir}/checkpoints/`` for ``global_step_*`` and
    ``best_model`` directories that contain ``COMMITTED``.
    """
    checkpoints_root = os.path.join(local_run_dir, "checkpoints")
    if not os.path.isdir(checkpoints_root):
        return []

    results: list[str] = []
    for name in sorted(os.listdir(checkpoints_root)):
        step_dir = os.path.join(checkpoints_root, name)
        if not os.path.isdir(step_dir):
            continue
        if name != "best_model" and not name.startswith("global_step_"):
            continue
        if is_committed_checkpoint(step_dir):
            results.append(os.path.join("checkpoints", name))
    return results


def local_files(step_dir: str) -> list[str]:
    files: list[str] = []
    for root, _, names in os.walk(step_dir):
        files.extend(os.path.join(root, name) for name in names)
    return files


def marker_mtime(local_run_dir: str, relative_dir: str) -> float:
    return os.path.getmtime(
        os.path.join(local_run_dir, relative_dir, CHECKPOINT_COMMIT_MARKER)
    )


def _clear_proxy_env() -> None:
    for proxy_var in _PROXY_ENV_VARS:
        os.environ.pop(proxy_var, None)


def create_oss_client() -> Any:
    """Build a boto3 S3 client for Aliyun OSS."""
    import boto3
    from botocore.config import Config

    _clear_proxy_env()

    access_key = os.environ.get("ALI__AWS_ACCESS_KEY_ID") or os.environ.get(
        "AWS_ACCESS_KEY_ID"
    )
    secret_key = os.environ.get("ALI__AWS_SECRET_ACCESS_KEY") or os.environ.get(
        "AWS_SECRET_ACCESS_KEY"
    )
    endpoint = os.environ.get(
        "ALI__OSS_ENDPOINT", "https://oss-cn-wulanchabu-internal.aliyuncs.com"
    )
    addressing_style = os.environ.get("ALI__AWS_S3_ADDRESSING_STYLE", "virtual")
    if not access_key or not secret_key:
        raise RuntimeError(
            "ALI__AWS_ACCESS_KEY_ID/AWS_ACCESS_KEY_ID and corresponding "
            "secret keys are required"
        )
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="cn-wulanchabu",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": addressing_style},
            # Newer botocore otherwise uses aws-chunked trailers, which
            # Aliyun OSS's S3-compatible API rejects.
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
        ),
    )


def remote_size(client: Any, bucket: str, key: str) -> int | None:
    from botocore.exceptions import ClientError

    try:
        return int(client.head_object(Bucket=bucket, Key=key)["ContentLength"])
    except ClientError as error:
        if error.response.get("Error", {}).get("Code") in {
            "404",
            "NoSuchKey",
            "NotFound",
        }:
            return None
        raise


def upload_step(
    client: Any,
    bucket: str,
    remote_prefix: str,
    local_run_dir: str,
    relative_dir: str,
    workers: int,
) -> None:
    """Upload and verify every file in one committed checkpoint directory."""
    step_dir = os.path.join(local_run_dir, relative_dir)
    files = local_files(step_dir)
    if not files:
        raise RuntimeError(f"Checkpoint contains no files: {step_dir}")
    log(
        f"uploading {step_dir} ({len(files)} files) -> "
        f"s3://{bucket}/{object_key(remote_prefix, relative_dir)}"
    )

    def upload_one(local_path: str) -> None:
        relative_path = os.path.relpath(local_path, local_run_dir)
        destination = object_key(remote_prefix, relative_path)
        expected_size = os.path.getsize(local_path)
        if remote_size(client, bucket, destination) == expected_size:
            return
        client.upload_file(local_path, bucket, destination)

    errors: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(upload_one, path): path for path in files}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as error:  # noqa: BLE001 - collect then fail
                errors.append((futures[future], repr(error)))

    if errors:
        for path, error in errors[:20]:
            log(f"ERROR {path}: {error}")
        raise RuntimeError(f"{len(errors)} file(s) failed to upload")

    invalid = []
    for local_path in files:
        relative_path = os.path.relpath(local_path, local_run_dir)
        destination = object_key(remote_prefix, relative_path)
        if remote_size(client, bucket, destination) != os.path.getsize(local_path):
            invalid.append(relative_path)
    if invalid:
        raise RuntimeError(
            f"Remote verification failed for {len(invalid)} file(s): {invalid[:10]}"
        )

    log(f"verified {len(files)} files for {step_dir}")


def sync_run_file(
    client: Any,
    bucket: str,
    remote_prefix: str,
    local_run_dir: str,
    filename: str,
) -> None:
    """Upload a mutable experiment-level file when it exists."""
    source = os.path.join(local_run_dir, filename)
    if not os.path.isfile(source):
        return
    destination = object_key(remote_prefix, filename)
    try:
        client.upload_file(source, bucket, destination)
    except Exception as error:  # noqa: BLE001 - retry on next poll
        log(f"{filename} sync failed; will retry: {error!r}")


def select_pending_uploads(
    local_run_dir: str,
    uploaded_mtimes: dict[str, float],
    now: float,
    settle_secs: int,
) -> tuple[list[tuple[str, float]], bool]:
    """Return settled checkpoints that still need upload, and whether any remain.

    The second value is True while a committed tree is still settling or has
    not been uploaded yet, so the drain loop does not exit early.
    """
    pending: list[tuple[str, float]] = []
    has_unfinished = False
    for relative_dir in committed_checkpoint_dirs(local_run_dir):
        committed_at = marker_mtime(local_run_dir, relative_dir)
        last_uploaded = uploaded_mtimes.get(relative_dir)
        if last_uploaded is not None and last_uploaded >= committed_at:
            continue
        has_unfinished = True
        if now - committed_at < settle_secs:
            continue
        pending.append((relative_dir, committed_at))
    return pending, has_unfinished


def watch_and_upload(
    local_run_dir: str,
    oss_run_dir: str,
    *,
    poll_secs: int = 60,
    settle_secs: int = 10,
    workers: int = 16,
    max_idle_polls: int = 0,
) -> int:
    """Poll ``local_run_dir`` and upload newly committed checkpoints."""
    bucket, remote_prefix = parse_oss_uri(oss_run_dir)
    client = create_oss_client()
    uploaded_mtimes: dict[str, float] = {}
    idle_polls = 0
    log(f"watching {local_run_dir} -> {oss_run_dir}")

    while True:
        now = time.time()
        pending, has_unfinished = select_pending_uploads(
            local_run_dir, uploaded_mtimes, now, settle_secs
        )

        for relative_dir, committed_at in pending:
            try:
                upload_step(
                    client,
                    bucket,
                    remote_prefix,
                    local_run_dir,
                    relative_dir,
                    workers,
                )
                uploaded_mtimes[relative_dir] = committed_at
            except Exception as error:  # noqa: BLE001 - retry on next poll
                log(f"{relative_dir} failed; will retry: {error!r}")

        sync_run_file(client, bucket, remote_prefix, local_run_dir, "loss.txt")

        if max_idle_polls and not has_unfinished:
            idle_polls += 1
            if idle_polls >= max_idle_polls:
                log(f"no pending checkpoints for {idle_polls} polls; exiting")
                return 0
        else:
            idle_polls = 0

        time.sleep(poll_secs)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--local-run-dir",
        required=True,
        help="Experiment directory containing checkpoints/",
    )
    parser.add_argument(
        "--oss-run-dir",
        required=True,
        help="s3://bucket/prefix destination for this experiment",
    )
    parser.add_argument("--poll-secs", type=int, default=60)
    parser.add_argument("--settle-secs", type=int, default=10)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-idle-polls", type=int, default=0)
    args = parser.parse_args()

    return watch_and_upload(
        args.local_run_dir,
        args.oss_run_dir,
        poll_secs=args.poll_secs,
        settle_secs=args.settle_secs,
        workers=args.workers,
        max_idle_polls=args.max_idle_polls,
    )


if __name__ == "__main__":
    sys.exit(main())
