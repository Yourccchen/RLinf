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

import os
import re

CHECKPOINT_COMMIT_MARKER = "COMMITTED"


def parse_global_step_from_checkpoint_path(
    checkpoint_path: str | os.PathLike[str],
) -> int:
    """Extract the global step from a checkpoint directory path.

    Args:
        checkpoint_path: Path ending in a ``global_step_<step>`` directory.

    Returns:
        The non-negative global step encoded in the directory name.

    Raises:
        ValueError: If the final directory is not named ``global_step_<step>``.
    """
    checkpoint_dir = os.path.basename(os.path.normpath(checkpoint_path))
    match = re.fullmatch(r"global_step_(\d+)", checkpoint_dir)
    if match is None:
        raise ValueError(
            "Checkpoint path must end with a 'global_step_<step>' directory, "
            f"but got {os.fspath(checkpoint_path)!r}."
        )
    return int(match.group(1))


def write_checkpoint_commit_marker(checkpoint_dir: str | os.PathLike[str]) -> None:
    """Mark ``checkpoint_dir`` as fully written.

    The OSS sidecar uploads a step only after this file exists, so a
    partial DCP tree is never copied. The write is atomic via rename.
    """
    directory = os.fspath(checkpoint_dir)
    os.makedirs(directory, exist_ok=True)
    marker_path = os.path.join(directory, CHECKPOINT_COMMIT_MARKER)
    tmp_path = f"{marker_path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write("ok\n")
    os.replace(tmp_path, marker_path)


def is_committed_checkpoint(checkpoint_dir: str | os.PathLike[str]) -> bool:
    """Return whether ``checkpoint_dir`` has a commit marker."""
    return os.path.isfile(
        os.path.join(os.fspath(checkpoint_dir), CHECKPOINT_COMMIT_MARKER)
    )
