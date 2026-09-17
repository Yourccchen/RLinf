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

"""Shared helpers: metrics, checkpoint paths, and resume."""

from __future__ import annotations

import importlib.util
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.runners.reasoning_runner import ReasoningRunner
from rlinf.utils.metric_utils import compute_evaluate_metrics, compute_rollout_metrics


def test_compute_evaluate_metrics_reports_interact_delay_wait_time_stats():
    metrics = compute_evaluate_metrics(
        [
            {
                "success": torch.tensor([1.0, 0.0]),
                "interact_delay": torch.tensor([0.10, 0.30]),
            },
            {
                "success": torch.tensor([0.0, 1.0]),
                "interact_delay": torch.tensor([0.20, 0.40]),
            },
        ]
    )

    assert math.isclose(float(metrics["success"]), 0.5)
    assert float(metrics["average_delay"]) == pytest.approx(0.25)
    assert float(metrics["median_delay"]) == pytest.approx(0.25)
    assert float(metrics["max_delay"]) == pytest.approx(0.40)
    assert float(metrics["min_delay"]) == pytest.approx(0.10)
    assert metrics["num_trajectories"] == 4


def test_compute_evaluate_metrics_ignores_delay_samples_for_trajectory_count():
    metrics = compute_evaluate_metrics(
        [{"interact_delay": torch.tensor([0.05, 0.15, 0.25])}]
    )

    assert float(metrics["average_delay"]) == pytest.approx(0.15)
    assert metrics["num_trajectories"] == 0


def test_compute_evaluate_metrics_reports_prefixed_interact_delay_stats():
    metrics = compute_evaluate_metrics(
        [
            {
                "env/success": torch.tensor([1.0]),
                "env/interact_delay": torch.tensor([0.12, 0.24]),
            }
        ]
    )

    assert float(metrics["env/average_delay"]) == pytest.approx(0.18)
    assert float(metrics["env/median_delay"]) == pytest.approx(0.18)
    assert float(metrics["env/max_delay"]) == pytest.approx(0.24)
    assert float(metrics["env/min_delay"]) == pytest.approx(0.12)


@pytest.fixture
def single_rank_reduction(monkeypatch):
    from rlinf.scheduler.worker.worker import Worker

    monkeypatch.setattr(
        Worker, "torch_platform", SimpleNamespace(current_device=lambda: "cpu")
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)


def test_compute_rollout_metrics_reports_loss_mask_fraction(single_rank_reduction):
    metrics = compute_rollout_metrics(
        {
            "loss_mask": torch.tensor([[[True], [False]], [[True], [True]]]),
            "rewards": torch.tensor([[[1.0], [8.0]], [[2.0], [3.0]]]),
        }
    )

    assert metrics["loss_mask_fraction"] == pytest.approx(0.75)
    assert metrics["rewards"] == pytest.approx(2.0)


def test_compute_rollout_metrics_omits_loss_mask_fraction_without_mask(
    single_rank_reduction,
):
    metrics = compute_rollout_metrics({"rewards": torch.tensor([[[1.0], [3.0]]])})

    assert "loss_mask_fraction" not in metrics
    assert metrics["rewards"] == pytest.approx(2.0)


def _load_checkpoint_utils():
    module_path = (
        Path(__file__).resolve().parents[2] / "rlinf" / "utils" / "checkpoint.py"
    )
    assert module_path.exists(), "checkpoint path utilities are not implemented"
    spec = importlib.util.spec_from_file_location(
        "_rlinf_utils_checkpoint_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "checkpoint_path",
    [
        "/tmp/checkpoints/global_step_30",
        "/tmp/checkpoints/global_step_30/",
        "/tmp/checkpoints/global_step_30///",
    ],
)
def test_parse_global_step_accepts_trailing_slashes(checkpoint_path):
    checkpoint_utils = _load_checkpoint_utils()

    assert (
        checkpoint_utils.parse_global_step_from_checkpoint_path(checkpoint_path) == 30
    )


@pytest.mark.parametrize(
    "checkpoint_path",
    [
        "/tmp/checkpoints/step_30",
        "/tmp/checkpoints/global_step_latest/",
        "/tmp/checkpoints/global_step_30/actor",
    ],
)
def test_parse_global_step_rejects_invalid_checkpoint_directories(checkpoint_path):
    checkpoint_utils = _load_checkpoint_utils()

    with pytest.raises(ValueError, match="global_step_<step>"):
        checkpoint_utils.parse_global_step_from_checkpoint_path(checkpoint_path)


class _StubRunner:
    """Expose only the checkpoint helpers and state used by these tests."""

    def __init__(self, critic=None):
        self.critic = critic

    _is_complete_checkpoint = ReasoningRunner._is_complete_checkpoint


class _ImmediateHandle:
    def wait(self):
        return None


class _Actor:
    def save_checkpoint(self, path: str, _step: int):
        os.makedirs(path, exist_ok=True)
        return _ImmediateHandle()


class _Dataloader:
    def state_dict(self):
        return {"offset": 3}


def _write_checkpoint(
    root: Path, step: int, *, complete: bool, with_critic: bool = False
) -> Path:
    checkpoint_dir = root / f"global_step_{step}"
    (checkpoint_dir / "actor").mkdir(parents=True)
    if with_critic:
        (checkpoint_dir / "critic").mkdir()
    if complete:
        data_dir = checkpoint_dir / "data"
        data_dir.mkdir()
        (data_dir / "data.pt").write_bytes(b"dataloader-state")
    return checkpoint_dir


def _resolve_auto_resume(log_path: Path, *, critic=None) -> str | None:
    cfg = OmegaConf.create(
        {"runner": {"resume_dir": "auto", "logger": {"log_path": str(log_path)}}}
    )
    runner = _StubRunner(critic=critic)
    runner.cfg = cfg
    runner.init_rollout_workers = lambda: None
    runner.init_actor_critic_workers = lambda: None

    ReasoningRunner.init_workers(runner)
    return cfg.runner.resume_dir


def _saving_runner(tmp_path: Path) -> _StubRunner:
    runner = _StubRunner()
    runner.cfg = OmegaConf.create(
        {
            "runner": {
                "output_dir": str(tmp_path),
                "experiment_name": "experiment",
            }
        }
    )
    runner.global_steps = 8
    runner.actor = _Actor()
    runner.train_dataloader = _Dataloader()
    return runner


@pytest.mark.parametrize(
    "completeness,expected_step",
    [
        pytest.param({40: True, 80: False}, 40, id="skips-the-incomplete-newest"),
        pytest.param({40: True, 80: True}, 80, id="takes-the-newest-complete"),
        pytest.param({40: False}, None, id="starts-fresh-when-none-is-complete"),
    ],
)
def test_auto_resume_selects_the_newest_complete_checkpoint(
    tmp_path, completeness, expected_step
):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    for step, complete in completeness.items():
        _write_checkpoint(checkpoints_dir, step, complete=complete)

    expected = (
        None
        if expected_step is None
        else str(checkpoints_dir / f"global_step_{expected_step}")
    )
    assert _resolve_auto_resume(tmp_path) == expected


def test_checkpoint_requires_the_critic_only_when_configured(tmp_path):
    checkpoints_dir = tmp_path / "checkpoints"
    checkpoints_dir.mkdir()
    checkpoint = _write_checkpoint(checkpoints_dir, 40, complete=True)

    assert _StubRunner()._is_complete_checkpoint(str(checkpoint))
    assert not _StubRunner(critic=object())._is_complete_checkpoint(str(checkpoint))


def test_dataloader_state_is_published_atomically(tmp_path, monkeypatch):
    runner = _saving_runner(tmp_path)
    written_paths = []

    def save(_state, path):
        written_paths.append(path)
        Path(path).write_bytes(b"complete")

    monkeypatch.setattr("rlinf.runners.reasoning_runner.torch.save", save)

    ReasoningRunner._save_checkpoint(runner)

    checkpoint = tmp_path / "experiment" / "checkpoints" / "global_step_8"
    final_path = checkpoint / "data" / "data.pt"
    assert written_paths == [f"{final_path}.tmp"]
    assert final_path.read_bytes() == b"complete"
    assert not Path(f"{final_path}.tmp").exists()
    assert runner._is_complete_checkpoint(str(checkpoint))


def test_interrupted_dataloader_save_does_not_publish_completion(tmp_path, monkeypatch):
    runner = _saving_runner(tmp_path)

    def interrupted_save(_state, path):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("interrupted")

    monkeypatch.setattr("rlinf.runners.reasoning_runner.torch.save", interrupted_save)

    with pytest.raises(RuntimeError, match="interrupted"):
        ReasoningRunner._save_checkpoint(runner)

    checkpoint = tmp_path / "experiment" / "checkpoints" / "global_step_8"
    final_path = checkpoint / "data" / "data.pt"
    assert not final_path.exists()
    assert not Path(f"{final_path}.tmp").exists()
    assert not runner._is_complete_checkpoint(str(checkpoint))


def test_write_checkpoint_commit_marker_is_atomic(tmp_path):
    checkpoint_utils = _load_checkpoint_utils()
    checkpoint_dir = tmp_path / "checkpoints" / "global_step_8"
    checkpoint_dir.mkdir(parents=True)

    checkpoint_utils.write_checkpoint_commit_marker(checkpoint_dir)

    marker = checkpoint_dir / checkpoint_utils.CHECKPOINT_COMMIT_MARKER
    assert marker.read_text(encoding="utf-8") == "ok\n"
    assert not Path(f"{marker}.tmp").exists()
    assert checkpoint_utils.is_committed_checkpoint(checkpoint_dir)
    assert not checkpoint_utils.is_committed_checkpoint(tmp_path)


def test_sft_save_checkpoint_writes_commit_marker_after_actor_save(tmp_path):
    from rlinf.runners.sft_runner import SFTRunner

    runner = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "runner": {
                    "logger": {
                        "log_path": str(tmp_path),
                        "experiment_name": "exp",
                    }
                }
            }
        ),
        global_step=8,
        actor=_Actor(),
        early_stop=None,
    )

    SFTRunner._save_checkpoint(runner)

    checkpoint = tmp_path / "exp" / "checkpoints" / "global_step_8"
    assert (checkpoint / "actor").is_dir()
    assert (checkpoint / "COMMITTED").read_text(encoding="utf-8") == "ok\n"


def test_sft_save_best_checkpoint_writes_commit_marker(tmp_path):
    from rlinf.runners.sft_runner import SFTRunner

    runner = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "runner": {
                    "logger": {
                        "log_path": str(tmp_path),
                        "experiment_name": "exp",
                    }
                }
            }
        ),
        global_step=8,
        actor=_Actor(),
        early_stop=None,
    )

    SFTRunner._save_checkpoint(runner, is_best=True)

    checkpoint = tmp_path / "exp" / "checkpoints" / "best_model"
    assert (checkpoint / "COMMITTED").is_file()


def test_sft_loss_log_appends_scalar_metrics(tmp_path):
    from rlinf.runners.sft_runner import SFTRunner

    loss_log_path = tmp_path / "loss.txt"
    runner = SimpleNamespace(loss_log_path=str(loss_log_path))

    SFTRunner._write_loss_log(
        runner,
        7,
        {
            "train/loss": 1.25,
            "train/grad_norm": torch.tensor(0.5),
            "metadata": object(),
        },
    )

    assert loss_log_path.read_text(encoding="utf-8") == (
        "Step 7: train/grad_norm=0.5, train/loss=1.25\n"
    )


def _load_oss_ckpt_uploader():
    module_path = (
        Path(__file__).resolve().parents[2] / "toolkits" / "oss_ckpt_uploader.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_rlinf_oss_ckpt_uploader_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_oss_ckpt_uploader_parse_oss_uri():
    uploader = _load_oss_ckpt_uploader()

    assert uploader.parse_oss_uri(
        "s3://shengshu-base2-test/caimengchen/rlinf_runs/exp"
    ) == ("shengshu-base2-test", "caimengchen/rlinf_runs/exp")
    assert uploader.parse_oss_uri("oss://bucket/prefix/run") == ("bucket", "prefix/run")
    assert uploader.parse_oss_uri("s3+ali://bucket/a/b") == ("bucket", "a/b")


@pytest.mark.parametrize(
    "uri",
    ["s3://bucket-only", "https://oss.example/bucket/key", ""],
)
def test_oss_ckpt_uploader_rejects_invalid_uri(uri):
    uploader = _load_oss_ckpt_uploader()

    with pytest.raises(ValueError):
        uploader.parse_oss_uri(uri)


def test_oss_ckpt_uploader_lists_committed_dcp_trees(tmp_path):
    uploader = _load_oss_ckpt_uploader()
    checkpoints = tmp_path / "checkpoints"
    ready = checkpoints / "global_step_2500"
    pending = checkpoints / "global_step_5000"
    best = checkpoints / "best_model"
    ignored = checkpoints / "tmp_scratch"
    for path in (ready, pending, best, ignored):
        (path / "actor" / "dcp_checkpoint").mkdir(parents=True)
        (path / "actor" / "dcp_checkpoint" / "shard.pt").write_bytes(b"dcp")
        (path / "actor" / "model_state_dict").mkdir(parents=True, exist_ok=True)
        (path / "actor" / "model_state_dict" / "full_weights.pt").write_bytes(b"full")
    (ready / "COMMITTED").write_text("ok\n")
    (best / "COMMITTED").write_text("ok\n")

    listed = uploader.committed_checkpoint_dirs(str(tmp_path))

    assert listed == [
        os.path.join("checkpoints", "best_model"),
        os.path.join("checkpoints", "global_step_2500"),
    ]
    assert uploader.object_key(
        "bucket/exp", listed[1] + "/actor/dcp_checkpoint/shard.pt"
    ) == ("bucket/exp/checkpoints/global_step_2500/actor/dcp_checkpoint/shard.pt")


def test_oss_ckpt_uploader_waits_for_settle_before_idle(tmp_path):
    uploader = _load_oss_ckpt_uploader()
    step_dir = tmp_path / "checkpoints" / "global_step_2500"
    (step_dir / "actor").mkdir(parents=True)
    (step_dir / "COMMITTED").write_text("ok\n")
    committed_at = (step_dir / "COMMITTED").stat().st_mtime
    relative = os.path.join("checkpoints", "global_step_2500")

    pending, has_unfinished = uploader.select_pending_uploads(
        str(tmp_path), {}, committed_at + 1, settle_secs=10
    )
    assert pending == []
    assert has_unfinished is True

    pending, has_unfinished = uploader.select_pending_uploads(
        str(tmp_path), {}, committed_at + 11, settle_secs=10
    )
    assert pending == [(relative, committed_at)]
    assert has_unfinished is True

    pending, has_unfinished = uploader.select_pending_uploads(
        str(tmp_path), {relative: committed_at}, committed_at + 11, settle_secs=10
    )
    assert pending == []
    assert has_unfinished is False


def test_oss_ckpt_uploader_syncs_loss_log(tmp_path):
    uploader = _load_oss_ckpt_uploader()
    loss_log = tmp_path / "loss.txt"
    loss_log.write_text("Step 0: train/loss=1.25\n", encoding="utf-8")
    uploads = []
    client = SimpleNamespace(
        upload_file=lambda source, bucket, destination: uploads.append(
            (source, bucket, destination)
        )
    )

    uploader.sync_run_file(
        client,
        "bucket",
        "runs/experiment",
        str(tmp_path),
        "loss.txt",
    )

    assert uploads == [(str(loss_log), "bucket", "runs/experiment/loss.txt")]
