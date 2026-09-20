import copy
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.rollout import validate_rlt_stage2_configs
from rlinf.data.rlt import build_offline_rlt_trajectories
from rlinf.envs.remote_songling import SonglingActionCodec
from rlinf.models.embodiment.mlp_policy.rlt_td3_mlp_policy import RLTTD3MLPPolicy
from rlinf.serving.sseval_contract import (
    ACTION_DIM,
    CHUNK_LEN,
    RLT_SSEVAL_PROTOCOL_VERSION,
)
from rlinf.serving.sseval_learner import (
    InProcessLearnerConfig,
    InProcessRLTTD3Learner,
    make_sseval_run_save_dir,
    replay_catch_up_updates,
    residual_temporal_losses,
)
from rlinf.serving.sseval_runtime import (
    SsEvalRLTRuntime,
    align_feature_model_from_sft_config,
    apply_sseval_restore,
    resolve_sseval_checkpoint_dir,
    resolve_sseval_replay_dir,
    sseval_replay_weights_id,
)


class FakeFeatureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()), requires_grad=False)
        self.rlt_checkpoint_version = "fake-stage1"

    def extract_rlt_obs(self, env_obs, rng=None):
        batch = env_obs["states"].shape[0]
        value = env_obs["states"][:, :1]
        return {
            "z_rl": value.repeat(1, 8),
            "proprio": env_obs["states"],
            "ref_chunk": torch.zeros(batch, CHUNK_LEN, ACTION_DIM),
        }


class FakeLearner:
    def __init__(self):
        self.actor_ready = False
        self.submitted = []
        self.candidate = None
        self.saved = []

    def raise_if_failed(self):
        return None

    def submit(self, trajectory):
        self.submitted.append(trajectory)

    def take_candidate(self):
        candidate, self.candidate = self.candidate, None
        return candidate

    def save_checkpoint(self, path):
        self.saved.append(str(path))

    def close(self):
        pass


def _observation(value=0.0):
    return {
        "states": np.full(14, value, dtype=np.float32),
        "main_images": np.zeros((4, 5, 3), dtype=np.uint8),
        "wrist_images": np.zeros((2, 4, 5, 3), dtype=np.uint8),
    }


def _feedback(
    chunk_id=0,
    *,
    done=False,
    truncated=False,
    mode="actor",
    reward=0.0,
    episode_id="episode-1",
):
    rewards = np.zeros(CHUNK_LEN, dtype=np.float32)
    terminated = np.zeros(CHUNK_LEN, dtype=bool)
    trunc = np.zeros(CHUNK_LEN, dtype=bool)
    if done:
        terminated[-1] = True
        rewards[-1] = float(reward)
    if truncated:
        trunc[-1] = True
    return {
        "rlt_protocol_version": RLT_SSEVAL_PROTOCOL_VERSION,
        "episode_id": episode_id,
        "chunk_id": chunk_id,
        "selected_mode": mode,
        "executed_actions": np.full((CHUNK_LEN, ACTION_DIM), 0.5, dtype=np.float32),
        "valid_step_mask": np.ones(CHUNK_LEN, dtype=bool),
        "rewards": rewards,
        "terminated": terminated,
        "truncated": trunc,
        "intervene_flags": np.full(CHUNK_LEN, mode == "human", dtype=bool),
        "rlt_switch_flags": np.ones(CHUNK_LEN, dtype=bool),
        "actor_version": 0,
        "timestamps": {},
    }


def _runtime():
    policy = RLTTD3MLPPolicy(
        z_dim=8, proprio_dim=14, action_dim=14, num_action_chunks=CHUNK_LEN
    )
    learner = FakeLearner()
    runtime = SsEvalRLTRuntime(
        feature_model=FakeFeatureModel(),
        active_policy_model=policy,
        action_codec=SonglingActionCodec([-1.0] * 14, [1.0] * 14),
        learner=learner,
    )
    return runtime, learner


def test_runtime_inference_does_not_autosave():
    runtime, learner = _runtime()
    runtime.start_episode("episode-1", "fold clothes")
    runtime.infer_candidates(_observation(0.0))
    runtime.infer_candidates(_observation(1.0), _feedback(0))
    runtime.close()

    assert learner.saved == []


def _offline_rlt_trajectory(chunk_len: int = 10, model_weights_id: str = "offline"):
    features = [
        {
            "z_rl": torch.zeros(8),
            "proprio": torch.zeros(14),
            "ref_chunk": torch.zeros(chunk_len, 14),
        }
        for _ in range(chunk_len + 1)
    ]
    episode = {
        "executed_actions": torch.zeros(chunk_len, 14),
        "rewards": torch.zeros(chunk_len),
        "terminated": torch.tensor([False] * (chunk_len - 1) + [True]),
        "truncated": torch.zeros(chunk_len, dtype=torch.bool),
    }
    return build_offline_rlt_trajectories(
        episode,
        features,
        chunk_len=chunk_len,
        transition_stride=chunk_len,
        model_weights_id=model_weights_id,
    )[0]


def _td3_learner(*, save_interval=0, save_dir=None, utd=2):
    model = RLTTD3MLPPolicy(
        z_dim=8,
        proprio_dim=14,
        action_dim=14,
        num_action_chunks=10,
        actor_noise_sigma=0.0,
    )
    return InProcessRLTTD3Learner(
        model,
        InProcessLearnerConfig(
            batch_size=1,
            min_buffer_size=1,
            replay_capacity=8,
            utd=utd,
            actor_update_interval=2,
            warmup_updates=2,
        ),
        start_background=False,
        save_interval=save_interval,
        save_dir=save_dir,
    )


def test_learner_autosaves_every_save_interval_update_steps(tmp_path):
    learner = _td3_learner(save_interval=2, save_dir=tmp_path, utd=2)
    trajectory = _offline_rlt_trajectory()
    learner.add_and_train(trajectory)
    learner.add_and_train(trajectory)

    run_dir = learner.save_dir
    assert run_dir is not None
    assert run_dir.parent == tmp_path
    assert (run_dir / "step_2" / "learner.pt").is_file()
    assert (run_dir / "step_4" / "learner.pt").is_file()
    assert not (run_dir / "step_1").exists()
    assert not (run_dir / "step_3").exists()
    assert not (tmp_path / "step_2").exists()
    learner.close()


def test_learner_close_force_saves_remainder(tmp_path):
    learner = _td3_learner(save_interval=5, save_dir=tmp_path, utd=2)
    learner.add_and_train(_offline_rlt_trajectory())
    run_dir = learner.save_dir
    assert run_dir is not None
    assert not (run_dir / "step_2").exists()

    learner.close()

    assert (run_dir / "step_2" / "learner.pt").is_file()
    assert not (tmp_path / "step_2").exists()


def test_make_sseval_run_save_dir_nests_timestamp_and_avoids_collision(tmp_path):
    first = make_sseval_run_save_dir(tmp_path, stamp="20260920_153000")
    second = make_sseval_run_save_dir(tmp_path, stamp="20260920_153000")
    assert first == tmp_path / "20260920_153000"
    assert second == tmp_path / "20260920_153000_1"
    assert first.is_dir() and second.is_dir()


def test_sseval_replay_weights_id_sanitizes_episode_and_names_mode():
    assert sseval_replay_weights_id("episode-1", 0, "actor") == (
        "ep-episode-1_chunk-0_actor"
    )
    assert sseval_replay_weights_id("sess/foo bar", 3, "vla") == (
        "ep-sess-foo-bar_chunk-3_vla"
    )
    assert sseval_replay_weights_id("///", 0, "human") == "ep-episode_chunk-0_human"


def test_runtime_emits_candidates_and_holds_feedback_until_outcome():
    runtime, learner = _runtime()
    runtime.start_episode("episode-1", "fold clothes")
    first = runtime.infer_candidates(_observation(0.0))
    second = runtime.infer_candidates(_observation(1.0), _feedback(0))

    assert first.episode_id == "episode-1"
    assert first.chunk_id == 0
    assert second.chunk_id == 1
    assert first.vla_action.shape == (CHUNK_LEN, ACTION_DIM)
    assert first.actor_action.shape == (CHUNK_LEN, ACTION_DIM)
    assert learner.submitted == []

    runtime.end_episode(_feedback(1, done=True, reward=1.0))

    assert len(learner.submitted) == 2
    trajectory = learner.submitted[0]
    assert trajectory.model_weights_id == "ep-episode-1_chunk-0_actor"
    assert learner.submitted[1].model_weights_id == "ep-episode-1_chunk-1_actor"
    torch.testing.assert_close(
        trajectory.actions.reshape(CHUNK_LEN, ACTION_DIM),
        torch.full((CHUNK_LEN, ACTION_DIM), 0.5),
    )
    assert trajectory.forward_inputs["actor_switch"].all()
    assert not trajectory.terminations.any()
    assert learner.submitted[1].terminations.any()
    assert learner.submitted[1].rewards.reshape(-1)[-1] == 1.0


def test_runtime_rejects_out_of_order_feedback():
    runtime, _ = _runtime()
    runtime.start_episode("episode-1", "fold clothes")
    runtime.infer_candidates(_observation())
    with pytest.raises(ValueError, match="Out-of-order"):
        runtime.infer_candidates(_observation(), _feedback(2))


def test_runtime_activates_candidate_only_when_starting_episode():
    runtime, learner = _runtime()
    actor_state = {
        key: torch.full_like(value, 0.25)
        for key, value in runtime.active_policy_model.actor.state_dict().items()
    }
    learner.candidate = (12, actor_state)
    learner.actor_ready = True

    runtime.start_episode("episode-1", "fold clothes")

    assert runtime.active_actor_version == 12
    assert all(
        torch.all(value == 0.25)
        for value in runtime.active_policy_model.actor.state_dict().values()
    )


def test_in_process_learner_runs_utd_and_delays_actor_update():
    model = RLTTD3MLPPolicy(
        z_dim=8,
        proprio_dim=14,
        action_dim=14,
        num_action_chunks=10,
        actor_noise_sigma=0.0,
    )
    config = InProcessLearnerConfig(
        batch_size=1,
        min_buffer_size=1,
        replay_capacity=8,
        utd=2,
        actor_update_interval=2,
        residual_velocity_weight=0.5,
        residual_acceleration_weight=0.1,
        warmup_updates=2,
    )
    learner = InProcessRLTTD3Learner(model, config, start_background=False)
    features = [
        {
            "z_rl": torch.zeros(8),
            "proprio": torch.zeros(14),
            "ref_chunk": torch.zeros(10, 14),
        },
        {
            "z_rl": torch.ones(8),
            "proprio": torch.ones(14),
            "ref_chunk": torch.zeros(10, 14),
        },
    ]
    episode = {
        "executed_actions": torch.zeros(10, 14),
        "rewards": torch.zeros(10),
        "terminated": torch.tensor([False] * 9 + [True]),
        "truncated": torch.zeros(10, dtype=torch.bool),
    }
    # One C-step transition needs current and terminal next features.
    features = [features[0], *([features[0]] * 9), features[1]]
    trajectory = build_offline_rlt_trajectories(
        episode, features, chunk_len=10, transition_stride=10
    )[0]

    learner.add_and_train(trajectory)

    assert learner.update_step == 2
    assert learner.actor_ready
    assert learner.last_metrics["actor_updated"] == 1.0
    assert learner.last_metrics["residual_velocity_loss"] >= 0.0
    assert learner.last_metrics["residual_acceleration_loss"] >= 0.0
    assert learner.last_metrics["residual_smoothness_loss"] >= 0.0
    assert learner.take_candidate() is not None
    learner.close()


def test_terminal_feedback_closes_runtime_episode():
    runtime, learner = _runtime()
    runtime.start_episode("episode-1", "fold clothes")
    runtime.infer_candidates(_observation())
    runtime.infer_candidates(_observation(1.0), _feedback(0, done=True, reward=1.0))
    assert len(learner.submitted) == 1
    assert learner.submitted[0].terminations.any()
    with pytest.raises(RuntimeError, match="start_episode"):
        runtime.infer_candidates(_observation(2.0))


def test_runtime_commits_failure_episode_to_replay():
    runtime, learner = _runtime()
    runtime.start_episode("episode-1", "fold clothes")
    runtime.infer_candidates(_observation(0.0))
    runtime.infer_candidates(_observation(1.0), _feedback(0))
    runtime.end_episode(_feedback(1, done=True, reward=0.0))

    assert len(learner.submitted) == 2
    assert learner.submitted[-1].terminations.any()
    assert not learner.submitted[-1].truncations.any()
    assert learner.submitted[-1].rewards.reshape(-1)[-1] == 0.0


def test_human_feedback_writes_executed_actions_into_ref_chunk():
    runtime, learner = _runtime()
    runtime.start_episode("episode-1", "fold clothes")
    runtime.infer_candidates(_observation(0.0))
    runtime.infer_candidates(_observation(1.0), _feedback(0, mode="human"))
    runtime.end_episode(_feedback(1, done=True, reward=1.0, mode="human"))

    assert len(learner.submitted) == 2
    for trajectory in learner.submitted:
        flags = trajectory.intervene_flags.reshape(-1)
        assert flags.all()
        ref = trajectory.curr_obs["ref_chunk"].reshape(-1, ACTION_DIM)
        act = trajectory.actions.reshape(-1, ACTION_DIM)
        torch.testing.assert_close(ref, act)
        assert torch.allclose(ref, torch.full_like(ref, 0.5))


def test_actor_feedback_keeps_vla_ref_chunk():
    runtime, learner = _runtime()
    runtime.start_episode("episode-1", "fold clothes")
    runtime.infer_candidates(_observation(0.0))
    runtime.infer_candidates(_observation(1.0), _feedback(0, mode="actor"))
    runtime.end_episode(_feedback(1, done=True, reward=1.0, mode="actor"))

    assert len(learner.submitted) == 2
    for trajectory in learner.submitted:
        assert not trajectory.intervene_flags.any()
        ref = trajectory.curr_obs["ref_chunk"].reshape(-1)
        assert torch.count_nonzero(ref) == 0


def test_runtime_drops_truncated_episode():
    runtime, learner = _runtime()
    runtime.start_episode("episode-1", "fold clothes")
    runtime.infer_candidates(_observation(0.0))
    runtime.infer_candidates(_observation(1.0), _feedback(0, mode="human"))
    runtime.end_episode(_feedback(1, truncated=True, mode="human"))

    assert learner.submitted == []
    with pytest.raises(RuntimeError, match="start_episode"):
        runtime.infer_candidates(_observation(2.0))


def test_runtime_drops_truncated_feedback_on_next_observation():
    runtime, learner = _runtime()
    runtime.start_episode("episode-1", "fold clothes")
    runtime.infer_candidates(_observation())
    runtime.infer_candidates(_observation(1.0), _feedback(0, truncated=True))

    assert learner.submitted == []
    with pytest.raises(RuntimeError, match="start_episode"):
        runtime.infer_candidates(_observation(2.0))


def test_runtime_drops_unscored_buffer_on_reset_and_close():
    runtime, learner = _runtime()
    runtime.start_episode("episode-1", "fold clothes")
    runtime.infer_candidates(_observation(0.0))
    runtime.infer_candidates(_observation(1.0), _feedback(0))
    runtime.reset()

    assert learner.submitted == []
    runtime.start_episode("episode-2", "fold clothes")
    runtime.infer_candidates(_observation(0.0))
    runtime.infer_candidates(_observation(1.0), _feedback(0, episode_id="episode-2"))
    runtime.close()
    assert learner.submitted == []


def test_in_process_learner_checkpoint_round_trip(tmp_path):
    def make_learner():
        model = RLTTD3MLPPolicy(
            z_dim=8,
            proprio_dim=14,
            action_dim=14,
            num_action_chunks=10,
            actor_noise_sigma=0.0,
        )
        return InProcessRLTTD3Learner(
            model,
            InProcessLearnerConfig(
                batch_size=1,
                min_buffer_size=1,
                replay_capacity=8,
                utd=2,
                actor_update_interval=2,
                warmup_updates=2,
            ),
            start_background=False,
        )

    features = [
        {
            "z_rl": torch.zeros(8),
            "proprio": torch.zeros(14),
            "ref_chunk": torch.zeros(10, 14),
        }
        for _ in range(11)
    ]
    episode = {
        "executed_actions": torch.zeros(10, 14),
        "rewards": torch.zeros(10),
        "terminated": torch.tensor([False] * 9 + [True]),
        "truncated": torch.zeros(10, dtype=torch.bool),
    }
    trajectory = build_offline_rlt_trajectories(
        episode, features, chunk_len=10, transition_stride=10
    )[0]
    first = make_learner()
    first.add_and_train(trajectory)
    checkpoint = tmp_path / "learner"
    first.save_checkpoint(checkpoint)

    restored = make_learner()
    restored.load_checkpoint(checkpoint)

    assert restored.update_step == 2
    assert restored.total_transitions == 1
    assert restored.replay.total_samples == 1
    assert restored.take_candidate() is not None
    first.close()
    restored.close()


def test_resolve_sseval_checkpoint_dir_requires_step_directory(tmp_path):
    with pytest.raises(ValueError, match="step_N directory"):
        resolve_sseval_checkpoint_dir(tmp_path / "learner.pt")
    empty = tmp_path / "step_1"
    empty.mkdir()
    with pytest.raises(ValueError, match="missing learner.pt"):
        resolve_sseval_checkpoint_dir(empty)
    (empty / "learner.pt").write_bytes(b"ok")
    assert resolve_sseval_checkpoint_dir(empty) == empty.resolve()


def test_runtime_loads_step_directory_checkpoint(tmp_path):
    def make_runtime():
        policy = RLTTD3MLPPolicy(
            z_dim=8,
            proprio_dim=14,
            action_dim=14,
            num_action_chunks=10,
            actor_noise_sigma=0.0,
        )
        learner = InProcessRLTTD3Learner(
            copy.deepcopy(policy).train().requires_grad_(True),
            InProcessLearnerConfig(
                batch_size=1,
                min_buffer_size=1,
                replay_capacity=8,
                utd=2,
                actor_update_interval=2,
                warmup_updates=2,
            ),
            start_background=False,
        )
        return SsEvalRLTRuntime(
            feature_model=FakeFeatureModel(),
            active_policy_model=policy,
            action_codec=SonglingActionCodec([-1.0] * 14, [1.0] * 14),
            learner=learner,
            chunk_len=10,
        )

    features = [
        {
            "z_rl": torch.zeros(8),
            "proprio": torch.zeros(14),
            "ref_chunk": torch.zeros(10, 14),
        }
        for _ in range(11)
    ]
    episode = {
        "executed_actions": torch.zeros(10, 14),
        "rewards": torch.zeros(10),
        "terminated": torch.tensor([False] * 9 + [True]),
        "truncated": torch.zeros(10, dtype=torch.bool),
    }
    trajectory = build_offline_rlt_trajectories(
        episode, features, chunk_len=10, transition_stride=10
    )[0]
    first = make_runtime()
    first.learner.add_and_train(trajectory)
    step_dir = tmp_path / "step_2"
    first.save_checkpoint(step_dir)
    saved = {
        key: value.detach().cpu().clone()
        for key, value in first.learner.model.state_dict().items()
    }

    restored = make_runtime()
    restored.load_checkpoint(step_dir)

    assert restored.learner.update_step == 2
    assert restored.learner.replay.total_samples == 1
    for key, value in saved.items():
        torch.testing.assert_close(
            restored.active_policy_model.state_dict()[key].cpu(), value
        )
    first.close()
    restored.close()


def test_replay_catch_up_updates_matches_live_collection():
    assert replay_catch_up_updates(199, min_buffer_size=200, utd=5) == 0
    assert replay_catch_up_updates(200, min_buffer_size=200, utd=5) == 5
    assert replay_catch_up_updates(499, min_buffer_size=200, utd=5) == 1500


def test_resolve_sseval_replay_dir_accepts_step_or_replay_folder(tmp_path):
    step_dir = tmp_path / "step_1500"
    replay_dir = step_dir / "replay"
    replay_dir.mkdir(parents=True)
    (replay_dir / "metadata.json").write_text("{}", encoding="utf-8")
    assert resolve_sseval_replay_dir(step_dir) == replay_dir.resolve()
    assert resolve_sseval_replay_dir(replay_dir) == replay_dir.resolve()
    empty = tmp_path / "step_1"
    empty.mkdir()
    with pytest.raises(ValueError, match="metadata.json"):
        resolve_sseval_replay_dir(empty)


def test_apply_sseval_restore_rejects_actor_and_replay_together():
    runtime, _ = _runtime()
    cfg = OmegaConf.create(
        {
            "actor_checkpoint": "results/ckpts/step_1",
            "replay_checkpoint": "results/ckpts/step_2",
        }
    )
    with pytest.raises(ValueError, match="cannot both be set"):
        apply_sseval_restore(runtime, cfg, Path("unused.yaml"))


def _catch_up_learner(*, min_buffer_size, utd=2, save_interval=0, save_dir=None):
    model = RLTTD3MLPPolicy(
        z_dim=8,
        proprio_dim=14,
        action_dim=14,
        num_action_chunks=10,
        actor_noise_sigma=0.0,
    )
    return InProcessRLTTD3Learner(
        model,
        InProcessLearnerConfig(
            batch_size=1,
            min_buffer_size=min_buffer_size,
            replay_capacity=8,
            utd=utd,
            actor_update_interval=2,
            warmup_updates=100,
        ),
        start_background=False,
        save_interval=save_interval,
        save_dir=save_dir,
    )


def test_load_replay_catches_up_without_loading_actor(tmp_path):
    first = _catch_up_learner(min_buffer_size=3, utd=2)
    for index in range(5):
        first.add_and_train(_offline_rlt_trajectory(model_weights_id=f"t{index}"))
    step_dir = tmp_path / "step_6"
    first.save_checkpoint(step_dir)
    saved_actor = {
        key: value.detach().cpu().clone()
        for key, value in first.model.actor.state_dict().items()
    }
    (step_dir / "learner.pt").unlink()

    restored = _catch_up_learner(
        min_buffer_size=3, utd=2, save_interval=2, save_dir=tmp_path / "caught"
    )
    init_actor = {
        key: value.detach().cpu().clone()
        for key, value in restored.model.actor.state_dict().items()
    }
    restored.load_replay(step_dir / "replay")

    assert restored.replay.total_samples == 5
    assert restored.total_transitions == 5
    assert restored.update_step == replay_catch_up_updates(5, min_buffer_size=3, utd=2)
    assert restored.update_step == 6
    assert any(
        not torch.allclose(restored.model.actor.state_dict()[key], saved_actor[key])
        for key in saved_actor
    )
    assert any(
        not torch.equal(restored.model.actor.state_dict()[key], init_actor[key])
        for key in init_actor
    )
    run_dir = restored.save_dir
    assert run_dir is not None
    assert run_dir.parent == tmp_path / "caught"
    assert (run_dir / "step_2" / "learner.pt").is_file()
    assert (run_dir / "step_4" / "learner.pt").is_file()
    assert (run_dir / "step_6" / "learner.pt").is_file()
    assert not (run_dir / "step_1").exists()
    first.close()
    restored.close()


def test_load_replay_skips_catch_up_below_min_buffer(tmp_path):
    first = _catch_up_learner(min_buffer_size=1, utd=2)
    first.add_and_train(_offline_rlt_trajectory())
    step_dir = tmp_path / "step_2"
    first.save_checkpoint(step_dir)

    restored = _catch_up_learner(min_buffer_size=5, utd=2)
    init_actor = {
        key: value.detach().cpu().clone()
        for key, value in restored.model.actor.state_dict().items()
    }
    restored.load_replay(step_dir / "replay")

    assert restored.replay.total_samples == 1
    assert restored.update_step == 0
    assert not restored.actor_ready
    for key, value in init_actor.items():
        torch.testing.assert_close(restored.model.actor.state_dict()[key], value)
    first.close()
    restored.close()


def test_runtime_load_replay_keeps_fresh_actor(tmp_path):
    def make_runtime():
        policy = RLTTD3MLPPolicy(
            z_dim=8,
            proprio_dim=14,
            action_dim=14,
            num_action_chunks=10,
            actor_noise_sigma=0.0,
        )
        learner = InProcessRLTTD3Learner(
            copy.deepcopy(policy).train().requires_grad_(True),
            InProcessLearnerConfig(
                batch_size=1,
                min_buffer_size=1,
                replay_capacity=8,
                utd=2,
                actor_update_interval=2,
                warmup_updates=2,
            ),
            start_background=False,
        )
        return SsEvalRLTRuntime(
            feature_model=FakeFeatureModel(),
            active_policy_model=policy,
            action_codec=SonglingActionCodec([-1.0] * 14, [1.0] * 14),
            learner=learner,
            chunk_len=10,
        )

    first = make_runtime()
    first.learner.add_and_train(_offline_rlt_trajectory())
    step_dir = tmp_path / "step_2"
    first.save_checkpoint(step_dir)
    saved = {
        key: value.detach().cpu().clone()
        for key, value in first.learner.model.state_dict().items()
    }

    restored = make_runtime()
    init_actor = {
        key: value.detach().cpu().clone()
        for key, value in restored.active_policy_model.actor.state_dict().items()
    }
    restored.load_replay_checkpoint(step_dir)

    assert restored.learner.update_step == 2
    assert restored.learner.replay.total_samples == 1
    assert restored.active_actor_version == 2
    assert not all(
        torch.allclose(restored.active_policy_model.state_dict()[key].cpu(), value)
        for key, value in saved.items()
    )
    assert any(
        not torch.equal(
            restored.active_policy_model.actor.state_dict()[key].cpu(), init_actor[key]
        )
        for key in init_actor
    )
    first.close()
    restored.close()


def test_align_copies_stage1_fields_from_sft_yaml(tmp_path):
    sft = tmp_path / "sft.yaml"
    sft.write_text(
        "\n".join(
            [
                "hydra:",
                "  searchpath:",
                "    - file://${oc.env:EMBODIED_PATH}/config/",
                "actor:",
                "  model:",
                "    model_type: openpi_rlinf",
                "    num_action_chunks: 10",
                "    action_dim: 14",
                "    num_steps: 5",
                "    add_value_head: false",
                "    openpi_data:",
                "      repo_id: songling/bfjm",
                "      default_prompt: build with blocks",
                "      norm_stats_path: /tmp/norm_stats.json",
                "    openpi:",
                "      task: sft",
                "      config_name: pi05_rlt_songling_joint",
                "      action_chunk: ${actor.model.num_action_chunks}",
                "      action_env_dim: ${actor.model.action_dim}",
                "      num_steps: ${actor.model.num_steps}",
                "      action_horizon: 99",
                "      model_action_dim: 32",
                "      use_rlt: true",
                "      rlt_embed_dim: 2048",
                "      rlt_image_only: false",
                "      rlt_use_mask: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text(
        "\n".join(
            [
                f"stage1_sft_config: {sft}",
                "feature_model:",
                "  model_path: /tmp/stage1",
                "  precision: bf16",
                "  num_action_chunks: 99",
                "actor_model:",
                "  num_action_chunks: 99",
                "  ref_num_action_chunks: 99",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = OmegaConf.load(str(runtime))
    align_feature_model_from_sft_config(cfg, runtime)

    assert cfg.feature_model.model_path == "/tmp/stage1"
    assert cfg.feature_model.precision == "bf16"
    assert cfg.feature_model.num_action_chunks == 10
    assert cfg.feature_model.action_dim == 14
    assert cfg.actor_model.num_action_chunks == 10
    assert cfg.actor_model.ref_num_action_chunks == 10
    assert cfg.feature_model.num_steps == 5
    assert cfg.feature_model.openpi.task == "eval"
    assert cfg.feature_model.openpi.config_name == "pi05_rlt_songling_joint"
    assert cfg.feature_model.openpi.action_chunk == 10
    assert cfg.feature_model.openpi.action_horizon == 10
    assert cfg.feature_model.openpi.action_env_dim == 14
    assert cfg.feature_model.openpi.num_steps == 5
    assert cfg.feature_model.openpi_data.repo_id == "songling/bfjm"
    assert cfg.feature_model.openpi_data.norm_stats_path == "/tmp/norm_stats.json"


def test_align_keeps_eval_norm_stats_path(tmp_path):
    sft = tmp_path / "sft.yaml"
    sft.write_text(
        "\n".join(
            [
                "actor:",
                "  model:",
                "    model_type: openpi_rlinf",
                "    num_action_chunks: 10",
                "    action_dim: 14",
                "    num_steps: 5",
                "    add_value_head: false",
                "    openpi_data:",
                "      repo_id: songling/bfjm",
                "      default_prompt: build with blocks",
                "      norm_stats_path: /train/norm_stats.json",
                "    openpi:",
                "      task: sft",
                "      config_name: pi05_rlt_songling_joint",
                "      action_horizon: 10",
                "      model_action_dim: 32",
                "      use_rlt: true",
                "      rlt_embed_dim: 2048",
                "      rlt_image_only: false",
                "      rlt_use_mask: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    runtime = tmp_path / "runtime.yaml"
    runtime.write_text(
        "\n".join(
            [
                f"stage1_sft_config: {sft}",
                "feature_model:",
                "  model_path: /tmp/stage1",
                "  precision: bf16",
                "  openpi_data:",
                "    norm_stats_path: /eval/norm_stats.json",
                "actor_model:",
                "  num_action_chunks: 99",
                "  ref_num_action_chunks: 99",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    cfg = OmegaConf.load(str(runtime))
    align_feature_model_from_sft_config(cfg, runtime)

    assert cfg.feature_model.openpi_data.repo_id == "songling/bfjm"
    assert cfg.feature_model.openpi_data.default_prompt == "build with blocks"
    assert cfg.feature_model.openpi_data.norm_stats_path == "/eval/norm_stats.json"


def test_align_is_noop_without_stage1_sft_config():
    cfg = OmegaConf.create({"feature_model": {"num_action_chunks": 10}})
    align_feature_model_from_sft_config(cfg, Path("unused.yaml"))
    assert cfg.feature_model.num_action_chunks == 10


def test_align_checked_in_songling_sft_config():
    runtime = (
        Path(__file__).resolve().parents[2]
        / "examples/embodiment/config/songling_rlt_sseval_td3.yaml"
    )
    cfg = OmegaConf.load(str(runtime))
    align_feature_model_from_sft_config(cfg, runtime)
    assert cfg.feature_model.openpi.task == "eval"
    assert cfg.feature_model.openpi.config_name == "pi05_rlt_songling_joint"
    assert cfg.feature_model.openpi_data.repo_id == "songling/bfjm"
    assert cfg.feature_model.num_action_chunks == 50
    assert cfg.feature_model.openpi.action_horizon == 50
    assert cfg.feature_model.openpi.action_chunk == 50
    assert cfg.actor_model.num_action_chunks == 50
    assert cfg.actor_model.ref_num_action_chunks == 50
    raw_feature = OmegaConf.to_container(cfg.feature_model, resolve=False)
    raw_actor = OmegaConf.to_container(cfg.actor_model, resolve=False)
    validate_rlt_stage2_configs(raw_actor, raw_feature)
    assert raw_feature["model_path"] == (
        "/home/fanyiming/RLinf/checkpoints/"
        "songling_bfjm_rlt_stage1_sft_openpi_pi05/global_step_10000/actor"
    )
    assert raw_feature["openpi_data"]["norm_stats_path"] == (
        "/home/fanyiming/RLinf/checkpoints/"
        "songling_bfjm_rlt_stage1_sft_openpi_pi05/global_step_10000/norm_stats.json"
    )
    assert cfg.feature_model.precision == "bf16"


def test_residual_temporal_losses_are_zero_for_constant_residual():
    reference = torch.randn(2, 6, 3)
    predicted = reference + 0.25
    valid = torch.ones(2, 6, dtype=torch.bool)

    velocity, acceleration = residual_temporal_losses(
        predicted, reference, valid
    )

    torch.testing.assert_close(velocity, torch.zeros_like(velocity))
    torch.testing.assert_close(acceleration, torch.zeros_like(acceleration))


def test_residual_temporal_losses_penalize_alternating_residual():
    reference = torch.zeros(1, 6, 2)
    residual = torch.tensor([0.2, -0.2, 0.2, -0.2, 0.2, -0.2]).reshape(
        1, 6, 1
    )
    predicted = reference + residual.expand(-1, -1, 2)
    valid = torch.ones(1, 6, dtype=torch.bool)

    velocity, acceleration = residual_temporal_losses(
        predicted, reference, valid
    )

    assert velocity > 0
    assert acceleration > velocity


def test_residual_temporal_losses_ignore_invalid_tail():
    reference = torch.zeros(1, 6, 2)
    predicted = reference.clone()
    predicted[:, 3] = 10.0
    predicted[:, 4] = -10.0
    predicted[:, 5] = 10.0
    valid = torch.tensor([[True, True, True, False, False, False]])

    velocity, acceleration = residual_temporal_losses(
        predicted, reference, valid
    )

    torch.testing.assert_close(velocity, torch.zeros_like(velocity))
    torch.testing.assert_close(acceleration, torch.zeros_like(acceleration))


def test_learner_rejects_negative_residual_smoothness_weights():
    with pytest.raises(ValueError, match="residual_velocity_weight"):
        InProcessLearnerConfig.from_mapping({"residual_velocity_weight": -0.1})
