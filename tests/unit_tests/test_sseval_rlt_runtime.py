import copy

import numpy as np
import pytest
import torch

from rlinf.data.rlt import build_offline_rlt_trajectories
from rlinf.envs.remote_songling import SonglingActionCodec
from rlinf.models.embodiment.mlp_policy.rlt_td3_mlp_policy import RLTTD3MLPPolicy
from rlinf.serving.sseval_contract import RLT_SSEVAL_PROTOCOL_VERSION
from rlinf.serving.sseval_learner import (
    InProcessLearnerConfig,
    InProcessRLTTD3Learner,
)
from rlinf.serving.sseval_runtime import SsEvalRLTRuntime


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
            "ref_chunk": torch.zeros(batch, 10, 14),
        }


class FakeLearner:
    def __init__(self):
        self.actor_ready = False
        self.submitted = []
        self.candidate = None

    def raise_if_failed(self):
        return None

    def submit(self, trajectory):
        self.submitted.append(trajectory)

    def take_candidate(self):
        candidate, self.candidate = self.candidate, None
        return candidate

    def close(self):
        pass


def _observation(value=0.0):
    return {
        "states": np.full(14, value, dtype=np.float32),
        "main_images": np.zeros((4, 5, 3), dtype=np.uint8),
        "wrist_images": np.zeros((2, 4, 5, 3), dtype=np.uint8),
    }


def _feedback(chunk_id=0, *, done=False, mode="actor"):
    return {
        "rlt_protocol_version": RLT_SSEVAL_PROTOCOL_VERSION,
        "episode_id": "episode-1",
        "chunk_id": chunk_id,
        "selected_mode": mode,
        "executed_actions": np.full((10, 14), 0.5, dtype=np.float32),
        "valid_step_mask": np.ones(10, dtype=bool),
        "rewards": np.zeros(10, dtype=np.float32),
        "terminated": np.array([False] * 9 + [done]),
        "truncated": np.zeros(10, dtype=bool),
        "intervene_flags": np.full(10, mode == "human", dtype=bool),
        "rlt_switch_flags": np.ones(10, dtype=bool),
        "actor_version": 0,
        "timestamps": {},
    }


def _runtime():
    policy = RLTTD3MLPPolicy(
        z_dim=8, proprio_dim=14, action_dim=14, num_action_chunks=10
    )
    learner = FakeLearner()
    runtime = SsEvalRLTRuntime(
        feature_model=FakeFeatureModel(),
        active_policy_model=policy,
        action_codec=SonglingActionCodec([-1.0] * 14, [1.0] * 14),
        learner=learner,
    )
    return runtime, learner


def test_runtime_emits_candidates_and_ingests_executed_feedback():
    runtime, learner = _runtime()
    runtime.start_episode("episode-1", "fold clothes")
    first = runtime.infer_candidates(_observation(0.0))
    second = runtime.infer_candidates(_observation(1.0), _feedback(0))

    assert first.episode_id == "episode-1"
    assert first.chunk_id == 0
    assert second.chunk_id == 1
    assert first.vla_action.shape == (10, 14)
    assert first.actor_action.shape == (10, 14)
    assert len(learner.submitted) == 1
    trajectory = learner.submitted[0]
    torch.testing.assert_close(
        trajectory.actions.reshape(10, 14), torch.full((10, 14), 0.5)
    )
    assert trajectory.forward_inputs["actor_switch"].all()


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
    assert learner.take_candidate() is not None
    learner.close()


def test_terminal_feedback_closes_runtime_episode():
    runtime, learner = _runtime()
    runtime.start_episode("episode-1", "fold clothes")
    runtime.infer_candidates(_observation())
    runtime.infer_candidates(_observation(1.0), _feedback(0, done=True))
    assert len(learner.submitted) == 1
    with pytest.raises(RuntimeError, match="start_episode"):
        runtime.infer_candidates(_observation(2.0))


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
