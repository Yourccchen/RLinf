"""HALO-WA model and online learner contracts."""

import copy

import pytest
import torch
from omegaconf import OmegaConf


def _model():
    from rlinf.models import get_model

    cfg = OmegaConf.create(
        dict(
            model_type="halo_wa_policy",
            precision="fp32",
            visual_dim=3,
            hidden_dim=16,
            num_heads=2,
            max_visual_tokens=8,
            proprio_dim=14,
            action_dim=14,
            num_action_chunks=2,
            load_to_device=False,
        )
    )
    model = get_model(cfg)
    assert model is not None, "halo_wa_policy must be registered independently"
    return model


def _obs(batch=2):
    return dict(
        z_rl=torch.randn(batch, 5, 3),
        proprio=torch.randn(batch, 14),
        ref_chunk=torch.randn(batch, 2, 14).clamp(-1, 1),
    )


def _trajectory():
    from test_sseval_rlt_runtime import _offline_rlt_trajectory

    t = _offline_rlt_trajectory(chunk_len=2)
    for obs in (t.curr_obs, t.next_obs):
        obs["z_rl"] = torch.randn(1, 1, 5, 3)
        obs["ref_chunk"] = torch.full((1, 1, 2, 14), 0.2)
    t.actions.fill_(-0.3)
    t.intervene_flags.fill_(True)
    return t


def _learner(**overrides):
    from rlinf.serving.halo_wa_learner import HaloWALearner

    return HaloWALearner(
        _model(),
        dict(
            batch_size=4,
            min_buffer_size=1,
            utd=1,
            replay_capacity=16,
            bc_warmup_steps=2,
            log_interval=0,
            **overrides,
        ),
        feature_identity="test-vae",
        start_background=False,
    )


def test_attention_model_uses_visual_tokens_and_has_independent_parameters():
    model = _model()
    obs = _obs()
    a = model.predict_action_batch(obs, mode="eval")[0]
    b = model.predict_action_batch(
        {**obs, "z_rl": obs["z_rl"] + torch.randn_like(obs["z_rl"])}, mode="eval"
    )[0]
    assert a.shape == (2, 2, 14) and not torch.equal(a, b)
    assert not (
        {id(p) for p in model.actor.parameters()}
        & {id(p) for p in model.q_head.parameters()}
    )
    from rlinf.models.embodiment.base_policy import ForwardType

    prediction = model(forward_type=ForwardType.SAC, obs=obs)[0]
    q = model(forward_type=ForwardType.SAC_Q, obs=obs, actions=prediction)
    assert q.shape == (2, 2)
    (q.sum() + prediction.square().sum()).backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    with pytest.raises(ValueError, match="visual"):
        model.predict_action_batch({**obs, "z_rl": torch.zeros(2, 9, 3)})


def test_halo_warmup_then_balanced_online_updates_preserve_reference():
    learner = _learner()
    demo = _trajectory()
    ref = demo.curr_obs["ref_chunk"].clone()
    learner.add_demo_trajectories([demo])
    learner.replay.add_trajectories([_trajectory()])
    critic_before = copy.deepcopy(learner.model.q_head.state_dict())
    assert not learner.actor_ready
    for _ in range(2):
        metrics = learner.train_once()
        assert metrics["stage"] == "bc"
        assert metrics["online_samples"] == 0
    assert learner.actor_ready
    assert all(
        torch.equal(v, learner.model.q_head.state_dict()[k])
        for k, v in critic_before.items()
    )
    assert torch.equal(demo.curr_obs["ref_chunk"], ref)
    for _ in range(2):
        metrics = learner.train_once()
    assert metrics["demo_samples"] == metrics["online_samples"] == 2
    assert metrics["actor_updated"]
    assert torch.isfinite(torch.tensor(metrics["actor_loss"]))
    learner.close()


def test_halo_checkpoint_restores_sampling_and_rejects_identity(tmp_path):
    torch.manual_seed(321)
    learner = _learner()
    learner.add_demo_trajectories([_trajectory()])
    learner.replay.add_trajectories([_trajectory()])
    for _ in range(4):
        learner.train_once()
    learner.save_checkpoint(tmp_path / "ckpt")
    expected = learner.train_once()
    expected_model = copy.deepcopy(learner.model.state_dict())
    restored = _learner()
    restored.load_checkpoint(tmp_path / "ckpt")
    actual = restored.train_once()
    assert actual == expected
    for k, v in expected_model.items():
        assert torch.equal(v, restored.model.state_dict()[k]), k
    restored.feature_identity = "wrong"
    with pytest.raises(ValueError, match="identity"):
        restored.load_checkpoint(tmp_path / "ckpt")
    learner.close()
    restored.close()


def test_halo_requires_demo_data_and_collection_never_enables_actor():
    learner = _learner()
    with pytest.raises(ValueError, match="demo"):
        learner.train_once()
    learner.close()
    collector = _learner(training_stage="collect")
    collector.add_and_train(_trajectory())
    assert collector.replay.total_samples == 1
    assert collector.update_step == 0 and not collector.actor_ready
    collector.close()


def test_collection_close_saves_usable_demo_checkpoint(tmp_path):
    collector = _learner(training_stage="collect")
    collector.save_dir = tmp_path
    collector.save_interval = 100
    collector.add_and_train(_trajectory())
    collector.close()
    learner = _learner()
    learner.load_demo_checkpoint(tmp_path / "collect_1")
    assert learner.demo.total_samples == 1
    assert learner.train_once()["stage"] == "bc"
    learner.close()


def test_halo_feedback_preserves_wa_reference_for_actor_input():
    from rlinf.serving.sseval_runtime import (
        _PendingInference,
        _build_feedback_trajectory,
    )
    from rlinf.serving.sseval_contract import TransitionFeedback
    from rlinf.envs.remote_songling import SonglingActionCodec
    from test_sseval_rlt_runtime import _feedback

    obs = dict(
        z_rl=torch.ones(1, 5, 3),
        proprio=torch.zeros(1, 14),
        ref_chunk=torch.full((1, 2, 14), 0.2),
    )
    pending = _PendingInference(
        episode_id="episode-1", chunk_id=0, replay_obs=obs, actor_version=0
    )
    fb = TransitionFeedback.from_mapping(_feedback(chunk_len=2, mode="human"))
    codec = SonglingActionCodec([-1.0] * 14, [1.0] * 14)
    t = _build_feedback_trajectory(
        pending, obs, fb, codec, preserve_reference_actions=True
    )
    assert torch.equal(t.curr_obs["ref_chunk"].flatten(), torch.full((28,), 0.2))
    assert torch.equal(t.actions.flatten(), torch.full((28,), 0.5))


def test_runtime_configuration_selects_halo_and_publishes_visual_weights(tmp_path):
    from rlinf.serving.sseval_runtime import SsEvalRLTRuntime
    from test_sseval_rlt_runtime import FakeFeatureModel

    class Provider(FakeFeatureModel):
        preserve_reference_actions = True

        def validate_config(self, actor, codec):
            assert actor.model_type == "halo_wa_policy"
            self.rlt_checkpoint_version = "test-vae"

    collector = _learner(training_stage="collect")
    collector.add_and_train(_trajectory())
    collector.save_checkpoint(tmp_path / "demo")
    collector.close()
    cfg = dict(
        device="cpu",
        learner_type="halo_wa",
        demo_checkpoint=str(tmp_path / "demo"),
        actor_model=dict(
            model_type="halo_wa_policy",
            precision="fp32",
            visual_dim=3,
            hidden_dim=16,
            num_heads=2,
            max_visual_tokens=8,
            proprio_dim=14,
            action_dim=14,
            num_action_chunks=2,
        ),
        action_codec=dict(action_low=[-1.0] * 14, action_high=[1.0] * 14),
        learner=dict(
            batch_size=4,
            min_buffer_size=1,
            utd=1,
            replay_capacity=16,
            bc_warmup_steps=2,
            log_interval=0,
        ),
    )
    path = tmp_path / "runtime.yaml"
    OmegaConf.save(OmegaConf.create(cfg), path)
    runtime = SsEvalRLTRuntime.from_config(path, feature_model=Provider())
    try:
        runtime.learner.add_and_train(_trajectory())
        assert runtime.learner.actor_ready
        runtime.start_episode("next", "pick")
        for key, value in runtime.learner.model.actor.state_dict().items():
            assert torch.equal(
                value, runtime.active_policy_model.actor.state_dict()[key]
            ), key
        with pytest.raises(ValueError, match="replay-only"):
            runtime.learner.load_replay(tmp_path)
    finally:
        runtime.close()
    cfg["replay_checkpoint"] = str(tmp_path / "demo")
    OmegaConf.save(OmegaConf.create(cfg), path)
    with pytest.raises(ValueError, match="complete actor_checkpoint"):
        SsEvalRLTRuntime.from_config(path, feature_model=Provider())


def test_example_resolves_environment_channels(monkeypatch):
    from pathlib import Path
    from rlinf.models import get_model

    monkeypatch.setenv("SOMA_VAE_CHANNELS", "3")
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/embodiment/config/soma_halo_wa_sseval.yaml"
    )
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)
    model = get_model(cfg.actor_model)
    assert model.visual_dim == 3 and model.chunk_len == 12
    assert model.actor.visual_compressor[0].out_features == 512
    assert model.predict_action_batch(
        dict(
            z_rl=torch.zeros(1, 5, 3),
            proprio=torch.zeros(1, 14),
            ref_chunk=torch.zeros(1, 12, 14),
        )
    )[0].shape == (1, 12, 14)


def test_collection_without_periodic_saving_uses_unique_run_directories(tmp_path):
    from rlinf.serving.halo_wa_learner import HaloWALearner

    config = dict(
        training_stage="collect",
        batch_size=4,
        min_buffer_size=1,
        replay_capacity=16,
        log_interval=0,
    )
    paths = []
    for _ in range(2):
        collector = HaloWALearner(
            _model(),
            config,
            feature_identity="test-vae",
            save_dir=tmp_path,
            save_interval=0,
            start_background=False,
        )
        collector.add_and_train(_trajectory())
        paths.append(collector.save_dir / "collect_1")
        collector.close()
    assert paths[0] != paths[1]
    assert all((p / "halo_wa.json").is_file() for p in paths)
