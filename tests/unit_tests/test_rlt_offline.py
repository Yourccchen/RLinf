import copy

import torch
import torch.nn.functional as F

from rlinf.data.rlt import build_offline_rlt_trajectories
from rlinf.data.storage.replay import TrajectoryReplayBuffer
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.mlp_policy.rlt_td3_mlp_policy import RLTTD3MLPPolicy


def _features(length: int):
    return [
        {
            "z_rl": torch.full((8,), float(i)),
            "proprio": torch.full((14,), float(i)),
            "ref_chunk": torch.full((10, 14), float(i)),
        }
        for i in range(length)
    ]


def _episode(length: int = 12):
    terminated = torch.zeros(length, dtype=torch.bool)
    terminated[-1] = True
    return {
        "executed_actions": torch.arange(length * 14, dtype=torch.float32).reshape(
            length, 14
        ),
        "rewards": torch.cat([torch.zeros(length - 1), torch.ones(1)]),
        "terminated": terminated,
        "truncated": torch.zeros(length, dtype=torch.bool),
    }


def test_offline_builder_constructs_overlapping_executed_action_windows():
    trajectories = build_offline_rlt_trajectories(
        _episode(), _features(13), chunk_len=10, transition_stride=2
    )

    assert len(trajectories) == 2
    first, terminal = trajectories
    assert first.actions.shape == (1, 1, 140)
    assert first.forward_inputs["valid_step_mask"].all()
    assert torch.equal(first.next_obs["z_rl"].reshape(-1), torch.full((8,), 10.0))
    assert terminal.dones.any()
    assert terminal.forward_inputs["valid_step_mask"].all()
    expected = _episode()["executed_actions"][2:12].reshape(-1)
    torch.testing.assert_close(terminal.actions.reshape(-1), expected)


def test_offline_builder_pads_terminal_short_chunk_and_replay_roundtrips(tmp_path):
    episode = _episode(length=5)
    trajectories = build_offline_rlt_trajectories(
        episode, _features(6), chunk_len=10, transition_stride=2
    )
    assert len(trajectories) == 1
    valid = trajectories[0].forward_inputs["valid_step_mask"].reshape(-1)
    assert valid.tolist() == [True] * 5 + [False] * 5

    replay = TrajectoryReplayBuffer(enable_cache=True, sample_window_size=1)
    replay.add_trajectories(trajectories)
    checkpoint = tmp_path / "replay"
    replay.save_checkpoint(str(checkpoint))
    replay.close()

    restored = TrajectoryReplayBuffer(enable_cache=True, sample_window_size=1)
    restored.load_checkpoint(str(checkpoint))
    batch = restored.sample(1)
    restored.close()
    assert batch["actions"].shape == (1, 140)
    assert batch["forward_inputs"]["valid_step_mask"].shape == (1, 10)


def test_offline_builder_rejects_unlabeled_demonstrations():
    episode = _episode()
    episode["terminated"].zero_()

    try:
        build_offline_rlt_trajectories(episode, _features(13))
    except ValueError as exc:
        assert "terminal labels" in str(exc)
    else:
        raise AssertionError("unlabeled demonstration was accepted for TD3")


def test_offline_replay_batch_drives_one_td3_optimizer_step():
    torch.manual_seed(7)
    trajectory = build_offline_rlt_trajectories(
        _episode(length=10), _features(11), chunk_len=10, transition_stride=2
    )[0]
    replay = TrajectoryReplayBuffer(enable_cache=True, sample_window_size=1)
    replay.add_trajectories([trajectory])
    batch = replay.sample(1)
    replay.close()

    model = RLTTD3MLPPolicy(
        z_dim=8,
        proprio_dim=14,
        action_dim=14,
        num_action_chunks=10,
        actor_noise_sigma=0.0,
    )
    target = copy.deepcopy(model).requires_grad_(False)
    critic_optimizer = torch.optim.Adam(model.q_head.parameters(), lr=1e-3)
    actor_optimizer = torch.optim.Adam(model.actor.parameters(), lr=1e-3)
    actions = batch["actions"].clamp(-1.0, 1.0)

    with torch.no_grad():
        next_actions, _, _ = model(
            forward_type=ForwardType.SAC,
            obs=batch["next_obs"],
            deterministic=True,
        )
        next_q = (
            target(
                forward_type=ForwardType.SAC_Q,
                obs=batch["next_obs"],
                actions=next_actions,
            )
            .min(dim=-1, keepdim=True)
            .values
        )
        reward = batch["rewards"].sum(dim=-1, keepdim=True)
        critic_target = reward + 0.99**10 * next_q

    critic_before = model.q_head.q1.mlp.net[0].weight.detach().clone()
    q_values = model(
        forward_type=ForwardType.SAC_Q, obs=batch["curr_obs"], actions=actions
    )
    critic_loss = F.mse_loss(q_values, critic_target.expand_as(q_values))
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    actor_before = torch.cat(
        [parameter.detach().reshape(-1) for parameter in model.actor.parameters()]
    ).clone()
    predicted, _, _ = model(
        forward_type=ForwardType.SAC,
        obs=batch["curr_obs"],
        deterministic=True,
    )
    q1 = model(
        forward_type=ForwardType.SAC_Q,
        obs=batch["curr_obs"],
        actions=predicted,
    )[:, :1]
    actor_loss = -0.05 * q1.mean() + F.mse_loss(predicted, actions)
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    assert torch.isfinite(critic_loss)
    assert torch.isfinite(actor_loss)
    assert not torch.equal(critic_before, model.q_head.q1.mlp.net[0].weight)
    actor_after = torch.cat(
        [parameter.detach().reshape(-1) for parameter in model.actor.parameters()]
    )
    assert not torch.equal(actor_before, actor_after)


def test_replay_checkpoint_restores_next_sampling_sequence(tmp_path):
    replay = TrajectoryReplayBuffer(seed=31, enable_cache=True, sample_window_size=3)
    trajectories = []
    for index in range(3):
        trajectory = build_offline_rlt_trajectories(
            _episode(length=10),
            _features(11),
            chunk_len=10,
            transition_stride=10,
            model_weights_id=f"episode_{index}",
        )[0]
        trajectory.rewards.fill_(float(index))
        trajectories.append(trajectory)
    replay.add_trajectories(trajectories)
    replay.sample(2)
    checkpoint = tmp_path / "rng_replay"
    replay.save_checkpoint(str(checkpoint))
    expected = replay.sample(8)["rewards"]
    replay.close()

    restored = TrajectoryReplayBuffer(seed=999, enable_cache=True, sample_window_size=3)
    restored.load_checkpoint(str(checkpoint))
    actual = restored.sample(8)["rewards"]
    restored.close()

    torch.testing.assert_close(actual, expected)
