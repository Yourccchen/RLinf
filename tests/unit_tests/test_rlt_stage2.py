import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.rlt.rollout import (
    validate_online_transition_stride,
    validate_rlt_stage2_configs,
)
from rlinf.algorithms.rlt.route import RLTRouteContext, RealworldRLTRoute
from rlinf.algorithms.rlt.transition import overwrite_rlt_ref_with_human
from rlinf.data.schema.embodied_types import Trajectory
from rlinf.data.storage.replay.buffer import TrajectoryReplayBuffer
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.mlp_policy.rlt_td3_mlp_policy import RLTTD3MLPPolicy
from rlinf.models.embodiment.openpi_rlinf.eval_action_model import (
    OpenPiPytorchEvalActionModel,
)
from rlinf.serving.sseval_contract import CHUNK_LEN
from rlinf.workers.actor.fsdp_rlt_td3_policy_worker import RLTTD3LossMixin
from rlinf.workers.actor.fsdp_sac_policy_worker import should_update_actor
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


def _route_context(*, version: int) -> RLTRouteContext:
    batch_size, chunk_len, action_dim = 2, 10, 14
    ref_chunk = torch.zeros(batch_size, chunk_len, action_dim)
    student_actions = torch.ones_like(ref_chunk)
    return RLTRouteContext(
        env_obs={},
        rlt_obs={
            "z_rl": torch.zeros(batch_size, 2048),
            "proprio": torch.zeros(batch_size, 14),
            "ref_chunk": ref_chunk,
        },
        student_actions=student_actions,
        result={
            "forward_inputs": {
                "ref_chunk": ref_chunk,
                "action": student_actions.reshape(batch_size, -1),
            }
        },
        mode="train",
        rlt_switch_flags=torch.ones(batch_size, 1, dtype=torch.bool),
        version=version,
    )


def test_overwrite_rlt_ref_with_human_replaces_intervened_steps_only():
    ref = torch.zeros(2, 4, 3)
    actions = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3)
    flags = torch.tensor(
        [[True, True, False, False], [False, True, True, False]],
        dtype=torch.bool,
    )

    out = overwrite_rlt_ref_with_human(ref, actions.reshape(2, -1), flags)

    torch.testing.assert_close(out[0, :2], actions[0, :2])
    torch.testing.assert_close(out[0, 2:], torch.zeros(2, 3))
    torch.testing.assert_close(out[1, 1:3], actions[1, 1:3])
    torch.testing.assert_close(out[1, 0], torch.zeros(3))
    torch.testing.assert_close(out[1, 3], torch.zeros(3))
    assert overwrite_rlt_ref_with_human(ref, actions, None) is ref
    torch.testing.assert_close(
        overwrite_rlt_ref_with_human(ref, actions, torch.zeros_like(flags)),
        ref,
    )


def test_online_transition_stride_requires_one_post_chunk_transition():
    validate_online_transition_stride(10, chunk_len=10)
    with pytest.raises(ValueError, match="post-chunk state"):
        validate_online_transition_stride(2, chunk_len=10)


def test_td3_rollout_sync_excludes_critic_parameters():
    names = [
        name
        for name in ("actor.mlp.net.0.weight", "q_head.q1.mlp.net.0.weight")
        if RLTTD3LossMixin._is_rollout_parameter(name)
    ]
    assert names == ["actor.mlp.net.0.weight"]


def test_realworld_route_records_critical_warmup_but_uses_reference():
    route = RealworldRLTRoute(use_schedule=True, warmup_updates=5)
    output = route.route(_route_context(version=4))

    assert torch.count_nonzero(output.actions) == 0
    assert output.result["forward_inputs"]["record_transition"].all()
    assert not output.result["forward_inputs"]["actor_switch"].any()


def test_realworld_route_uses_actor_after_warmup():
    route = RealworldRLTRoute(use_schedule=True, warmup_updates=5)
    output = route.route(_route_context(version=5))

    assert torch.all(output.actions == 1)
    assert output.result["forward_inputs"]["record_transition"].all()
    assert output.result["forward_inputs"]["actor_switch"].all()


def test_realworld_route_does_not_record_before_critical_phase():
    route = RealworldRLTRoute(use_schedule=True, warmup_updates=5)
    ctx = _route_context(version=5)
    ctx.rlt_switch_flags = torch.zeros(2, 1, dtype=torch.bool)
    output = route.route(ctx)

    assert torch.count_nonzero(output.actions) == 0
    assert not output.result["forward_inputs"]["record_transition"].any()
    assert not output.result["forward_inputs"]["actor_switch"].any()


def test_songling_td3_shapes_and_reference_horizon_slice():
    policy = RLTTD3MLPPolicy(
        z_dim=2048,
        proprio_dim=14,
        action_dim=14,
        num_action_chunks=10,
        ref_num_action_chunks=50,
        ref_action_dropout=0.5,
    )
    obs = {
        "z_rl": torch.randn(3, 2048),
        "proprio": torch.randn(3, 14),
        "ref_chunk": torch.randn(3, 50, 14),
    }

    actions, _, _ = policy(
        forward_type=ForwardType.SAC,
        obs=copy.deepcopy(obs),
        deterministic=True,
        apply_action_noise=False,
    )
    q_values = policy(
        forward_type=ForwardType.SAC_Q,
        obs=obs,
        actions=actions,
    )

    assert actions.shape == (3, 140)
    assert q_values.shape == (3, 2)
    assert torch.all(actions >= -1.0)
    assert torch.all(actions <= 1.0)


def test_residual_actor_copies_reference_before_training():
    torch.manual_seed(0)
    policy = RLTTD3MLPPolicy(
        z_dim=8,
        proprio_dim=4,
        action_dim=2,
        num_action_chunks=3,
        actor_residual=True,
        actor_noise_sigma=0.0,
        ref_action_dropout=0.5,
    )
    obs = {
        "z_rl": torch.randn(4, 8),
        "proprio": torch.randn(4, 4),
        "ref_chunk": torch.rand(4, 3, 2) * 1.6 - 0.8,
    }
    expected = obs["ref_chunk"].reshape(4, -1).clamp(-1.0, 1.0)

    actions, _, _ = policy(
        forward_type=ForwardType.SAC,
        obs=copy.deepcopy(obs),
        deterministic=True,
        apply_action_noise=False,
    )
    torch.testing.assert_close(actions, expected)

    dropped, _, _ = policy(
        forward_type=ForwardType.SAC,
        obs=copy.deepcopy(obs),
        deterministic=False,
        apply_action_noise=False,
        apply_reference_dropout=True,
        reference_dropout_prob=1.0,
    )
    torch.testing.assert_close(dropped, expected)


def test_residual_actor_adds_correction_to_reference():
    policy = RLTTD3MLPPolicy(
        z_dim=8,
        proprio_dim=4,
        action_dim=2,
        num_action_chunks=3,
        actor_residual=True,
        actor_noise_sigma=0.0,
    )
    last = None
    for child in policy.actor.mlp.modules():
        if isinstance(child, torch.nn.Linear):
            last = child
    assert last is not None
    last.bias.data.fill_(0.1)

    obs = {
        "z_rl": torch.zeros(2, 8),
        "proprio": torch.zeros(2, 4),
        "ref_chunk": torch.full((2, 3, 2), 0.2),
    }
    actions, _, _ = policy(
        forward_type=ForwardType.SAC,
        obs=obs,
        deterministic=True,
        apply_action_noise=False,
    )
    torch.testing.assert_close(actions, torch.full((2, 6), 0.3))


def test_td3_get_model_wires_actor_residual():
    from rlinf.models.embodiment.mlp_policy import get_model as get_mlp_model

    cfg = OmegaConf.create(
        {
            "model_type": "rlt_td3_mlp_policy",
            "z_dim": 8,
            "proprio_dim": 4,
            "action_dim": 2,
            "num_action_chunks": 3,
            "actor_residual": True,
            "actor_noise_sigma": 0.0,
        }
    )
    model = get_mlp_model(cfg)
    assert model.actor.residual is True


def _songling_eval_wrapper() -> OpenPiPytorchEvalActionModel:
    wrapper = object.__new__(OpenPiPytorchEvalActionModel)
    wrapper.config_name = "pi05_rlt_songling_joint"
    wrapper.state_indices = None
    return wrapper


def test_rlt_reference_sampling_is_reproducible_with_explicit_generator():
    class FakeFlowModel:
        action_horizon = 10
        action_dim = 32

        @staticmethod
        def run_suffix(observation, actions, timestep, kv_cache, prefix_mask):
            return torch.zeros_like(actions)

        @staticmethod
        def velocity_from_suffix(suffix):
            return suffix

    wrapper = _songling_eval_wrapper()
    wrapper.model = FakeFlowModel()
    wrapper.num_steps = 2
    observation = SimpleNamespace(state=torch.zeros(2, 32))
    first_rng = torch.Generator().manual_seed(17)
    second_rng = torch.Generator().manual_seed(17)

    first = wrapper._sample_actions_from_prefix_cache(
        observation, torch.ones(2, 1, dtype=torch.bool), (), rng=first_rng
    )
    second = wrapper._sample_actions_from_prefix_cache(
        observation, torch.ones(2, 1, dtype=torch.bool), (), rng=second_rng
    )

    torch.testing.assert_close(first, second)


def test_rlt_token_only_path_skips_reference_sampler():
    wrapper = _songling_eval_wrapper()
    z_rl = torch.randn(2, 2048)
    proprio = torch.randn(2, 14)
    wrapper._extract_rlt_prefix_features = lambda env_obs: (
        None,
        None,
        None,
        None,
        z_rl,
        proprio,
    )
    wrapper._sample_actions_from_prefix_cache = lambda *args, **kwargs: (
        _ for _ in ()
    ).throw(AssertionError("token-only extraction must not sample reference actions"))

    result = wrapper.extract_rlt_token_obs({})

    assert result == {"z_rl": z_rl, "proprio": proprio}


def test_songling_rlt_proprio_uses_normalized_stage1_state():
    wrapper = _songling_eval_wrapper()
    raw = torch.full((2, 14), 100.0)
    normalized = torch.arange(64, dtype=torch.float32).reshape(2, 32)

    proprio = wrapper._select_rlt_proprio(raw, normalized)

    assert torch.equal(proprio, normalized[:, :14])


def test_songling_repack_preserves_all_three_camera_views():
    wrapper = _songling_eval_wrapper()
    main = torch.zeros(2, 32, 48, 3, dtype=torch.uint8)
    wrists = torch.stack([torch.ones_like(main), torch.full_like(main, 2)], dim=1)

    repacked = wrapper._repack_env_obs(
        {
            "states": torch.zeros(2, 14),
            "main_images": main,
            "wrist_images": wrists,
            "task_descriptions": ["fold clothes", "fold clothes"],
        }
    )

    assert repacked["observation/cam_high"] is main
    assert torch.equal(repacked["observation/cam_left_wrist"], wrists[:, 0])
    assert torch.equal(repacked["observation/cam_right_wrist"], wrists[:, 1])


def test_songling_repack_rejects_missing_wrist_views():
    wrapper = _songling_eval_wrapper()
    with pytest.raises(KeyError, match="require wrist_images"):
        wrapper._repack_env_obs(
            {
                "states": torch.zeros(1, 14),
                "main_images": torch.zeros(1, 32, 48, 3, dtype=torch.uint8),
                "task_descriptions": ["fold clothes"],
            }
        )


def test_songling_stage2_yaml_resolves_json_environment_sequences(monkeypatch):
    values = {
        "EMBODIED_PATH": str(Path(__file__).parents[2] / "examples/embodiment"),
        "SONGLING_ENV_RPC_ENDPOINT": "ws://127.0.0.1:9999",
        "SONGLING_RLT_STAGE1_CHECKPOINT": "/tmp/stage1",
        "SONGLING_ACTION_LOW": "[" + ",".join(["-1"] * 14) + "]",
        "SONGLING_ACTION_HIGH": "[" + ",".join(["1"] * 14) + "]",
        "SONGLING_ACTION_UNITS": "[" + ",".join(['"rad"'] * 14) + "]",
        "SONGLING_ACTION_FREQUENCY_HZ": "50",
        "SONGLING_IMAGE_SHAPE": "[240,320,3]",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    path = (
        Path(__file__).parents[2]
        / "examples/embodiment/config/songling_rlt_stage2_td3_mlp.yaml"
    )
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)

    assert cfg.env.train.action_units == values["SONGLING_ACTION_UNITS"]
    assert cfg.env.train.action_codec.action_low == values["SONGLING_ACTION_LOW"]


def test_stage1_stage2_config_validation_accepts_songling_contract():
    validate_rlt_stage2_configs(
        {
            "z_dim": 2048,
            "proprio_dim": 14,
            "action_dim": 14,
            "num_action_chunks": CHUNK_LEN,
            "ref_num_action_chunks": CHUNK_LEN,
        },
        {
            "action_dim": 14,
            "num_action_chunks": CHUNK_LEN,
            "openpi_data": {"repo_id": "songling/all_tasks"},
            "openpi": {
                "task": "eval",
                "config_name": "pi05_rlt_songling_all",
                "use_rlt": True,
                "rlt_embed_dim": 2048,
                "model_action_dim": 32,
                "num_images_in_input": 3,
                "rlt_image_only": False,
                "rlt_use_mask": True,
                "rlt_prefix_seq_len": 1024,
                "rlt_num_layers": 2,
                "rlt_num_heads": 8,
                "rlt_encoder_type": "append_self_attention",
            },
        },
    )


def test_stage1_stage2_config_validation_accepts_explicit_custom_norm_stats():
    validate_rlt_stage2_configs(
        {
            "z_dim": 2048,
            "proprio_dim": 14,
            "action_dim": 14,
            "num_action_chunks": CHUNK_LEN,
            "ref_num_action_chunks": CHUNK_LEN,
        },
        {
            "action_dim": 14,
            "num_action_chunks": CHUNK_LEN,
            "openpi_data": {
                "repo_id": "songling/bfjm_0915",
                "norm_stats_path": "/models/songling/bfjm_0915/norm_stats.json",
            },
            "openpi": {
                "task": "eval",
                "config_name": "pi05_rlt_songling_joint",
                "use_rlt": True,
                "rlt_embed_dim": 2048,
                "model_action_dim": 32,
                "num_images_in_input": 3,
                "rlt_image_only": False,
                "rlt_use_mask": True,
                "rlt_prefix_seq_len": 1024,
                "rlt_num_layers": 2,
                "rlt_num_heads": 8,
                "rlt_encoder_type": "append_self_attention",
            },
        },
    )


def test_stage1_stage2_config_validation_rejects_custom_repo_without_norm_stats():
    policy = {
        "z_dim": 2048,
        "proprio_dim": 14,
        "action_dim": 14,
        "num_action_chunks": CHUNK_LEN,
        "ref_num_action_chunks": CHUNK_LEN,
    }
    feature = {
        "action_dim": 14,
        "num_action_chunks": CHUNK_LEN,
        "openpi_data": {
            "repo_id": "songling/bfjm_0915",
            "norm_stats_path": None,
        },
        "openpi": {
            "task": "eval",
            "config_name": "pi05_rlt_songling_joint",
            "use_rlt": True,
            "rlt_embed_dim": 2048,
            "model_action_dim": 32,
            "num_images_in_input": 3,
            "rlt_image_only": False,
            "rlt_use_mask": True,
            "rlt_prefix_seq_len": 1024,
            "rlt_num_layers": 2,
            "rlt_num_heads": 8,
            "rlt_encoder_type": "append_self_attention",
        },
    }

    with pytest.raises(ValueError, match="norm_stats_path"):
        validate_rlt_stage2_configs(policy, feature)


def test_stage1_stage2_config_validation_rejects_songling_prefix_semantics():
    policy = {
        "z_dim": 2048,
        "proprio_dim": 14,
        "action_dim": 14,
        "num_action_chunks": CHUNK_LEN,
        "ref_num_action_chunks": CHUNK_LEN,
    }
    feature = {
        "action_dim": 14,
        "num_action_chunks": CHUNK_LEN,
        "openpi_data": {"repo_id": "songling/all_tasks"},
        "openpi": {
            "task": "eval",
            "config_name": "pi05_rlt_songling_all",
            "use_rlt": True,
            "rlt_embed_dim": 2048,
            "model_action_dim": 32,
            "num_images_in_input": 3,
            "rlt_image_only": True,
            "rlt_use_mask": True,
            "rlt_prefix_seq_len": 1024,
            "rlt_num_layers": 2,
            "rlt_num_heads": 8,
            "rlt_encoder_type": "append_self_attention",
        },
    }

    with pytest.raises(ValueError, match="rlt_image_only"):
        validate_rlt_stage2_configs(policy, feature)


def test_songling_train_config_action_horizon_matches_execute_chunk():
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    joint = get_openpi_config("pi05_rlt_songling_joint")
    all_tasks = get_openpi_config("pi05_rlt_songling_all")
    assert joint.model.action_horizon == CHUNK_LEN
    assert all_tasks.model.action_horizon == CHUNK_LEN
    overridden = get_openpi_config("pi05_rlt_songling_joint", action_horizon=10)
    assert overridden.model.action_horizon == 10
    assert (
        get_openpi_config("pi05_rlt_songling_joint").model.action_horizon == CHUNK_LEN
    )


def test_stage1_stage2_config_validation_accepts_matching_custom_chunk_len():
    validate_rlt_stage2_configs(
        {
            "z_dim": 2048,
            "proprio_dim": 14,
            "action_dim": 14,
            "num_action_chunks": 10,
        },
        {
            "action_dim": 14,
            "num_action_chunks": 10,
            "openpi_data": {"repo_id": "songling/all_tasks"},
            "openpi": {
                "task": "eval",
                "config_name": "pi05_rlt_songling_all",
                "use_rlt": True,
                "rlt_embed_dim": 2048,
                "model_action_dim": 32,
                "num_images_in_input": 3,
                "rlt_image_only": False,
                "rlt_use_mask": True,
                "rlt_prefix_seq_len": 1024,
                "rlt_num_layers": 2,
                "rlt_num_heads": 8,
                "rlt_encoder_type": "append_self_attention",
            },
        },
    )


def test_stage1_stage2_config_validation_accepts_shorter_stage2_horizon():
    validate_rlt_stage2_configs(
        {
            "z_dim": 2048,
            "proprio_dim": 14,
            "action_dim": 14,
            "num_action_chunks": 10,
        },
        {
            "action_dim": 14,
            "num_action_chunks": CHUNK_LEN,
            "openpi_data": {"repo_id": "songling/all_tasks"},
            "openpi": {
                "task": "eval",
                "config_name": "pi05_rlt_songling_all",
                "use_rlt": True,
                "rlt_embed_dim": 2048,
                "action_horizon": CHUNK_LEN,
                "action_chunk": CHUNK_LEN,
                "model_action_dim": 32,
                "num_images_in_input": 3,
                "rlt_image_only": False,
                "rlt_use_mask": True,
                "rlt_prefix_seq_len": 1024,
                "rlt_num_layers": 2,
                "rlt_num_heads": 8,
                "rlt_encoder_type": "append_self_attention",
            },
        },
    )


def test_stage1_stage2_config_validation_rejects_reference_over_stage1():
    policy = {
        "z_dim": 2048,
        "proprio_dim": 14,
        "action_dim": 14,
        "num_action_chunks": CHUNK_LEN + 1,
        "ref_num_action_chunks": CHUNK_LEN + 1,
    }
    feature = {
        "action_dim": 14,
        "num_action_chunks": CHUNK_LEN,
        "openpi_data": {"repo_id": "songling/all_tasks"},
        "openpi": {
            "task": "eval",
            "config_name": "pi05_rlt_songling_all",
            "use_rlt": True,
            "rlt_embed_dim": 2048,
            "action_horizon": CHUNK_LEN,
            "action_chunk": CHUNK_LEN,
            "model_action_dim": 32,
            "num_images_in_input": 3,
            "rlt_image_only": False,
            "rlt_use_mask": True,
            "rlt_prefix_seq_len": 1024,
            "rlt_num_layers": 2,
            "rlt_num_heads": 8,
            "rlt_encoder_type": "append_self_attention",
        },
    }

    with pytest.raises(ValueError, match="must not exceed"):
        validate_rlt_stage2_configs(policy, feature)


def test_stage1_stage2_config_validation_reports_horizon_mismatch():
    with pytest.raises(ValueError, match="Stage2 reference horizon"):
        validate_rlt_stage2_configs(
            {
                "z_dim": 2048,
                "proprio_dim": 14,
                "action_dim": 14,
                "num_action_chunks": 10,
                "ref_num_action_chunks": 50,
            },
            {
                "action_dim": 14,
                "num_action_chunks": 10,
                "openpi_data": {"repo_id": "songling/all_tasks"},
                "openpi": {
                    "task": "eval",
                    "config_name": "pi05_rlt_songling_all",
                    "use_rlt": True,
                    "rlt_embed_dim": 2048,
                    "model_action_dim": 32,
                    "num_images_in_input": 3,
                    "rlt_image_only": False,
                    "rlt_use_mask": True,
                    "rlt_prefix_seq_len": 1024,
                    "rlt_num_layers": 2,
                    "rlt_num_heads": 8,
                    "rlt_encoder_type": "append_self_attention",
                },
            },
        )


def test_short_chunk_masks_rewards_and_disables_bootstrap():
    loss = object.__new__(RLTTD3LossMixin)
    loss.cfg = SimpleNamespace(algorithm=SimpleNamespace(gamma=0.5))
    loss.torch_dtype = torch.float32
    rewards = torch.tensor([[1.0, 2.0, 100.0, 100.0]])
    valid = torch.tensor([[True, True, False, False]])

    reward_target = loss._discounted_chunk_rewards(rewards, valid)
    discount, has_full_horizon = loss._chunk_bootstrap_discount(
        valid, dtype=torch.float32
    )

    torch.testing.assert_close(reward_target, torch.tensor([[2.0]]))
    torch.testing.assert_close(discount, torch.zeros_like(discount))
    assert not has_full_horizon.any()


def test_td3_target_action_uses_current_actor_not_target_actor():
    loss = object.__new__(RLTTD3LossMixin)
    calls = []

    def current_actor(**kwargs):
        calls.append(kwargs)
        return torch.ones(2, 140), torch.zeros(2, 140), None

    loss.model = current_actor
    loss.target_model = lambda **kwargs: (_ for _ in ()).throw(
        AssertionError("target actor must not be called")
    )
    loss.cfg = SimpleNamespace(algorithm={"target_action_noise": False})

    actions, _, _ = loss._next_actions_for_critic_target({"z_rl": torch.zeros(2, 2048)})

    assert actions.shape == (2, 140)
    assert len(calls) == 1
    assert calls[0]["apply_action_noise"] is False


def test_replay_buffer_evicts_oldest_trajectory_at_sample_limit():
    replay = TrajectoryReplayBuffer(
        enable_cache=True,
        sample_window_size=2,
        max_num_samples=3,
        auto_save=False,
    )
    try:
        replay.add_trajectories(
            [Trajectory(rewards=torch.zeros(2, 1), model_weights_id="first")]
        )
        replay.add_trajectories(
            [Trajectory(rewards=torch.zeros(2, 1), model_weights_id="second")]
        )

        assert replay.total_samples == 2
        assert replay.size == 1
        assert replay._trajectory_id_list == [1]
    finally:
        replay.close()


class _ClipProbeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Parameter(torch.tensor([1.0]))
        self.critic = torch.nn.Parameter(torch.tensor([1.0]))
        self.visible_grad_ids = set()

    def clip_grad_norm_(self, max_norm):
        self.visible_grad_ids = {
            id(param) for param in self.parameters() if param.grad is not None
        }
        return torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)


def test_td3_gradient_clipping_uses_fsdp_model_for_one_optimizer_group():
    loss = object.__new__(RLTTD3LossMixin)
    loss.model = _ClipProbeModel()
    loss.device = torch.device("cpu")
    actor_optimizer = torch.optim.SGD([loss.model.actor], lr=0.1)
    loss.model.actor.grad = torch.tensor([4.0])
    loss.model.critic.grad = torch.tensor([3.0])

    norm = loss._clip_optimizer_grad_norm(actor_optimizer, max_norm=2.0)

    torch.testing.assert_close(norm, torch.tensor(4.0))
    assert loss.model.visible_grad_ids == {id(loss.model.actor)}
    torch.testing.assert_close(loss.model.actor.grad, torch.tensor([2.0]))
    torch.testing.assert_close(loss.model.critic.grad, torch.tensor([3.0]))


def test_candidate_actor_activates_only_at_episode_boundary():
    worker = object.__new__(MultiStepRolloutWorker)
    worker.defer_rlt_actor_activation = True
    worker.hf_model = RLTTD3MLPPolicy(
        z_dim=8, proprio_dim=14, action_dim=14, num_action_chunks=10
    )
    worker._candidate_hf_model = copy.deepcopy(worker.hf_model)
    with torch.no_grad():
        for parameter in worker.hf_model.actor.parameters():
            parameter.zero_()
        for parameter in worker._candidate_hf_model.actor.parameters():
            parameter.fill_(0.25)
    worker._candidate_version = 9
    worker._rollout_started = True
    worker.version = 4

    worker._activate_candidate_at_boundary({"episode_boundary": torch.tensor([False])})
    assert worker.version == 4
    assert worker._candidate_version == 9
    assert all(
        torch.count_nonzero(parameter) == 0
        for parameter in worker.hf_model.actor.parameters()
    )

    worker._activate_candidate_at_boundary({"episode_boundary": torch.tensor([True])})
    assert worker.version == 9
    assert worker._candidate_version is None
    assert all(
        torch.all(parameter == 0.25) for parameter in worker.hf_model.actor.parameters()
    )


def test_rlt_rollout_payload_marks_reset_and_completed_episode_boundaries():
    from rlinf.workers.env.env_worker import EnvWorker

    worker = object.__new__(EnvWorker)
    worker.enable_rlt = True
    batch = {
        "obs": {"states": torch.zeros(2, 14)},
        "final_obs": None,
        "dones": torch.tensor([[False, False], [False, True]]),
        "rlt_switch_flags": torch.ones(2, 1, dtype=torch.bool),
        "intervene_flags": None,
    }

    regular = worker._build_rollout_input_data(batch)
    reset = worker._build_rollout_input_data(batch, force_episode_boundary=True)

    assert regular["episode_boundary"].tolist() == [False, True]
    assert reset["episode_boundary"].tolist() == [True, True]


def test_td3_actor_updates_after_every_second_critic_step():
    assert [should_update_actor(step, 2) for step in range(6)] == [
        False,
        True,
        False,
        True,
        False,
        True,
    ]
