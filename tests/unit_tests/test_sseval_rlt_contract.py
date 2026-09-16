import numpy as np
import pytest

from rlinf.serving.sseval_contract import (
    RLT_SSEVAL_PROTOCOL_VERSION,
    DualActionCandidates,
    SelectedMode,
    TransitionFeedback,
)


def _feedback(mode="actor"):
    valid = np.ones(10, dtype=bool)
    return {
        "rlt_protocol_version": RLT_SSEVAL_PROTOCOL_VERSION,
        "episode_id": "episode-1",
        "chunk_id": 3,
        "selected_mode": mode,
        "executed_actions": np.zeros((10, 14), dtype=np.float32),
        "valid_step_mask": valid,
        "rewards": np.zeros(10, dtype=np.float32),
        "terminated": np.zeros(10, dtype=bool),
        "truncated": np.zeros(10, dtype=bool),
        "intervene_flags": np.full(10, mode == "human", dtype=bool),
        "rlt_switch_flags": np.ones(10, dtype=bool),
        "actor_version": 9,
        "timestamps": {"executed": 1.0},
    }


def test_dual_candidates_emit_backward_compatible_kwargs():
    candidates = DualActionCandidates(
        episode_id="episode-1",
        chunk_id=0,
        vla_action=np.zeros((10, 14)),
        actor_action=np.ones((10, 14)),
        actor_ready=True,
        actor_version=4,
        feature_checkpoint_hash=11,
        reference_seed=17,
    )
    kwargs = candidates.to_action_kwargs()
    assert kwargs["rlt_protocol_version"] == RLT_SSEVAL_PROTOCOL_VERSION
    assert kwargs["episode_id"] == "episode-1"
    assert kwargs["chunk_id"] == 0
    assert kwargs["actor_action"].shape == (10, 14)


def test_feedback_validates_mode_and_terminal_state():
    payload = _feedback("human")
    payload["terminated"][-1] = True
    feedback = TransitionFeedback.from_mapping(payload)
    assert feedback.selected_mode is SelectedMode.HUMAN
    assert feedback.done
    assert feedback.key == ("episode-1", 3)


def test_feedback_rejects_human_actions_without_intervention_flags():
    payload = _feedback("human")
    payload["intervene_flags"][:] = False
    with pytest.raises(ValueError, match="Human-mode"):
        TransitionFeedback.from_mapping(payload)


def test_candidate_shape_is_strict():
    with pytest.raises(ValueError, match="shape"):
        DualActionCandidates(
            episode_id="episode-1",
            chunk_id=0,
            vla_action=np.zeros((9, 14)),
            actor_action=np.zeros((10, 14)),
            actor_ready=False,
            actor_version=0,
            feature_checkpoint_hash=0,
            reference_seed=0,
        )
