import threading

import msgpack
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs.remote_songling.client import (
    SonglingRPCClient,
    SonglingRPCProtocolError,
    SonglingRPCTransportError,
    _from_wire,
    _to_wire,
)
from rlinf.envs.remote_songling.codec import SonglingActionCodec
from rlinf.envs.remote_songling.env import RemoteSonglingEnv


def _observation(value: int = 0) -> dict:
    image = np.full((4, 5, 3), value, dtype=np.uint8)
    return {
        "state": np.linspace(-0.5, 0.5, 14, dtype=np.float32),
        "head_camera": image,
        "left_camera": image + 1,
        "right_camera": image + 2,
        "instruction": "fold clothes",
        "freshness": {
            "state": True,
            "head_camera": True,
            "left_camera": True,
            "right_camera": True,
        },
        "timestamps": {
            "state": 1.0,
            "head_camera": 1.0,
            "left_camera": 1.0,
            "right_camera": 1.0,
        },
    }


class _FakeClient:
    def __init__(self, result=None, error=None):
        self.result = result
        self.error = error
        self.calls = []

    def call(self, method, params=None):
        self.calls.append((method, params))
        if self.error is not None:
            raise self.error
        return self.result


class _FakeConnection:
    def __init__(self):
        self.sent = []
        self.closed = False

    def send(self, payload):
        self.sent.append(payload)

    def recv(self, timeout=None):
        import msgpack

        request = msgpack.unpackb(self.sent[-1], raw=False)
        return msgpack.packb(
            {
                "protocol_version": request["protocol_version"],
                "request_id": request["request_id"],
                "ok": True,
                "result": {"echo": request["params"]},
            },
            use_bin_type=True,
        )

    def close(self):
        self.closed = True


class _MismatchedConnection(_FakeConnection):
    def recv(self, timeout=None):
        import msgpack

        request = msgpack.unpackb(self.sent[-1], raw=False)
        return msgpack.packb(
            {
                "protocol_version": request["protocol_version"],
                "request_id": "wrong-request",
                "ok": True,
                "result": {},
            },
            use_bin_type=True,
        )


def _env(client) -> RemoteSonglingEnv:
    env = object.__new__(RemoteSonglingEnv)
    env.client = client
    env.codec = SonglingActionCodec([-1.0] * 14, [1.0] * 14)
    env.max_chunk_len = 10
    env.action_frequency_hz = 50.0
    env.default_critical_phase = True
    env.task_description = "fold clothes"
    env.image_shape = (4, 5, 3)
    env.camera_keys = {
        "head": "head_camera",
        "left": "left_camera",
        "right": "right_camera",
    }
    env.cfg = {}
    env._session_id = "session"
    env._episode_id = "episode"
    env._chunk_id = 0
    env._policy_version = "7"
    env._elapsed_steps = np.zeros(1, dtype=np.int32)
    env._last_obs = env._observation(_observation())
    return env


def test_songling_action_codec_round_trip_numpy_and_torch():
    codec = SonglingActionCodec([-2.0] * 14, [4.0] * 14)
    normalized = np.linspace(-1.0, 1.0, 28, dtype=np.float32).reshape(2, 14)
    physical = codec.decode(normalized)
    np.testing.assert_allclose(codec.encode(physical), normalized, atol=1e-6)

    normalized_torch = torch.from_numpy(normalized)
    physical_torch = codec.decode(normalized_torch)
    torch.testing.assert_close(codec.encode(physical_torch), normalized_torch)


def test_songling_action_codec_decodes_json_environment_limits():
    codec = SonglingActionCodec.from_config(
        {
            "action_low": "[" + ",".join(["-2"] * 14) + "]",
            "action_high": "[" + ",".join(["4"] * 14) + "]",
        }
    )

    np.testing.assert_array_equal(codec.low, np.full(14, -2.0, dtype=np.float32))
    np.testing.assert_array_equal(codec.high, np.full(14, 4.0, dtype=np.float32))

    list_config_codec = SonglingActionCodec.from_config(
        OmegaConf.create({"action_low": [-2.0] * 14, "action_high": [4.0] * 14})
    )
    np.testing.assert_array_equal(list_config_codec.low, codec.low)
    np.testing.assert_array_equal(list_config_codec.high, codec.high)


def test_songling_action_codec_rejects_broadcast_and_nonfinite_actions():
    codec = SonglingActionCodec([-1.0] * 14, [1.0] * 14)
    with pytest.raises(ValueError, match="last dimension 14"):
        codec.decode(np.zeros(1, dtype=np.float32))
    invalid = torch.zeros(2, 14)
    invalid[0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        codec.encode(invalid)


def test_songling_wire_format_round_trip_arrays():
    payload = {
        "image": np.arange(24, dtype=np.uint8).reshape(2, 3, 4),
        "actions": np.arange(28, dtype=np.float32).reshape(2, 14),
    }
    decoded = _from_wire(_to_wire(payload))
    np.testing.assert_array_equal(decoded["image"], payload["image"])
    np.testing.assert_array_equal(decoded["actions"], payload["actions"])


def test_rpc_client_frames_binary_request_and_matches_response_id():
    connection = _FakeConnection()
    client = SonglingRPCClient("ws://unused", protocol_version="songling-env-v1")
    client._connection = connection

    actions = np.arange(28, dtype=np.float32).reshape(2, 14)
    result = client.call("chunk_step", {"actions": actions})

    assert len(connection.sent) == 1
    np.testing.assert_array_equal(result["echo"]["actions"], actions)


def test_rpc_client_drops_connection_after_response_id_mismatch():
    connection = _MismatchedConnection()
    client = SonglingRPCClient("ws://unused", protocol_version="songling-env-v1")
    client._connection = connection

    with pytest.raises(SonglingRPCProtocolError, match="request_id mismatch"):
        client.call("health")

    assert connection.closed
    assert client._connection is None


def test_remote_songling_short_chunk_pads_and_marks_valid_prefix():
    observations = [_observation(3), _observation(4)]
    result = {
        "observations": observations,
        "executed_actions": np.stack(
            [np.full(14, 0.25, dtype=np.float32), np.full(14, 0.5, dtype=np.float32)]
        ),
        "rewards": np.array([0.0, 1.0], dtype=np.float32),
        "terminated": np.array([False, True]),
        "truncated": np.array([False, False]),
        "valid_step_mask": np.array([True, True]),
        "intervene_flags": np.array([False, False]),
        "outcome": "success",
        "final_observation": observations[-1],
    }
    env = _env(_FakeClient(result=result))

    obs, rewards, terminated, truncated, infos = env.chunk_step(
        np.zeros((1, 4, 14), dtype=np.float32)
    )

    assert len(obs) == 4
    assert rewards.shape == (1, 4)
    assert rewards.tolist() == [[0.0, 1.0, 0.0, 0.0]]
    assert terminated.tolist() == [[False, True, False, False]]
    assert not truncated.any()
    assert infos[-1]["valid_step_mask"].tolist() == [[True, True, False, False]]
    assert infos[-1]["executed_actions"].shape == (1, 56)
    assert env.elapsed_steps.tolist() == [2]


def test_remote_songling_transport_failure_is_non_retriable_truncation():
    client = _FakeClient(error=SonglingRPCTransportError("connection lost"))
    env = _env(client)

    _, _, terminated, truncated, infos = env.chunk_step(
        np.zeros((1, 3, 14), dtype=np.float32)
    )

    assert len(client.calls) == 1
    assert not terminated.any()
    assert truncated.tolist() == [[True, False, False]]
    assert not infos[-1]["valid_step_mask"].any()


def test_rpc_client_round_trips_against_local_websocket_server():
    from websockets.sync.server import serve

    def handler(connection):
        request = msgpack.unpackb(connection.recv(), raw=False)
        connection.send(
            msgpack.packb(
                {
                    "protocol_version": request["protocol_version"],
                    "request_id": request["request_id"],
                    "ok": True,
                    "result": {
                        "method": request["method"],
                        "params": request["params"],
                    },
                },
                use_bin_type=True,
            )
        )

    with serve(handler, "127.0.0.1", 0) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        host, port = server.socket.getsockname()[:2]
        client = SonglingRPCClient(
            f"ws://{host}:{port}", protocol_version="songling-env-v1"
        )
        try:
            actions = np.arange(28, dtype=np.float32).reshape(2, 14)
            result = client.call("chunk_step", {"actions": actions})
            assert result["method"] == "chunk_step"
            np.testing.assert_array_equal(result["params"]["actions"], actions)
        finally:
            client.close()
            server.shutdown()
            thread.join(timeout=2)
