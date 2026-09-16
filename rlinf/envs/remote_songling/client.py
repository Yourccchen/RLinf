# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import threading
import uuid
from typing import Any

import numpy as np


class SonglingRPCError(RuntimeError):
    pass


class SonglingRPCProtocolError(SonglingRPCError):
    pass


class SonglingRPCTransportError(SonglingRPCError):
    pass


def _to_wire(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "__ndarray__": True,
            "dtype": array.dtype.str,
            "shape": list(array.shape),
            "data": array.tobytes(),
        }
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_wire(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_wire(item) for item in value]
    return value


def _from_wire(value: Any) -> Any:
    if isinstance(value, dict) and value.get("__ndarray__") is True:
        try:
            array = np.frombuffer(value["data"], dtype=np.dtype(value["dtype"]))
            return array.reshape(tuple(int(dim) for dim in value["shape"])).copy()
        except (KeyError, TypeError, ValueError) as exc:
            raise SonglingRPCProtocolError(f"Invalid ndarray payload: {exc}") from exc
    if isinstance(value, dict):
        return {key: _from_wire(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_wire(item) for item in value]
    return value


class SonglingRPCClient:
    """Persistent MessagePack-over-WebSocket client with no action retries."""

    def __init__(
        self,
        endpoint: str,
        *,
        protocol_version: str,
        connect_timeout_s: float = 10.0,
        request_timeout_s: float = 30.0,
        max_message_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.endpoint = str(endpoint)
        self.protocol_version = str(protocol_version)
        self.connect_timeout_s = float(connect_timeout_s)
        self.request_timeout_s = float(request_timeout_s)
        self.max_message_bytes = int(max_message_bytes)
        if self.max_message_bytes <= 0:
            raise ValueError("max_message_bytes must be positive.")
        self._connection = None
        self._lock = threading.Lock()

    def _connect(self):
        if self._connection is not None:
            return self._connection
        try:
            import msgpack
            from websockets.sync.client import connect
        except ImportError as exc:
            raise RuntimeError(
                "RemoteSonglingEnv requires the 'msgpack' and 'websockets' packages."
            ) from exc
        del msgpack
        try:
            self._connection = connect(
                self.endpoint,
                open_timeout=self.connect_timeout_s,
                close_timeout=self.connect_timeout_s,
                max_size=self.max_message_bytes,
                legacy=True,
            )
        except Exception as exc:
            raise SonglingRPCTransportError(
                f"Could not connect to Songling RPC at {self.endpoint}: {exc}"
            ) from exc
        return self._connection

    def call(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        try:
            import msgpack
        except ImportError as exc:
            raise RuntimeError(
                "RemoteSonglingEnv requires the 'msgpack' package."
            ) from exc

        request_id = str(uuid.uuid4())
        request = {
            "protocol_version": self.protocol_version,
            "request_id": request_id,
            "method": str(method),
            "params": _to_wire(params or {}),
        }
        payload = msgpack.packb(request, use_bin_type=True)
        with self._lock:
            connection = self._connect()
            try:
                connection.send(payload)
                raw_response = connection.recv(timeout=self.request_timeout_s)
            except Exception as exc:
                # The server may have received an action command. Drop the
                # connection and surface uncertainty; never resend automatically.
                self._drop_connection()
                raise SonglingRPCTransportError(
                    f"Songling RPC {method!r} failed without retry: {exc}"
                ) from exc
            try:
                if isinstance(raw_response, str):
                    raise SonglingRPCProtocolError(
                        "RPC response must be binary MessagePack."
                    )
                try:
                    response = msgpack.unpackb(raw_response, raw=False)
                except Exception as exc:
                    raise SonglingRPCProtocolError(
                        f"Invalid MessagePack response: {exc}"
                    ) from exc
                if not isinstance(response, dict):
                    raise SonglingRPCProtocolError(
                        "RPC response envelope must be a mapping."
                    )
                if response.get("request_id") != request_id:
                    raise SonglingRPCProtocolError(
                        "RPC response request_id mismatch: "
                        f"expected {request_id}, got {response.get('request_id')}."
                    )
                if response.get("protocol_version") != self.protocol_version:
                    raise SonglingRPCProtocolError(
                        "RPC protocol version mismatch: "
                        f"expected {self.protocol_version}, "
                        f"got {response.get('protocol_version')}."
                    )
                if not bool(response.get("ok", False)):
                    error = response.get("error", {})
                    if not isinstance(error, dict):
                        raise SonglingRPCProtocolError(
                            "RPC error payload must be a mapping."
                        )
                    raise SonglingRPCError(
                        f"Songling RPC {method!r} rejected: "
                        f"{error.get('code', 'unknown')}: "
                        f"{error.get('message', error)}"
                    )
                result = _from_wire(response.get("result", {}))
                if not isinstance(result, dict):
                    raise SonglingRPCProtocolError("RPC result must be a mapping.")
                return result
            except SonglingRPCProtocolError:
                # A malformed or mismatched frame may leave request/response
                # alignment unknown. Reconnect before any later command.
                self._drop_connection()
                raise

    def _drop_connection(self) -> None:
        connection, self._connection = self._connection, None
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass

    def close(self) -> None:
        with self._lock:
            self._drop_connection()
