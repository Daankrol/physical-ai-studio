# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TypedDict
from unittest.mock import MagicMock

import numpy as np

from physicalai.inference.constants import ACTION
from physicalai.inference.runners.action_chunking import ActionChunking
from physicalai.inference.runners.base import InferenceRunner


CHUNK_SIZE = 5
ACTION_DIM = 3


def _make_chunk(start: int) -> np.ndarray:
    return np.arange(start, start + (CHUNK_SIZE * ACTION_DIM), dtype=np.float32).reshape(1, CHUNK_SIZE, ACTION_DIM)


def _make_inputs() -> dict[str, np.ndarray]:
    return {"state": np.arange(4, dtype=np.float32).reshape(1, 4)}


def _copy_inputs(inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {key: value.copy() for key, value in inputs.items()}


def _action_at(chunk: np.ndarray, index: int) -> np.ndarray:
    return chunk[:, index, :]


class RecordingRunner(InferenceRunner):
    class CallRecord(TypedDict):
        adapter: object
        inputs: dict[str, np.ndarray]

    def __init__(self, chunks: list[np.ndarray]) -> None:
        self._chunks = [chunk.copy() for chunk in chunks]
        self.calls: list[RecordingRunner.CallRecord] = []
        self.reset_calls = 0

    def run(self, adapter: object, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        call_index = len(self.calls)
        if call_index >= len(self._chunks):
            msg = "Unexpected extra inference call"
            raise AssertionError(msg)

        self.calls.append({"adapter": adapter, "inputs": _copy_inputs(inputs)})
        return {ACTION: self._chunks[call_index].copy()}

    def reset(self) -> None:
        self.reset_calls += 1


class TestActionChunkingBasic:
    def test_standard_chunking(self) -> None:
        first_chunk = _make_chunk(0)
        second_chunk = _make_chunk(100)
        inner_runner = RecordingRunner([first_chunk, second_chunk])
        runner = ActionChunking(inner_runner, chunk_size=CHUNK_SIZE)
        adapter = MagicMock()
        inputs = _make_inputs()

        first = runner.run(adapter, inputs)[ACTION]
        np.testing.assert_array_equal(first, _action_at(first_chunk, 0))
        assert len(inner_runner.calls) == 1

        second = runner.run(adapter, inputs)[ACTION]
        third = runner.run(adapter, inputs)[ACTION]
        fourth = runner.run(adapter, inputs)[ACTION]
        fifth = runner.run(adapter, inputs)[ACTION]

        np.testing.assert_array_equal(second, _action_at(first_chunk, 1))
        np.testing.assert_array_equal(third, _action_at(first_chunk, 2))
        np.testing.assert_array_equal(fourth, _action_at(first_chunk, 3))
        np.testing.assert_array_equal(fifth, _action_at(first_chunk, 4))
        assert len(inner_runner.calls) == 1

        sixth = runner.run(adapter, inputs)[ACTION]
        np.testing.assert_array_equal(sixth, _action_at(second_chunk, 0))
        assert len(inner_runner.calls) == 2
        assert inner_runner.calls[0]["adapter"] is adapter

    def test_reset_clears_queue(self) -> None:
        first_chunk = _make_chunk(0)
        second_chunk = _make_chunk(100)
        inner_runner = RecordingRunner([first_chunk, second_chunk])
        runner = ActionChunking(inner_runner, chunk_size=CHUNK_SIZE)
        adapter = MagicMock()
        inputs = _make_inputs()

        runner.run(adapter, inputs)
        runner.run(adapter, inputs)
        assert len(runner._action_queue) == 3

        runner.reset()

        assert len(runner._action_queue) == 0
        assert inner_runner.reset_calls == 1

        action = runner.run(adapter, inputs)[ACTION]
        np.testing.assert_array_equal(action, _action_at(second_chunk, 0))
        assert len(inner_runner.calls) == 2


class TestActionChunkingRTC:
    def test_rtc_triggers_early(self) -> None:
        first_chunk = _make_chunk(0)
        second_chunk = _make_chunk(100)
        inner_runner = RecordingRunner([first_chunk, second_chunk])
        runner = ActionChunking(
            inner_runner,
            chunk_size=CHUNK_SIZE,
            rtc_max_delay=3,
            action_dim=ACTION_DIM,
        )
        adapter = MagicMock()
        inputs = _make_inputs()

        first = runner.run(adapter, inputs)[ACTION]
        second = runner.run(adapter, inputs)[ACTION]
        third = runner.run(adapter, inputs)[ACTION]

        np.testing.assert_array_equal(first, _action_at(first_chunk, 0))
        np.testing.assert_array_equal(second, _action_at(first_chunk, 1))
        np.testing.assert_array_equal(third, _action_at(second_chunk, 3))
        assert len(inner_runner.calls) == 2
        assert "action_prefix" in inner_runner.calls[1]["inputs"]
        np.testing.assert_array_equal(inner_runner.calls[1]["inputs"]["delay"], np.array(3))

    def test_rtc_prefix_construction(self) -> None:
        first_chunk = _make_chunk(0)
        second_chunk = _make_chunk(100)
        inner_runner = RecordingRunner([first_chunk, second_chunk])
        runner = ActionChunking(
            inner_runner,
            chunk_size=CHUNK_SIZE,
            rtc_max_delay=3,
            action_dim=ACTION_DIM,
        )
        adapter = MagicMock()

        runner.run(adapter, _make_inputs())
        runner.run(adapter, _make_inputs())
        runner.run(adapter, _make_inputs())

        action_prefix = inner_runner.calls[1]["inputs"]["action_prefix"]
        expected_prefix = np.zeros((1, CHUNK_SIZE, ACTION_DIM), dtype=np.float32)
        expected_prefix[:, 0, :] = _action_at(first_chunk, 2)
        expected_prefix[:, 1, :] = _action_at(first_chunk, 3)
        expected_prefix[:, 2, :] = _action_at(first_chunk, 4)

        assert action_prefix.shape == (1, CHUNK_SIZE, ACTION_DIM)
        np.testing.assert_array_equal(action_prefix, expected_prefix)

    def test_rtc_postfix_only_enqueued(self) -> None:
        """After RTC inference, only postfix (chunk_size - delay) actions are enqueued."""
        first_chunk = _make_chunk(0)
        second_chunk = _make_chunk(100)
        inner_runner = RecordingRunner([first_chunk, second_chunk])
        runner = ActionChunking(
            inner_runner,
            chunk_size=CHUNK_SIZE,
            rtc_max_delay=1,
            action_dim=ACTION_DIM,
        )
        adapter = MagicMock()
        inputs = _make_inputs()

        # run 1: queue=0 ≤ 1, RTC delay=0 → enqueue all 5, pop [0]. Queue: [1,2,3,4]
        runner.run(adapter, inputs)
        # run 2-4: queue > 1, pop from queue
        runner.run(adapter, inputs)
        runner.run(adapter, inputs)
        runner.run(adapter, inputs)
        # Queue: [4], len=1 ≤ 1 → RTC with delay=1
        fifth = runner.run(adapter, inputs)[ACTION]
        # second_chunk returned, discard first 1, enqueue postfix [1,2,3,4]
        np.testing.assert_array_equal(fifth, _action_at(second_chunk, 1))
        assert len(inner_runner.calls) == 2

        sixth = runner.run(adapter, inputs)[ACTION]
        np.testing.assert_array_equal(sixth, _action_at(second_chunk, 2))
        seventh = runner.run(adapter, inputs)[ACTION]
        np.testing.assert_array_equal(seventh, _action_at(second_chunk, 3))

    def test_rtc_delay_zero_no_prefix(self) -> None:
        first_chunk = _make_chunk(0)
        inner_runner = RecordingRunner([first_chunk])
        runner = ActionChunking(
            inner_runner,
            chunk_size=CHUNK_SIZE,
            rtc_max_delay=3,
            action_dim=ACTION_DIM,
        )
        adapter = MagicMock()

        action = runner.run(adapter, _make_inputs())[ACTION]

        np.testing.assert_array_equal(action, _action_at(first_chunk, 0))
        assert "action_prefix" not in inner_runner.calls[0]["inputs"]
        assert "delay" not in inner_runner.calls[0]["inputs"]

    def test_rtc_disabled_when_zero(self) -> None:
        first_chunk = _make_chunk(0)
        second_chunk = _make_chunk(100)
        standard_inner = RecordingRunner([first_chunk, second_chunk])
        rtc_zero_inner = RecordingRunner([first_chunk, second_chunk])
        standard_runner = ActionChunking(standard_inner, chunk_size=CHUNK_SIZE)
        rtc_zero_runner = ActionChunking(
            rtc_zero_inner,
            chunk_size=CHUNK_SIZE,
            rtc_max_delay=0,
            action_dim=ACTION_DIM,
        )
        standard_outputs = []
        rtc_zero_outputs = []

        for _ in range(6):
            standard_outputs.append(standard_runner.run(MagicMock(), _make_inputs())[ACTION])
            rtc_zero_outputs.append(rtc_zero_runner.run(MagicMock(), _make_inputs())[ACTION])

        assert len(standard_inner.calls) == 2
        assert len(rtc_zero_inner.calls) == 2
        for standard_output, rtc_zero_output in zip(standard_outputs, rtc_zero_outputs, strict=True):
            np.testing.assert_array_equal(standard_output, rtc_zero_output)
        for call in rtc_zero_inner.calls:
            assert "action_prefix" not in call["inputs"]
            assert "delay" not in call["inputs"]
