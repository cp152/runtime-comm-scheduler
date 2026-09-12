"""Rank-local collective execution plane.

The scheduler owns the only host launch worker.  Executors only establish the
rank-local device/stream context and invoke an intent's asynchronous launcher.
"""

from __future__ import annotations

import threading
from typing import Any, Protocol, runtime_checkable

from .intent import CommIntent


@runtime_checkable
class LaunchExecutor(Protocol):
    """Execution-plane boundary used by the scheduler worker."""

    def launch(self, intent: CommIntent) -> Any:
        """Launch ``intent`` asynchronously and return a Work-like object."""


@runtime_checkable
class CompletionProbe(Protocol):
    """Backend-specific physical-completion observation."""

    supports_physical_completion: bool

    def is_completed(self, underlying: Any) -> bool:
        """Return whether the underlying operation physically completed."""


class WorkIsCompletedProbe:
    """Use the backend Work's non-blocking ``is_completed`` query.

    This is a physical-completion probe for the ProcessGroupNCCL versions
    qualified by the GPU capability harness and for the CPU test/Gloo works
    used by this project.  Other backends should provide their own probe.
    """

    supports_physical_completion = True

    def is_completed(self, underlying: Any) -> bool:
        return bool(underlying.is_completed())


class DirectLaunchExecutor:
    """Launch without a CUDA bridge (CPU/Gloo and unit-test execution)."""

    def launch(self, intent: CommIntent) -> Any:
        if intent.launch_fn is None:
            raise ValueError(f"intent {intent.key.as_list()} has no launch_fn")
        return intent.launch_fn()


class TorchProcessGroupExecutor:
    """Bridge producer dependencies into ProcessGroupNCCL.

    A single host worker uses this executor, but each logical process group has
    its own gate stream.  Sharing one gate stream across groups would make a
    later intent inherit earlier intents' producer-ready dependencies and
    introduce unnecessary cross-communicator head-of-line blocking.
    """

    def __init__(self, device: Any) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise RuntimeError("TorchProcessGroupExecutor requires torch") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("TorchProcessGroupExecutor requires CUDA")
        self._torch = torch
        self.device = (
            torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device)
        )
        if self.device.type != "cuda":
            raise ValueError(f"expected a CUDA device, got {self.device}")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._gate_streams: dict[str, Any] = {}

    def launch(self, intent: CommIntent) -> Any:
        if intent.launch_fn is None:
            raise ValueError(f"intent {intent.key.as_list()} has no launch_fn")
        self._validate_device(intent)
        torch = self._torch
        group_id = intent.key.process_group_id
        with torch.cuda.device(self.device):
            gate = self._gate_streams.get(group_id)
            if gate is None:
                gate = torch.cuda.Stream()
                self._gate_streams[group_id] = gate
            with torch.cuda.stream(gate):
                event = intent.ready_event
                if event is not None and not isinstance(event, threading.Event):
                    gate.wait_event(event)
                return intent.launch_fn()

    def gate_stream(self, group_id: str) -> Any | None:
        """Return an already-created gate stream (primarily for GPU QA)."""
        return self._gate_streams.get(group_id)

    def _validate_device(self, intent: CommIntent) -> None:
        torch = self._torch
        declared = intent.device
        tensor_device = getattr(intent.tensor, "device", None)
        for candidate in (declared, tensor_device):
            if candidate is None:
                continue
            device = (
                torch.device("cuda", candidate)
                if isinstance(candidate, int)
                else torch.device(candidate)
            )
            index = torch.cuda.current_device() if device.index is None else device.index
            if device.type != "cuda" or index != self.device.index:
                raise ValueError(
                    f"intent {intent.key.as_list()} uses device {device}, "
                    f"executor owns {self.device}"
                )


def validate_underlying_work(underlying: Any) -> None:
    """Reject synchronous/invalid launch results before binding them."""
    if underlying is None:
        raise TypeError("launch_fn must return an asynchronous Work, got None")
    for method in ("wait", "is_completed"):
        if not callable(getattr(underlying, method, None)):
            raise TypeError(
                "launch_fn must return a Work-like object with wait() and "
                f"is_completed(); missing {method}()"
            )
