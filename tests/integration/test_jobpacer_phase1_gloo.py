"""Two-rank Gloo acceptance test for the JobPacer Phase 1 baseline."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")


def _require_loopback_socket() -> None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
    except PermissionError:
        pytest.skip("sandbox does not permit the local Gloo rendezvous socket")


def test_phase1_bare_replay_emits_job_makespans_and_trace(tmp_path):
    _require_loopback_socket()
    root = Path(__file__).parents[2]
    output = tmp_path / "phase1.json"
    env = dict(os.environ)
    source_root = str(root / "src")
    env["PYTHONPATH"] = source_root + os.pathsep + env.get("PYTHONPATH", "")
    process = subprocess.run(
        [
            sys.executable,
            str(root / "examples/jobpacer/run_phase1.py"),
            "--workload",
            "balanced",
            "--backend",
            "gloo",
            "--world-size",
            "2",
            "--timeout",
            "20",
            "--output",
            str(output),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr + process.stdout
    payload = json.loads(output.read_text())
    assert payload["config"]["mode"] == "bare"
    assert payload["validation"]["status"] == "ok"
    assert payload["performance"]["workload_makespan_us"] > 0
    assert len(payload["performance"]["job_makespans"]) == 2
    for rank in payload["ranks"]:
        assert len(rank["launch_sequence"]) == 6
        for job in rank["jobs"]:
            assert job["makespan_us"] > 0
            assert len(job["tasks"]) == 3
            for task in job["tasks"]:
                assert task["submit_call_ts"] <= task["submit_return_ts"]
                assert task["submit_return_ts"] <= task["consumer_compute_end_ts"]
                assert task["first_wait_ts"] <= task["wait_return_ts"]
                assert task["wait_return_ts"] <= task["consumer_end_ts"]
