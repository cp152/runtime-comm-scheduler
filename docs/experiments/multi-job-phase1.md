# Phase 1 Multi-Job Replay

This experiment is the unscheduled baseline from `260914/JobPacer工作计划.md`.
Each job is a linear compute/collective sequence. The replay creates one thread
per job inside every rank process, uses one NCCL process group per job, and
launches raw `torch.distributed` collectives without `AdmissionScheduler`.

## Workload definition

The checked-in sample is
[`examples/workloads/multi_job_phase1.json`](../../examples/workloads/multi_job_phase1.json).
It is a deterministic description only; tensors, process groups, CUDA streams,
and launch functions are created by the worker at runtime.

Each task has a stable `task_id` and one of these forms:

```json
{"task_id": "a-compute-0", "kind": "compute", "duration_ms": 2.0}
{"task_id": "a-comm-0", "kind": "collective", "op": "all_reduce", "num_bytes": 16777216}
```

The `job_id`, `process_group_id`, and `ranks` fields identify the job's
communicator. Tasks are executed in array order in Phase 1; explicit DAG
dependencies are intentionally deferred to the later DAG phase.

## Replay and measurements

The driver is [`examples/run_multi_job_phase1.py`](../../examples/run_multi_job_phase1.py).
It starts one worker per rank. The worker records rank-local monotonic
timestamps for job start/completion and every task launch/completion. The
driver reports the maximum job makespan across ranks and preserves the full
per-rank trace in the optional output JSON.

The baseline intentionally has no scheduler policy. Phase 2 can reuse the
same workload and worker task loop, replacing the raw collective launch with
`AdmissionScheduler.submit()` while retaining the trace fields.