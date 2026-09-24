"""Tier 1: the ledger as a CHIA node, dispatched through a local Ray with the profiler on.
Skipped where Ray is not installed."""

import json
import os

import pytest

ray = pytest.importorskip("ray")


def test_ledger_node_records_and_reports_through_ray(tmp_path):
    from chia.base.ChiaFunction import get
    from chia.trace.ledger import LedgerNode
    from chia.trace.profiler import get_collector, reset_profiler, start_collector

    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False, logging_level="ERROR", ignore_reinit_error=True)
    try:
        log_dir = str(tmp_path / "profile")
        os.makedirs(log_dir)
        start_collector(log_dir=log_dir)
        reset_profiler()      # a profiler made before the collector existed stays disabled
        path = str(tmp_path / "run.jsonl")
        row = get(LedgerNode.record.chia_remote(path, "ipc", {"l2_sets": 256}, {"ipc": 1.41}, "council:prefetch", 3, "because", (), {"usd": 0.02}))
        assert row["index"] == 0 and row["value"] == 1.41
        get(LedgerNode.record.chia_remote(path, "ipc", {"l2_sets": 512}, {"ipc": 1.30}, "sweep", 3, None, ["power 1.2 W"], {}))
        report = get(LedgerNode.report.chia_remote({"run": path}, "ipc"))
        assert report["rows"]["run"]["evaluations"] == 2
        assert report["rows"]["run"]["final"] == 1.41
        assert report["rows"]["run"]["feasible_rate"] == 0.5
        # Events are recorded fire-and-forget; a blocking call on the collector lands every one sent.
        ray.get(get_collector().get_events.remote())
        events = [json.loads(line) for line in open(os.path.join(log_dir, "ChiaProfileCollector.log")) if line.strip()]
        types = set(event.get("type") for event in events)
        assert "candidate" in types and "complete" in types
    finally:
        ray.shutdown()
