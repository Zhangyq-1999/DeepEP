"""Exercise the offline report with different slow ranks, samples and missing data."""
import csv
import hashlib
import os
from pathlib import Path
import re
import subprocess

import pytest

HERE = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def reporter(tmp_path_factory):
    zig = os.environ.get("DEEPEP_TEST_ZIG")
    if not zig:
        pytest.skip("Set DEEPEP_TEST_ZIG for the standalone C++ report")
    exe = tmp_path_factory.mktemp("report-build") / "report.exe"
    subprocess.run([zig, "c++", "-std=c++17", "-O2", str(HERE / "profile_report.cpp"), "-o", str(exe)],
                   check=True, timeout=180)
    return exe


def write_csv(path, fields, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fixture_data(path, ranks=(0, 1), level=2):
    names = re.findall(r'"([a-z0-9_]+)"',
                       (HERE / "profile.hpp").read_text().split("names[] = {", 1)[1].split("};", 1)[0])
    fields = ["iteration", "device", "role", "profile_build"] + names
    timestamp_end = names.index("stage_rows")
    for rank in ranks:
        rows, events = [], []
        for iteration in range(3):
            # Median is iteration 2/rank 0; worst is iteration 1/rank 1.
            us = ((10, 11, 13), (9, 30, 12))[rank][iteration]
            events.append(dict(tokens=32, iteration=iteration, combine_event_us=us))
            for core in range(2):
                row = {name: 0 for name in names}
                for name in names[:timestamp_end]:
                    row[name] = ""
                if level == 1:
                    for name in names[names.index("init_prefix_ticks"):names.index("rank")]:
                        row[name] = ""
                row.update(iteration=iteration, device=6+rank, role="receive" if core else "send",
                           profile_build=1, entry_tick=(1 << 60) + rank * 1000000 + core * 100,
                           flush_done=1000+rank*100+iteration*10+core,
                           profile_init_end=2, magic_end=3, init_end=5,
                           rank=rank, core=core, tokens=32, world=2, schema=3, complete=1, valid=1, level=level,
                           input_lanes=2, output_lanes=2)
                if core == 0:
                    # Maximum per iteration is 100,20,40, across different ranks.
                    row.update(stage_begin=10, stage_end=10+(((10, 20, 40), (100, 10, 20))[rank][iteration]),
                               stage_rows=10+rank, direct_rows=3,
                               direct_begin=120, direct_setup_end=130, direct_end=180)
                    if level == 2:
                        row.update(direct_submit_ticks=111+rank*100+iteration,
                                   direct_submit_calls=3, direct_submit_max_ticks=20,
                                   direct_submit_max_peer=1-rank, direct_submit_max_token=17)
                else:
                    row.update(final_begin=10, final_end=80, drain_end=85, final_tokens=16)
                    if level == 2:
                        row.update(probe_ticks=50, probe_calls=10, probe_misses=4, probe_miss_ticks=20,
                                   probe_max_ticks=8, probe_max_token=9, probe_max_slot=4294967295)
                row["pipe_done"] = row["flush_done"] - 10
                rows.append(row)
        write_csv(path / f"rank-{rank}-profile.csv", fields, rows)
        write_csv(path / f"rank-{rank}.csv", ["tokens", "iteration", "combine_event_us"], events)


def test_report_same_sample_and_scope(reporter, tmp_path):
    fixture_data(tmp_path)
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}
    result = subprocess.run([str(reporter), str(tmp_path)], capture_output=True, text=True, check=True)
    out = result.stdout
    assert "coverage=2/2 FULL samples=3" in out
    assert "mean/min/max=17.67/10.00/30.00" in out
    assert re.search(r"stage\s+53\.33\s+100\s+0/1/0", out)
    assert "MEDIAN_EVENT_SAMPLE iter=2 scope_max_Event_us=13.00 Event_rank=0" in out
    assert "WORST_EVENT_SAMPLE iter=1 scope_max_Event_us=30.00 Event_rank=1" in out
    median, worst = out.split("MEDIAN_EVENT_SAMPLE", 1)[1].split("WORST_EVENT_SAMPLE", 1)
    assert "direct_submit_ticks=113" in median and "direct_submit_ticks=212" in worst
    assert "slot=all" in out and "body_ticks=1020" in out
    assert "pipe_tail_ticks=830" in median  # 1010 - 180, same sample/core.
    assert "send_pipe_tail" in out and "recv_pipe_tail" in out and "batch_flush" in out
    assert "rdma_tail" not in out
    assert "SEND longest_body iter=2 rank=0 dev=6 core=0 body_ticks=1020\n" in out
    assert "notify_wait" not in out  # Absent timestamps are not zero-duration phases.
    assert out.count("LL_COMBINE_PROFILE_SUMMARY BEGIN") == 1
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in tmp_path.iterdir()}


def test_local_scope_and_level_one(reporter, tmp_path):
    fixture_data(tmp_path, ranks=(1,), level=1)
    out = subprocess.check_output([str(reporter), str(tmp_path)], text=True)
    assert "coverage=1/2 PARTIAL: local files only" in out
    assert "scope_ranks=1" in out
    assert "top_ticks:" not in out and "polls calls/misses" not in out


@pytest.mark.parametrize("problem", ["missing_core", "duplicate_core", "missing_event", "nonmonotonic"])
def test_incomplete_report_rejected(reporter, tmp_path, problem):
    fixture_data(tmp_path)
    path = tmp_path / ("rank-1.csv" if problem == "missing_event" else "rank-1-profile.csv")
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    if problem in ("missing_core", "missing_event"):
        rows.pop()
    elif problem == "duplicate_core":
        rows.append(rows[0])
    else:
        rows[0]["stage_end"] = "1"
    write_csv(path, fields, rows)
    result = subprocess.run([str(reporter), str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "Profile summary failed:" in result.stderr


def test_return_submission_and_overlap_report(reporter, tmp_path):
    fixture_data(tmp_path)
    for rank in (0, 1):
        path = tmp_path / f"rank-{rank}-profile.csv"
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        for row in rows:
            if row["role"] == "send":
                row.update(rdma_first_stage=100, rdma_last_stage=400,
                           rdma_first_put=200, rdma_last_put=500, rdma_packets=3,
                           rdma_deferred_packets=1, rdma_stage_wait_ticks=50,
                           rdma_stage_wait_max_ticks=30, rdma_stage_wait_max_peer=17,
                           rdma_defer_gap_ticks=99999)
        write_csv(path, fields, rows)
    out = subprocess.check_output([str(reporter), str(tmp_path)], text=True)
    assert "rdma_return: packets=3 deferred=1" in out
    assert "RETURN largest_last_put_offset (own-core entry)" in out
    assert "stage_wait=50 defer_gap=99999" in out
    assert "max=30(peer=17)" in out
    for line in out.splitlines():
        if "top_ticks:" in line:
            assert "defer_gap" not in line  # useful preparation time is not a cost


def test_schema_two_report_remains_readable(reporter, tmp_path):
    fixture_data(tmp_path)
    added = {"rdma_first_stage", "rdma_last_stage", "rdma_first_put", "rdma_last_put",
             "rdma_deferred_packets", "rdma_defer_gap_ticks", "rdma_stage_wait_max_ticks", "rdma_stage_wait_max_peer"}
    for path in tmp_path.glob("*-profile.csv"):
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        fields = [f for f in fields if f not in added]
        rows = [{k: v for k, v in row.items() if k not in added} for row in rows]
        for row in rows:
            row["schema"] = 2
        write_csv(path, fields, rows)
    out = subprocess.check_output([str(reporter), str(tmp_path)], text=True)
    assert "coverage=2/2 FULL" in out
    assert "rdma_return:" not in out
