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


def test_compact_global_comparison_preserves_outliers(reporter, tmp_path):
    fields = ["ep", "tokens", "hidden", "topk", "experts", "dtype", "route", "seed",
              "warmup", "samples", "entry_sync", "profile_build"]
    config = dict(ep=2, tokens=32, hidden=3584, topk=8, experts=384, dtype="bf16", route="random",
                  seed=0, warmup=10, samples=6, entry_sync="device", profile_build=0)
    for name, values in (("0-base", [10, 10, 11, 10, 11, 100]), ("1-remote", [9, 10, 9, 9, 10, 10])):
        directory = tmp_path / name
        directory.mkdir()
        write_csv(directory / "summary.csv", fields, [config])
        write_csv(directory / "rank-max.csv", ["tokens", "iteration", "rank", "us"],
                  [dict(tokens=32, iteration=i, rank=i % 2, us=v) for i, v in enumerate(values)])
    out = subprocess.check_output([str(reporter), "--suite", str(tmp_path)], text=True)
    assert "no samples removed" in out
    row = next(line for line in out.splitlines() if line.startswith("0-base"))
    assert "25.33" in row and "100.00" in row and "WIDE" in row  # retain the 100 us tail
    assert "IQR 0-base n=1" in out and "5/1/100.00" in out
    assert out.index("variant ") < out.index("IQR 0-base")
    # A local node's subset must not silently become a global statistic.
    path = tmp_path / "0-base/rank-max.csv"
    original = path.read_text()
    path.write_text("\n".join(original.splitlines()[:-1]) + "\n")
    result = subprocess.run([str(reporter), "--suite", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0 and "Missing global samples" in result.stderr


def test_compact_diagnostics_select_same_global_iteration(reporter, tmp_path):
    directory = tmp_path / "0-ref-profile"
    directory.mkdir()
    fixture_data(directory)
    config = dict(ep=2, tokens=32, hidden=3584, topk=8, experts=384, dtype="bf16", route="random",
                  seed=0, warmup=10, samples=3, entry_sync="device", profile_build=1)
    write_csv(directory / "summary.csv", list(config), [config])
    write_csv(directory / "rank-max.csv", ["tokens", "iteration", "rank", "us"],
              [dict(tokens=32, iteration=i, rank=r, us=v) for i, r, v in [(0, 0, 10), (1, 1, 30), (2, 0, 13)]])
    records = []
    for rank in (0, 1):
        with (directory / f"rank-{rank}-profile.csv").open() as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            records.extend(row for row in reader if row["iteration"] in ("1", "2"))
    write_csv(directory / "diagnostic.csv", fields, records)
    out = subprocess.check_output([str(reporter), "--suite", str(tmp_path)], text=True)
    assert "D 0-ref-profile p50 i=2 event_r=0 us=13.00" in out
    assert "D 0-ref-profile worst i=1 event_r=1 us=30.00" in out
    assert "S r/c=1/0" in out and "R r/c=1/1" in out
    # Host 0 alone cannot satisfy the selected global sample contract.
    write_csv(directory / "diagnostic.csv", fields, [r for r in records if r["rank"] == "0"])
    result = subprocess.run([str(reporter), "--suite", str(tmp_path)], capture_output=True, text=True)
    assert result.returncode != 0 and "Partial global diagnostics" in result.stderr


def test_tail_link_and_recurrence_use_actual_slot_map(reporter, tmp_path):
    directory = tmp_path / "0-flag-profile"
    directory.mkdir()
    fixture_data(directory)
    config = dict(ep=2, tokens=32, hidden=3584, topk=8, experts=384, dtype="bf16", route="random",
                  seed=0, warmup=10, samples=3, entry_sync="host", profile_build=1)
    write_csv(directory / "summary.csv", list(config), [config])
    write_csv(directory / "rank-max.csv", ["tokens", "iteration", "rank", "us"],
              [dict(tokens=32, iteration=i, rank=1, us=v) for i, v in enumerate((10, 20, 30))])
    records = []
    for rank in (0, 1):
        with (directory / f"rank-{rank}-profile.csv").open() as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            for row in reader:
                if rank == 1 and row["role"] == "receive":
                    row.update(tail_token=7, tail_slot=2, tail_block=14, tail_missing_mask=4,
                               tail_missing_blocks=1, tail_observations=5, tail_first_seen=30,
                               tail_last_seen=60, tail_ready=70, schema=5)
                records.append(row)
    write_csv(directory / "diagnostic.csv", fields, records)
    p = dict(tokens=32, source=1, token=7, slot=2, producer=0, core=31, type=2, row=93, path="RDMA")
    write_csv(directory / "producers.csv", list(p), [p])
    out = subprocess.check_output([str(reporter), "--suite", str(tmp_path)], text=True)
    assert "event-tail-missed=3/3" in out and "0>1/RDMA/P:3" in out
    assert "same-slot=r1/t7/k2:3" in out
    assert "k2:r0/c31/RDMA/PARTIAL" in out and "first=k2/b14" in out
    assert "first/last/ready=30/60/70" in out


def test_local_first_fields_in_reports(reporter, tmp_path):
    directory = tmp_path / "0-local-first-profile"
    directory.mkdir()
    fixture_data(directory)
    config = dict(ep=2, tokens=32, hidden=3584, topk=8, experts=384, dtype="bf16", route="random",
                  seed=0, warmup=10, samples=3, entry_sync="host", profile_build=1, local_first=1)
    write_csv(directory / "summary.csv", list(config), [config])
    write_csv(directory / "rank-max.csv", ["tokens", "iteration", "rank", "us"],
              [dict(tokens=32, iteration=i, rank=r, us=v) for i, r, v in [(0, 0, 10), (1, 1, 30), (2, 0, 13)]])
    records = []
    for rank in (0, 1):
        path = directory / f"rank-{rank}-profile.csv"
        with path.open() as f:
            reader = csv.DictReader(f)
            fields, rows = reader.fieldnames, list(reader)
        for row in rows:
            row["schema"] = 7
            if row["role"] == "receive":
                row.update(local_contributions=2, remote_contributions=3, local_probe_calls=5, remote_probe_calls=8,
                           local_probe_misses=3, remote_probe_misses=5, local_probe_ticks=10, remote_probe_ticks=30,
                           local_reduce_ticks=7, remote_reduce_ticks=9)
        write_csv(path, fields, rows)
        records.extend(rows)
    write_csv(directory / "diagnostic.csv", fields, records)
    full = subprocess.check_output([str(reporter), str(directory)], text=True)
    assert "final_local: contributions=2 probes/misses=5/3 probe_ticks=10 reduce_ticks=7" in full
    assert "final_remote: contributions=3 probes/misses=8/5 probe_ticks=30 reduce_ticks=9" in full
    compact = subprocess.check_output([str(reporter), "--suite", str(tmp_path)], text=True)
    assert "final_by_host local/remote contributions=2/3 probe_ticks=10/30 misses=3/5 calls=5/8 reduce_ticks=7/9" in compact
