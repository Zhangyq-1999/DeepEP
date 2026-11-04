"""Export real Host CSV fixtures; check clock precision, tracks and sample identity."""
import csv
import json
import os
from pathlib import Path
import subprocess

import pytest
from test_profile_report import fixture_data, reporter, write_csv

HERE = Path(__file__).resolve().parent


def read_trace(path):
    trace = json.loads(path.read_text())
    assert trace['otherData']['clock_scope'] == 'core-local'
    slices = [e for e in trace['traceEvents'] if e['ph'] == 'X']
    assert all(e['ts'] >= 0 and e['dur'] >= 0 for e in slices)
    # Chrome X events on a track must nest; partial and ready can cross in kernels.
    for track in {(e['pid'], e['tid']) for e in slices}:
        stack = []
        for e in sorted((e for e in slices if (e['pid'], e['tid']) == track),
                        key=lambda e: (e['ts'], -e['dur'])):
            end = e['ts'] + e['dur']
            while stack and stack[-1] <= e['ts'] + 1e-12:
                stack.pop()
            assert not stack or end <= stack[-1] + 1e-12
            stack.append(end)
    return trace


def test_offline_trace_selection_precision_and_repeat(reporter, tmp_path):
    fixture_data(tmp_path)
    for hz in (1000000000, 500000000):
        subprocess.run([str(reporter), '--perfetto', str(tmp_path), str(hz)], check=True, capture_output=True)
    for suffix, factor in [('', 1), ('-1', 2)]:
        directory = tmp_path / ('perfetto-local' + suffix)
        median = read_trace(directory / 'bs32-p50-i2.perfetto.json')
        worst = read_trace(directory / 'bs32-worst-i1.perfetto.json')
        assert median['otherData']['sample_event_us'] == 13
        assert worst['otherData']['sample_event_us'] == 30
        assert 'coverage=2/2' in median['otherData']['selection']
        bodies = [e for e in median['traceEvents'] if e['name'] == 'profiled body']
        assert len(bodies) == 4  # all cores and ranks of the SAME iteration
        assert all(e['ts'] == 0 for e in bodies)
        assert bodies[0]['dur'] == pytest.approx(1.020 * factor)
        assert bodies[0]['args']['entry_tick'] == str(1 << 60)
        assert bodies[0]['args']['direct_submit_ticks'] == 113
        assert not any(e['name'] == 'direct_submit_ticks' for e in median['traceEvents'])
        assert not any(e['name'] == 'local ready wait' for e in median['traceEvents'])
        assert not list(directory.glob('*.tmp'))


def test_local_coverage_crossing_stages_and_tail(reporter, tmp_path):
    fixture_data(tmp_path, ranks=(1,))
    path = tmp_path / 'rank-1-profile.csv'
    with path.open() as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        if row['role'] == 'receive':
            row.update(core_notify_begin=3, core_notify_end=5, ready_begin=5, ready_observed=23,
                       ready_end=25, partial_begin=20, partial_end=40,
                       ready_missing_rank=0, ready_missing_core=55, ready_missing_count=1,
                       final_begin=40, tail_observations=2, tail_first_seen=50, tail_last_seen=60,
                       tail_ready=70, tail_slot=3, tail_token=7, tail_missing_blocks=2)
    write_csv(path, fields, rows)
    subprocess.run([str(reporter), '--perfetto', str(tmp_path)], check=True, capture_output=True)
    trace = read_trace(next((tmp_path/'perfetto-local').glob('*p50-*.json')))
    assert 'coverage=1/2' in trace['otherData']['selection']
    events = trace['traceEvents']
    ready = next(e for e in events if e['name'] == 'group ready wait')
    partial = next(e for e in events if e['name'] == 'FIRST_HOP partial')
    assert ready['tid'] != partial['tid']  # crossing events keep their true endpoints
    spans = {e['name']: e for e in events if e['ph'] == 'X'}
    assert spans['per-core SIMT publish']['dur'] == pytest.approx(.002)
    assert spans['group ready polling']['dur'] == pytest.approx(.018)
    assert spans['ready inspect and reset']['dur'] == pytest.approx(.002)
    assert spans['profiled body']['args']['ready_missing_core'] == 55
    tail = next(e for e in events if e['name'] == 'tail_ready')
    assert tail['ph'] == 'i' and tail['args']['tail_token'] == 7


def test_no_trace_for_missing_core_or_bad_clock(reporter, tmp_path):
    fixture_data(tmp_path)
    path = tmp_path / 'rank-1-profile.csv'
    lines = path.read_text().splitlines()
    path.write_text('\n'.join(lines[:-1])+'\n')
    result = subprocess.run([str(reporter), '--perfetto', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode and 'Missing profile' in result.stderr
    assert not list(tmp_path.glob('**/*.perfetto.json'))
    fixture_data(tmp_path)
    with path.open() as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    rows[-1]['rdma_first_put'] = 99999
    write_csv(path, fields, rows)
    result = subprocess.run([str(reporter), '--perfetto', str(tmp_path)], capture_output=True, text=True)
    assert result.returncode and 'Timestamp outside' in result.stderr
    assert not list(tmp_path.glob('**/*.tmp'))


def test_dependency_work_is_not_rendered_as_blocking_wait(reporter, tmp_path):
    fixture_data(tmp_path, ranks=(1,))
    path = tmp_path / 'rank-1-profile.csv'
    with path.open() as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        if row['role'] == 'receive':
            row.update(ready_begin=5, ready_observed=90, ready_end=95,
                       ready_first_task=8, ready_tasks_done=40, ready_task_rounds=3,
                       ready_tasks_before_all=2, ready_remote_before_all=1,
                       ready_retire_begin=85, ready_retire_end=95,
                       partial_begin=5, partial_end=40, final_begin=40)
    write_csv(path, fields, rows)
    subprocess.run([str(reporter), '--perfetto', str(tmp_path)], check=True, capture_output=True)
    trace = read_trace(next((tmp_path/'perfetto-local').glob('*p50-*.json')))
    spans = {e['name']: e for e in trace['traceEvents'] if e['ph'] == 'X'}
    assert 'group ready wait' not in spans and 'group ready polling' not in spans
    assert spans['group readiness observation window (includes work)']['dur'] == pytest.approx(.085)
    assert spans['dependency-driven partial work']['dur'] == pytest.approx(.032)
    assert spans['ready retirement']['dur'] == pytest.approx(.010)
    assert spans['profiled body']['args']['ready_remote_before_all'] == 1


def test_automatic_global_trace_gather(tmp_path):
    zig = os.environ.get('DEEPEP_TEST_ZIG')
    if not zig:
        pytest.skip('Set DEEPEP_TEST_ZIG for C++ Host test')
    (tmp_path/'mpi.h').write_text(r'''
#pragma once
#include <cstring>
#include <cstdint>
#include "../../csrc/kernels/combine/low_latency/combine_profile_layout.h"
using MPI_Comm=int;
constexpr int MPI_SUCCESS=0,MPI_COMM_WORLD=1,MPI_ERRORS_RETURN=0,MPI_COMM_NULL=0,MPI_COMM_TYPE_SHARED=0;
constexpr int MPI_INFO_NULL=0,MPI_INT=0,MPI_BYTE=0,MPI_UINT64_T=0,MPI_MAX_ERROR_STRING=1024;
inline int MPI_Error_string(int,char*,int*) { return 0; }
inline int MPI_Comm_set_errhandler(int,int) { return 0; }
inline int MPI_Comm_rank(int,int* r) { *r=0; return 0; }
inline int MPI_Comm_size(int,int* r) { *r=2; return 0; }
inline int MPI_Comm_free(int*) { return 0; }
inline int MPI_Comm_split_type(int,int,int,int,int* c) { *c=2;return 0; }
inline int MPI_Bcast(void*,int,int,int,int) { return 0; }
inline int MPI_Barrier(int) { return 0; }
inline int MPI_Allgather(const void* s,int n,int,void* d,int,int,int) {
    using namespace CombineProfile;
    std::memcpy(d,s,n); std::memcpy(static_cast<char*>(d)+n,s,n);
    // Emulate a second rank with the same shape but a completely different clock.
    auto second=reinterpret_cast<uint64_t*>(static_cast<char*>(d)+n);
    second[0]=13;
    for (int i=1; i<n/8; i+=kWords) { second[i+Rank]=1; second[i+Entry]+=1000000000ULL;
        for(unsigned w=1;w<StageRows;++w) if(second[i+Valid]&(1ULL<<w)) second[i+w]+=1000000000ULL; }
    return 0;
}
''')
    (tmp_path/'test.cpp').write_text(r'''
#include "profile_samples.hpp"
int main(int argc,char** argv) {
    using namespace CombineProfile;
    llbench::Options o; o.world=2; o.device=12; o.profile=2; o.iterations=3; o.output=argv[1];
    std::filesystem::create_directories(o.output);
    llbench::ProfileSamples samples;
    for(unsigned i=0;i<3;++i) {
        std::vector<uint64_t> data(2*kWords);
        for(unsigned c=0;c<2;++c) {
            auto r=data.data()+c*kWords;
            r[Entry]=(1ULL<<60)+c*100; r[FlushDone]=r[Entry]+1000+i*100+c;
            r[Valid]=(1ULL<<Entry)|(1ULL<<FlushDone); r[Schema]=kSchema; r[Complete]=1;
            r[World]=2;r[Core]=c;r[Tokens]=32;r[Level]=2;
        }
        samples.capture(data,2,1);
    }
    llbench::Coordinator peers;
    samples.write(peers,o,32,2,1,{30,10,20},"build-test");
}
''')
    exe = tmp_path/'test.exe'
    subprocess.run([zig, 'c++', '-std=c++17', '-O1', '-I', str(tmp_path), '-I', str(HERE),
                    str(tmp_path/'test.cpp'), '-o', str(exe)], check=True, timeout=180)
    output = tmp_path/'output'
    subprocess.run([str(exe), str(output)], check=True, timeout=30)
    traces = sorted((output/'perfetto').glob('*.json'))
    assert [p.name for p in traces] == ['bs32-p50-i2.perfetto.json', 'bs32-worst-i0.perfetto.json']
    median = read_trace(traces[0])
    assert median['otherData']['selection'] == 'global rank-MAX p50'
    bodies = [e for e in median['traceEvents'] if e['name'] == 'profiled body']
    assert len(bodies) == 4 and {e['pid'] for e in bodies} == {1,2}
    assert bodies[0]['dur'] == bodies[2]['dur'] == pytest.approx(1.2)
