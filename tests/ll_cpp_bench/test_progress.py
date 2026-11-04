"""Launch progress stays visible even while the worker blocks."""
import os
from pathlib import Path
import shutil
import subprocess

import pytest

HERE = Path(__file__).resolve().parent


def test_live_monitor_reports_stages_and_unchanged_wait(tmp_path):
    bash = "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else shutil.which("bash")
    if not bash:
        pytest.skip("Bash required")
    status = (tmp_path / "progress.txt").as_posix()
    script = r'''
set -e
bash "$1" "$2" 0 $$ 1 & monitor=$!
trap 'kill "$monitor" 2>/dev/null || true; wait "$monitor" 2>/dev/null || true' EXIT
sleep 2
printf '%s\n' 'rank=0 stage=SHMEM initialization' > "$2"
sleep 2
printf '%s\n' 'rank=0 variant=base BS=32 stage=sampling completed=10/100' > "$2"
sleep 2
'''
    result = subprocess.run([bash, "-c", script, "_", (HERE / "progress.sh").as_posix(), status],
                            capture_output=True, text=True, timeout=15, check=True)
    assert "MPI startup/rendezvous" in result.stdout
    assert "SHMEM initialization" in result.stdout
    assert "completed=10/100" in result.stdout
    assert "unchanged=" in result.stdout


def test_progress_stays_outside_event_and_before_barrier():
    source = (HERE / "main.cpp").read_text(encoding="utf-8")
    sample = source.split("for (uint32_t i = 0; i < o.iterations; ++i)", 1)[1]
    assert sample.index('session.report("sampling"') < sample.index("entry.enqueue(") < sample.index("start.record(stream);")
    timed = sample.split("start.record(stream);", 1)[1].split("end.record(stream);", 1)[0]
    assert "report(" not in timed and "launch();" in timed
    assert 'entry.complete("sampling", bs, i)' in sample
