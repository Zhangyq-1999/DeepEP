"""Check the pinned source identity without importing an SDK or using a device."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_submodule_gitlink_matches_dependency_lock():
    if not (ROOT / '.git').exists() or not shutil.which('git'):
        pytest.skip('Git source checkout required')
    lock = json.loads((ROOT / 'dependencies.lock.json').read_text(encoding='utf-8'))['shmem']
    entry = subprocess.check_output(['git', 'ls-files', '--stage', '--', 'third-party/shmem'],
                                    cwd=ROOT, text=True).strip().split()
    assert entry == ['160000', lock['revision'], '0', 'third-party/shmem']
    assert lock['repository'] == 'https://gitcode.com/cann/shmem'
    assert lock['ref'] == 'refs/merge-requests/1000/head'


def test_initialized_shmem_source_and_public_headers_match_lock():
    source = Path(os.environ.get('DEEPEP_TEST_SHMEM_SOURCE', ROOT / 'third-party/shmem'))
    if not (source / '.git').exists() or not shutil.which('git'):
        pytest.skip('Initialize only third-party/shmem to check source headers')
    lock = json.loads((ROOT / 'dependencies.lock.json').read_text(encoding='utf-8'))['shmem']
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    assert revision == lock['revision']
    for relative, expected in lock['header_sha256_lf'].items():
        data = (source / 'include' / relative).read_bytes().replace(b'\r\n', b'\n')
        assert hashlib.sha256(data).hexdigest() == expected, relative
