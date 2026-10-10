"""Final deployment including registered GPU jobs and sampler restoration."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import time

from deploy_verified214 import ROOT, invoke, manager, validate
from pipeline_numa189 import process
from profile_locality201 import checked_locality
from qualified_profile189 import sha
from query_numa_anon221 import inspect
from final_jobs286 import terminal_jobs
from deploy_verified228 import restored_runs as original_restored_runs

READER = '765e24f4c974f6eb95bd43f356866449f08baa7dfa8aacfad9ba3a932d2c48cf'
CPU = '29eb36ccd2ba941182c4923ecca822d99e18b330241982b426ccdb5b6fe2c27e'
JOBS = ('workspace187', 'numa189', 'boundary200', 'cache210', 'layer216',
        'numa223-observer', 'numa226-freezer')


def restored_runs():
    result = original_restored_runs()
    path = ROOT / 'results/sampler260-gpu-operator-proof.json'
    value = json.loads(path.read_text())
    assert value['status'] in ('sampler260_gpu_operator_checks_passed', 'sampler260_gpu_operator_checks_failed')
    assert value['script_sha256'] == sha(ROOT / 'scripts/pipeline_sampler260.py')
    assert not value['source_mutation_performed']
    restored = value['restored']
    assert restored['verification']['passed'] and restored['original_command_reused'] and restored['api_key_preserved']
    assert sha(ROOT / 'scripts/final_jobs286.py') == 'd26eff531724378de022241da9aa06f43dae1655d232626e63206d5bd9b8df76'
    result.append(dict(journal=str(path), sha256=sha(path), operator_status=value['status'], restored=restored))
    return result


def reader_proof():
    path = ROOT / 'results/numa222-query-cpu-proof.json'
    assert sha(ROOT / 'scripts/query_numa_anon221.py') == READER
    assert sha(path) == CPU
    value = json.loads(path.read_text())
    assert value['reader_sha256'] == READER
    assert sha(ROOT / 'scripts/test_numa_query222.py') == value['script_sha256']
    assert value['status'] == 'numa222_cpu_mixed_mapping_and_real_remote_negative_passed'
    positive, negative = value['fixtures']
    assert positive['passed'] and positive['remote_anon'] == 0
    assert negative['negative'] and not negative['passed'] and negative['remote_anon'] >= 4096
    return dict(reader_sha256=READER, cpu_proof=str(path), cpu_proof_sha256=CPU)


def public_locality(profile, snapshot, public_pid):
    proof = reader_proof()
    text = snapshot.read_text(errors='replace')
    workers = []
    for rank in (0, 1):
        pids = re.findall(r'\(Worker_TP' + str(rank) + r' pid=(\d+)\)', text)
        assert pids, f'No public rank {rank} PID in fresh log'
        pid = int(pids[-1])
        state, birth = process(pid)
        assert state not in ['missing', 'Z', 'X'] and birth is not None
        assert os.getpgid(pid) == public_pid, 'Worker is outside deployed public process group'
        worker = inspect(pid, rank, birth)
        assert worker['passed'] and worker['ambiguous_anonymous_pages'] == 0
        assert worker['query_nodes_argument'] is None and worker['query_flags'] == 0
        assert os.getpgid(pid) == public_pid
        workers.append(worker)
    checked = checked_locality(profile, dict(passed=True, workers=workers))
    return dict(**checked, **proof, log_snapshot=str(snapshot), log_snapshot_sha256=sha(snapshot))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    checked = validate(profile)
    proof = reader_proof()
    if args.validate_only:
        print(json.dumps(dict(status='deployment297_profile_audited_only', case=profile['case'], reader_proof=proof, **checked)), flush=True)
        return
    assert args.output is not None and not args.output.exists()
    terminal_jobs()
    lock = (ROOT / 'cache/gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        jobs = terminal_jobs()
        runs = restored_runs()
        checked = validate(profile)
        before = manager.discover()
        assert before is not None
        manager.check_idle(before, manager.api_key(before[1], before[2]))
        log = ROOT / 'logs/onecat-public71.log'
        offset = log.stat().st_size if log.exists() else 0
        invoke([str(args.profile)])
        try:
            current = manager.discover()
            assert current is not None and not current[4]
            result_path = ROOT / 'results/onecat-public-deployment71.json'
            result = json.loads(result_path.read_text())
            assert result['status'] == 'deployed' and result['case'] == profile['case']
            assert result['pid'] == current[0] and result['smoke']['passed']
            observed = dict(enabled=False)
            if checked['command_locality']['enabled']:
                snapshot = ROOT / 'logs' / f'public297-{current[0]}-workers.log'
                assert not snapshot.exists()
                with log.open('rb') as stream:
                    stream.seek(offset)
                    snapshot.write_bytes(stream.read())
                observed = public_locality(profile, snapshot, current[0])
            record = dict(status='verified_public_deployment297', utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                profile=str(args.profile), profile_sha256=sha(args.profile), case=profile['case'], pid=current[0],
                port=8096, api_key_preserved=result['api_key_preserved'], smoke=result['smoke'],
                public_deployment_proof=dict(path=str(result_path), sha256=sha(result_path)),
                actual_public_locality=observed, reader_proof=proof, terminal_optimization_jobs=jobs,
                terminal_restored_runs=runs, registered_jobs_barrier_sha256=sha(ROOT / 'scripts/final_jobs286.py'), wrapper_sha256=sha(Path(__file__).resolve()), **checked)
            args.output.write_text(json.dumps(record, indent=2))
            print(json.dumps(dict(status=record['status'], output=str(args.output), sha256=sha(args.output), case=profile['case'], pid=current[0])), flush=True)
        except BaseException:
            invoke(['--rollback'])
            raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
