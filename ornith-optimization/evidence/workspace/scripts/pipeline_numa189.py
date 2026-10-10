"""After187 fully passes, freeze its source and measure unchanged profiles with NUMA."""
import copy
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from qualified_profile189 import freeze, sha

ROOT = Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
PARENT, BIRTH = 825019, '231008717'
RUN = 'workspace187_numa189'
RUNNER_SHA = '01f36917804c07758408f7ffd0c789aebe47c612ea2c4491e7cec19baecd5fcb'
PLAN_SHA = '830e3c10cc1f47865ed533ee6ed858e8c8aef5b8c3f942379ceb36eb79d59470'
PROBE_SHA = 'cf3b46384406cf24d14433a8810f33a93caa46e6b62d7865830b692211b9445f'


def process(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return fields[0], fields[19]
    except FileNotFoundError:
        return 'missing', None


def record(status, **values):
    row = dict(status=status, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values)
    (ROOT / 'results/numa189-status.json').write_text(json.dumps(row, indent=2))
    print(json.dumps(row), flush=True)


def rows(path):
    if not path.exists():
        return []
    values = []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise
    return values


def locality(log):
    text = log.read_text(errors='replace') if log.exists() else ''
    result = []
    for rank in [0, 1]:
        pids = re.findall(r'\(Worker_TP' + str(rank) + r' pid=(\d+)\)', text)
        if not pids:
            return None
        pid = int(pids[-1])
        if process(pid)[0] in ['missing', 'Z', 'X']:
            return None
        proc = Path(f'/proc/{pid}')
        status = {line.split(':', 1)[0]: line.split(':', 1)[1].strip()
                  for line in (proc / 'status').read_text().splitlines() if ':' in line}
        expected = Path(f'/sys/devices/system/node/node{rank}/cpulist').read_text().strip()
        env = dict(line.split('=', 1) for line in (proc / 'environ').read_bytes().decode().split('\0') if '=' in line)
        bound_pages, remote_pages, anon_pages, policies = 0, 0, 0, set()
        for line in (proc / 'numa_maps').read_text().splitlines():
            if not re.search(r'\banon=\d+', line):
                continue
            parts = line.split()
            policies.add(parts[1])
            pages = {int(node): int(count) for node, count in re.findall(r'\bN(\d+)=(\d+)', line)}
            anon_pages += sum(pages.values())
            if parts[1] == f'bind:{rank}':
                bound_pages += pages.get(rank, 0)
                remote_pages += sum(count for node, count in pages.items() if node != rank)
        # Mems_allowed is the cpuset permission mask, not numactl's memory policy.
        # The real mappings must report bind:<rank> and have no remote pages.
        passed = (status['Cpus_allowed_list'] == expected
                  and env.get('VLLM_WORKER_MULTIPROC_METHOD') == 'spawn'
                  and bound_pages >= 4096 and remote_pages == 0)
        result.append(dict(rank=rank, pid=pid, birth=process(pid)[1], cpus=status['Cpus_allowed_list'],
                           mems=status['Mems_allowed_list'], expected_cpus=expected,
                           worker_method=env.get('VLLM_WORKER_MULTIPROC_METHOD'),
                           bound_anon_pages=bound_pages, remote_pages_in_bound_anon=remote_pages,
                           all_anon_pages=anon_pages, anonymous_policies=sorted(policies), passed=passed))
    return result


def main():
    parent_plan_path = ROOT / 'scripts/workspace187-plan.json'
    assert sha(parent_plan_path) == PLAN_SHA
    probe_path = ROOT / 'results/numa183-cpu-probe.json'
    assert sha(probe_path) == PROBE_SHA
    probe = json.loads(probe_path.read_text())
    assert probe['status'] == 'numa183_cpu_cli_and_worker_bind_passed' and not probe['cuda_initialized']
    assert probe['probe_sha256'] == sha(ROOT / 'scripts/probe_numa183.py')
    assert sha(ROOT / 'scripts/onecat_qualified_runner139.py') == RUNNER_SHA
    assert not (ROOT / 'results' / f'{RUN}.jsonl').exists()
    record('waiting_terminal_restored187', parent_pid=PARENT, parent_birth=BIRTH,
           no_gpu_work=True, source_mutation_performed=False)
    deadline = time.monotonic() + 24 * 3600
    while process(PARENT)[0] not in ['missing', 'Z', 'X']:
        assert process(PARENT)[1] == BIRTH
        assert time.monotonic() < deadline
        time.sleep(10)
    parent_journal = ROOT / 'results/workspace187_high_and_full_capacity.jsonl'
    entries = rows(parent_journal)
    assert entries[-1]['kind'] == 'run_complete'
    restored = [row for row in entries if row['kind'] == 'restored'][-1]
    assert restored['verification']['passed'] and restored['verification']['context'] == 240128
    complete = {row['case'] for row in entries if row['kind'] == 'case_complete'}
    failed = {row['case'] for row in entries if row['kind'] == 'case_failed'}
    plan = json.loads(parent_plan_path.read_text())
    candidates = [case for case in plan['cases'] if case['name'] in complete - failed]
    if not candidates:
        record('skipped_no_whole_model_qualified187', no_gpu_work=True, source_mutation_performed=False)
        return
    profiles = [freeze(parent_journal, case['name']) for case in candidates]
    cases = []
    for baseline in candidates:
        case = copy.deepcopy(baseline)
        assert '--numa-bind' not in case['command']
        case['command'] += ['--numa-bind', '--numa-bind-nodes', '0', '1']
        case['env']['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
        case['name'] += '_numa189'
        case.pop('admission_group', None)
        case['numa183_prerequisite'] = dict(proof=str(probe_path), sha256=PROBE_SHA,
            baseline_case=baseline['name'], baseline_journal=str(parent_journal),
            baseline_journal_sha256=sha(parent_journal), frozen_baselines=profiles)
        cases.append(case)
    plan.update(run=RUN, cases=cases, tests=copy.deepcopy(cases[0]['tests']), wait_for_gpu_lock=False)
    plan['candidate_source_sha256'].update(probe['source_sha256'])
    for relative, pin in plan['candidate_source_sha256'].items():
        assert sha(Path(cases[0]['cwd']) / relative) == pin
    plan['numa189_scope'] = 'Only rank-local CPU/memory binding withspawn;same source187/KV/batch/graphs/fullcapacity/MM/mixed/cold16. Actual worker/proc locality checked atcase_ready. Nohardware changes.'
    path = ROOT / 'scripts/numa189-plan.json'
    assert not path.exists()
    path.write_text(json.dumps(plan, indent=2))
    record('running_actual_numa189', plan=str(path), plan_sha256=sha(path), baseline_profiles=profiles,
           cases=[case['name'] for case in cases], source_mutation_performed=False)
    runner = subprocess.Popen([sys.executable, str(ROOT / 'scripts/onecat_qualified_runner139.py'), str(path)],
                              cwd=ROOT / 'scripts')
    observations = {}
    observation_path = ROOT / 'results/numa189-worker-locality.json'
    journal = ROOT / 'results' / f'{RUN}.jsonl'
    while runner.poll() is None:
        current = rows(journal)
        for case in cases:
            name = case['name']
            if name in observations or not any(row.get('case') == name and row['kind'] == 'case_ready' for row in current):
                continue
            checks = locality(ROOT / 'logs' / f'{RUN}_{name}.log')
            if checks is not None:
                observations[name] = dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), workers=checks,
                                           passed=all(check['passed'] for check in checks))
                observation_path.write_text(json.dumps(observations, indent=2))
                print(json.dumps(dict(status='actual_numa_workers_observed', case=name, **observations[name])), flush=True)
        time.sleep(5)
    current = rows(journal)
    assert current[-1]['kind'] == 'run_complete'
    assert any(row['kind'] == 'restored' and row['verification']['passed'] for row in current)
    qualified = current[-1]['qualified_cases']
    passed = bool(qualified) and all(observations.get(name, {}).get('passed', False) for name in qualified)
    record('complete' if passed else 'failed_qualification_or_locality', qualified_cases=qualified,
           model_gates_passed=current[-1]['passed'], actual_locality_passed=passed,
           locality_proof=str(observation_path) if observation_path.exists() else None,
           runner_returncode=runner.returncode, baseline_profiles=profiles, source_mutation_performed=False)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        record('failed', error_type=type(error).__name__, error=str(error), source_mutation_performed=False)
        raise
