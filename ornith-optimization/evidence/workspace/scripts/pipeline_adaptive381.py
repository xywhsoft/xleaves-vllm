"""Whole Ornith adaptive sampler validation, original service restored by139."""
import argparse
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from deploy_verified380 import restored_runs
from final_jobs286 import REQUIRED, check_job, terminal_jobs
from manage_onecat71 import api_key, check_idle, discover
from pipeline_layer216 import process
from qualified_profile189 import audit, sha

ROOT = Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
TARGET = ROOT / 'src/onecat-padded-kv140'
REL = 'vllm/v1/sample/ops/topk_topp_triton.py'
BASE = '35c2aa9437444104df8f23f3c1b6abe2efaa6e6909257e6a7cdebd4aeb8b3dd3'
NEW = '7e1e04a7af6badc8012cfbcae920495672a57d3a30b07d547adc82f8d8177c52'
RUN = 'adaptive_sampler381'
RUNNER = ROOT / 'scripts/onecat_qualified_runner139.py'
PLAN = ROOT / 'scripts/adaptive381-plan.json'
JOURNAL = ROOT / 'results' / f'{RUN}.jsonl'
PROOF = ROOT / 'results/adaptive381-whole-model-proof.json'


def rows():
    if not JOURNAL.exists():
        return []
    return [json.loads(x) for x in JOURNAL.read_text().splitlines()]


def record(status, **values):
    value = dict(status=status, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values)
    (ROOT / 'results/adaptive381-status.json').write_text(json.dumps(value, indent=2))
    print(json.dumps(value), flush=True)


def prerequisites():
    assert sha(TARGET / REL) == BASE
    assert sha(ROOT / 'scripts/topk_topp_adaptive370_candidate.py') == NEW
    assert sha(RUNNER) == '01f36917804c07758408f7ffd0c789aebe47c612ea2c4491e7cec19baecd5fcb'
    assert sha(ROOT / 'scripts/cache210-plan.json') == '714d2b3e8214f6ee85894149164bc14948550003383810b8e03d64b6bf854ecc'
    proof = ROOT / 'results/sampler374-gpu-operator-proof.json'
    assert sha(proof) == '81cf8f705a628c0bbc7af256597947629cd56932424128523d43c9fb7910c569'
    gpu = json.loads(proof.read_text())
    assert gpu['status'] == 'sampler374_gpu_operator_checks_passed' and gpu['failure'] is None
    assert len(gpu['completed']) == 8 and all(x['ratio'] <= 1.05 for x in gpu['fast_latency_ratios'])
    for entry in gpu['completed']:
        assert sha(entry['output']) == entry['sha256'] and sha(entry['log']) == entry['log_sha256']
    cpu = ROOT / 'results/sampler371-adaptive-cpu-bitwise-proof.json'
    assert sha(cpu) == 'a00b62d937b538844b4d14850eb82e5564f5827c87fcf2cd9aed1ebaaac2e95d'
    value = json.loads(cpu.read_text())
    assert len(value['checks']) == 360 and not value['cuda_initialized'] and not value['serving_source_mutation']
    profiles = []
    for name, pin in (
        ('qualified-guard-high318-final.json', '8e47d246a63a873ed34c92665fd9616a1a05ce716b87d332557d55aee0740b25'),
        ('qualified-guard-full24_342-final.json', '3462d297f46abbab9b06eb837963ffbf82a5c30266d7a07d9ea21d781dc6919a'),
        ('qualified-guard-full26_352-final.json', '81bf07eb086b314b8ba3ba9510dabc6d5a79d56cda45bde858da6206c2b1cd6d'),
    ):
        path = ROOT / 'results' / name
        assert sha(path) == pin
        audit(json.loads(path.read_text()))
        profiles.append(dict(path=str(path), sha256=pin))
    cache = json.loads((ROOT / 'scripts/cache210-plan.json').read_text())
    for relative, pin in cache['candidate_source_sha256'].items():
        assert sha(TARGET / relative) == pin
    return dict(cpu=str(cpu), cpu_sha256=sha(cpu), gpu=str(proof), gpu_sha256=sha(proof),
        candidate=str(ROOT / 'scripts/topk_topp_adaptive370_candidate.py'), candidate_sha256=NEW,
        baseline_sha256=BASE, serving_relative_path=REL, frozen_parents=profiles,
        checker=str(ROOT / 'scripts/test_sampler373_gpu.py'), checker_sha256=sha(ROOT / 'scripts/test_sampler373_gpu.py'))


def owner_and_others():
    path = ROOT / 'cache/adaptive381-job.json'
    deadline = time.monotonic() + 15
    while not path.exists():
        assert time.monotonic() < deadline
        time.sleep(.05)
    own = json.loads(path.read_text())
    state, birth = process(os.getpid())
    assert state not in ('missing', 'Z', 'X') and own['pid'] == os.getpid() and str(own['birth']) == birth
    assert own['script'] == str(Path(__file__).resolve()) and own['script_sha256'] == sha(__file__)
    assert own['resource'] == 'gpu_exclusive'
    names = set(REQUIRED)
    for descriptor in (ROOT / 'cache').glob('*-job.json'):
        job = json.loads(descriptor.read_text())
        if job.get('resource') == 'gpu_exclusive':
            names.add(descriptor.name.removesuffix('-job.json'))
    return [check_job(name) for name in sorted(names) if name != 'adaptive381'], sha(path)


def make_plan(requirement):
    cache = json.loads((ROOT / 'scripts/cache210-plan.json').read_text())
    boundary = json.loads((ROOT / 'scripts/boundary200-plan.json').read_text())
    high = next(c for c in cache['cases'] if c['minimum_full_contexts'] == 16)
    full26 = next(c for c in cache['cases'] if c['minimum_full_contexts'] == 26)
    alternatives = [c for c in boundary['cases'] if c['diverse_count'] in (176, 160)] + [high]
    assert [c['diverse_count'] for c in alternatives] == [176, 160, 144]
    cases = []
    for original in alternatives + [full26]:
        case = copy.deepcopy(original)
        case['name'] += '_adaptive381'
        case['admission_group'] = 'adaptive381_high' if case['diverse_count'] >= 144 else 'adaptive381_full26'
        case['guard209_prerequisite'] = copy.deepcopy(high['guard209_prerequisite'] if case['diverse_count'] >= 144 else full26['guard209_prerequisite'])
        case['adaptive370_prerequisite'] = copy.deepcopy(requirement)
        assert case['command'][case['command'].index('--max-model-len') + 1] == '240128'
        assert case['minimum_full_contexts'] >= 16 and case['required_cached_tokens'] == 236944
        assert case['image_checks'] and case['require_fresh_images_after_long'] and case['cold_concurrent_inputs']
        cases.append(case)
    plan = copy.deepcopy(cache)
    plan.update(run=RUN, cases=cases, tests=copy.deepcopy(cases[0]['tests']), wait_for_gpu_lock=False)
    plan['candidate_source_sha256'][REL] = NEW
    plan['adaptive370_scope'] = 'Only sampler source changes from current guard-qualified source. High176/160/144 fallback plus full26 must each pass applicable whole-model gates. Failed measurements excluded; source retention is for final review, not public deployment.'
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    requirement = prerequisites()
    previous = restored_runs()
    if args.validate_only:
        terminal_jobs()
        plan = make_plan(requirement)
        print(json.dumps(dict(status='adaptive381_cpu_inputs_checked_only', cases=[c['name'] for c in plan['cases']],
            script_sha256=sha(__file__), torch_imported='torch' in sys.modules, source_mutation_performed=False)))
        return
    assert not any(p.exists() for p in (PLAN, JOURNAL, PROOF))
    jobs, descriptor_sha = owner_and_others()
    lock = (ROOT / 'cache/gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    found = discover()
    assert found is not None
    check_idle(found, api_key(found[1], found[2]))
    backup = ROOT / 'scripts/baseline_sampler_adaptive381.py'
    assert not backup.exists()
    backup.write_bytes((TARGET / REL).read_bytes())
    installed, retained, child = False, False, None
    qualified, failure, hooks, restored = [], None, None, None

    def cancel(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    try:
        (TARGET / REL).write_bytes((ROOT / 'scripts/topk_topp_adaptive370_candidate.py').read_bytes())
        installed = True
        record('running_source_hooks', source_mutation_performed=True)
        environment = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PRE_COMMIT_HOME': str(ROOT / 'cache/pre-commit'),
            'UV_CACHE_DIR': str(ROOT / 'cache/uv'), 'GOPROXY': 'https://goproxy.cn,direct'}
        log = ROOT / 'logs/adaptive381-source-hooks.log'
        result = subprocess.run([str(TARGET / '.venv/bin/python'), '-m', 'pre_commit', 'run', '--files', REL],
            cwd=TARGET, env=environment, capture_output=True, text=True, timeout=300)
        log.write_text(result.stdout + result.stderr)
        hooks = dict(passed=result.returncode == 0 and sha(TARGET / REL) == NEW,
            candidate_sha256=sha(TARGET / REL), log=str(log), log_sha256=sha(log), returncode=result.returncode)
        hook_path = ROOT / 'results/adaptive381-source-hooks.json'
        hook_path.write_text(json.dumps(hooks, indent=2))
        assert hooks['passed'], 'Hooks must pass without changing the exact CPU/GPU-qualified source bytes'
        subprocess.run(['git', 'diff', '--check'], cwd=TARGET, check=True)
        requirement.update(hooks=str(hook_path), hooks_sha256=sha(hook_path),
            driver=str(Path(__file__).resolve()), driver_sha256=sha(__file__), job_descriptor_sha256=descriptor_sha)
        plan = make_plan(requirement)
        for relative, pin in plan['candidate_source_sha256'].items():
            assert sha(TARGET / relative) == pin
        PLAN.write_text(json.dumps(plan, indent=2))
        record('running_whole_model_qualification', plan=str(PLAN), plan_sha256=sha(PLAN),
            cases=[c['name'] for c in plan['cases']], source_mutation_performed=True)
        fcntl.flock(lock, fcntl.LOCK_UN)
        child = subprocess.Popen([sys.executable, str(RUNNER), str(PLAN)], cwd=ROOT / 'scripts', start_new_session=True)
        child.wait(timeout=24 * 3600)
        entries = rows()
        assert entries and entries[-1]['kind'] == 'run_complete'
        restored = next(x for x in reversed(entries) if x['kind'] == 'restored')
        assert restored['verification']['passed']
        qualified = sorted({x['case'] for x in entries if x['kind'] == 'case_complete'} -
            {x['case'] for x in entries if x['kind'] == 'case_failed'})
        retained = bool(qualified)
    except BaseException as error:
        failure = dict(type=type(error).__name__, message=str(error))
    finally:
        if child is not None and child.poll() is None:
            os.kill(child.pid, signal.SIGTERM)
            child.wait(timeout=900)
        assert child is None or child.poll() is not None, 'Never mutate source while model runner remains active'
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if installed and not retained:
            (TARGET / REL).write_bytes(backup.read_bytes())
            assert sha(TARGET / REL) == BASE
        entries = rows()
        if restored is None:
            restored = next((x for x in reversed(entries) if x['kind'] == 'restored'), None)
        current = discover()
        assert current is not None and current[4], 'Original llama.cpp serving process must be restored'
        assert current[1] == found[1], 'Original command changed'
        check_idle(current, api_key(current[1], current[2]))
        public = dict(pid=current[0], original_command_reused=current[1] == found[1],
            api_key_preserved=api_key(current[1], current[2]) == api_key(found[1], found[2]))
        value = dict(status='adaptive381_whole_model_completed' if failure is None else 'adaptive381_whole_model_failed',
            script_sha256=sha(__file__), job_descriptor_sha256=descriptor_sha, qualification_prerequisite=requirement,
            terminal_prior_jobs=jobs, terminal_prior_restorations=previous, hooks=hooks, failure=failure,
            journal=str(JOURNAL) if JOURNAL.exists() else None, journal_sha256=sha(JOURNAL) if JOURNAL.exists() else None,
            qualified_cases=qualified, source_mutation_performed=installed, source_retained_for_final_review=retained,
            final_source_sha256=sha(TARGET / REL), restored=restored, current_public=public,
            public_deployment_performed=False, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        PROOF.write_text(json.dumps(value, indent=2))
        record(value['status'], qualified_cases=qualified, source_retained_for_final_review=retained,
            proof=str(PROOF), proof_sha256=sha(PROOF), failure=failure, source_mutation_performed=installed)
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    if failure is not None:
        raise RuntimeError(failure)


if __name__ == '__main__':
    main()
