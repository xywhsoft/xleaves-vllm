"""Run inactive sampler GPU checks after the immutable model/baseline queue."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import manage_onecat71 as manager
from deploy_verified297 import restored_runs
from test_sampler373_gpu import prepare
from pipeline_layer216 import process
from qualified_profile189 import sha

ROOT = Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
TARGET = ROOT / 'src/onecat-padded-kv140'
CHECKER = ROOT / 'scripts/test_sampler373_gpu.py'
PINS = {
    CHECKER: '7ee48c3873903c0d3076d7ee8da9830947a7a159b454643eaa26ebc82bd4e435',
    ROOT / 'scripts/topk_topp_adaptive370_candidate.py': '7e1e04a7af6badc8012cfbcae920495672a57d3a30b07d547adc82f8d8177c52',
    ROOT / 'results/sampler371-adaptive-cpu-bitwise-proof.json': 'a00b62d937b538844b4d14850eb82e5564f5827c87fcf2cd9aed1ebaaac2e95d',
    TARGET / 'vllm/v1/sample/ops/topk_topp_triton.py': '35c2aa9437444104df8f23f3c1b6abe2efaa6e6909257e6a7cdebd4aeb8b3dd3',
    TARGET / 'vllm/v1/sample/ops/topk_topp_sampler.py': 'ab2ee9e52cc229ced835e840cb90cea3a1d7ed2d927d86507b670b532a56d52f',
}
RUN = 'sampler374'


def inputs():
    result = []
    for path, pin in PINS.items():
        assert sha(path) == pin, f'Changed prerequisite {path.name}'
        result.append(dict(path=str(path), sha256=pin))
    return result


def record(status, **values):
    value = dict(status=status, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values)
    (ROOT / 'results/sampler374-status.json').write_text(json.dumps(value, indent=2))
    print(json.dumps(value), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    pins = inputs()
    if args.validate_only:
        prepare()
        print(json.dumps(dict(status='sampler374_inputs_syntax_checked_only', pins=pins,
            script_sha256=sha(__file__), torch_imported='torch' in sys.modules,
            source_mutation_performed=False, gpu_tests_run=False)), flush=True)
        return
    proof_path = ROOT / 'results/sampler374-gpu-operator-proof.json'
    assert not proof_path.exists()
    descriptor = ROOT / 'cache/sampler374-job.json'
    deadline = time.monotonic() + 15
    while not descriptor.exists():
        assert time.monotonic() < deadline, 'Missing own registered GPU descriptor'
        time.sleep(0.05)
    own = json.loads(descriptor.read_text())
    state, birth = process(os.getpid())
    assert own['pid'] == os.getpid() and str(own['birth']) == birth
    assert own['script_sha256'] == sha(__file__) and own['resource'] == 'gpu_exclusive'
    jobs, runs = prepare(os.getpid()), restored_runs()
    pins = inputs()
    lock = (ROOT / 'cache/gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    found = manager.discover()
    assert found is not None, 'No restorable public service'
    key = manager.api_key(found[1], found[2])
    manager.check_idle(found, key)
    paused, restored, completed = False, None, []
    failure = None
    child = None

    def cancel(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    try:
        record('running_idle_gpu_operator_checks', terminal_jobs=jobs, restored_runs=runs,
            source_mutation_performed=False)
        try:
            manager.stop(found[0])
            paused = True
        except BaseException:
            paused = not manager.alive(found[0])
            raise
        for device in (0, 1):
            for mode, variant, force_cache in (('numeric', None, False), ('numeric', None, True), ('pressure', 'baseline', False), ('pressure', 'candidate', False)):
                suffix = f'{device}-{mode}' + ('-cached' if force_cache else '-fast' if mode == 'numeric' else f'-{variant}')
                output = ROOT / 'results' / f'sampler374-{suffix}.json'
                log = ROOT / 'logs' / f'sampler374-{suffix}.log'
                assert not output.exists() and not log.exists(), 'Use a fresh run for existing outputs'
                command = [sys.executable, str(CHECKER), '--device', str(device), '--mode', mode,
                    '--lock-fd', str(lock.fileno()), '--output', str(output)]
                if force_cache:
                    command += ['--force-cache']
                if variant:
                    command += ['--variant', variant]
                with log.open('x') as stream:
                    child = subprocess.Popen(command, cwd=ROOT / 'scripts',
                        env={**os.environ, 'CUDA_VISIBLE_DEVICES':'0,1'},
                        pass_fds=(lock.fileno(),), stdout=stream, stderr=subprocess.STDOUT,
                        start_new_session=True)
                    code = child.wait(timeout=1200)
                child = None
                assert code == 0, f'Sampler checker failed {suffix}; see {log.name}'
                value = json.loads(output.read_text())
                assert value['status'] == f'sampler373_gpu_{mode}_completed'
                assert value['script_sha256'] == PINS[CHECKER] and value['vocab'] == 248320
                assert not value['source_mutation_performed'] and value['model_requests_sent'] == 0
                assert value['force_cache'] == force_cache
                if mode == 'numeric':
                    assert value['checks'] and value['measurements']
                    assert all(x.get('output_bitwise_equal', x.get('filtered_logits_bitwise_equal')) for x in value['checks'])
                completed.append(dict(device=device, mode=mode, variant=variant, force_cache=force_cache, output=str(output),
                    sha256=sha(output), log=str(log), log_sha256=sha(log), returncode=code))
                inputs()
                record('running_idle_gpu_operator_checks', completed=completed,
                    source_mutation_performed=False, gpu_tests_run=True)
        assert len(completed) == 8
        fast_latency = []
        for entry in completed:
            value = json.loads(Path(entry['output']).read_text())
            if entry['mode'] == 'numeric' and not entry['force_cache']:
                for count in (16, 160, 176):
                    before = next(x for x in value['measurements'] if x['batch'] == count and x['variant'] == 'baseline')
                    after = next(x for x in value['measurements'] if x['batch'] == count and x['variant'] == 'candidate')
                    ratio = after['gpu_ms_median'] / before['gpu_ms_median']
                    fast_latency.append(dict(device=entry['device'], batch=count, ratio=ratio))
                    assert ratio <= 1.05, f'Fast route latency gate failed: {ratio}'
            if entry['mode'] == 'pressure' and entry['variant'] == 'candidate':
                samples = value['measurements'][0]['samples']
                assert len(samples) == 3 and all(not x['oom'] and x['output_bitwise_equal'] for x in samples)
                assert len(samples[0]['actual_ooms']) == 1
                assert all(not x['actual_ooms'] for x in samples[1:])
    except BaseException as error:
        failure = dict(type=type(error).__name__, error=str(error).replace(key, '[redacted]') if key else str(error))
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if child is not None and child.poll() is None:
            manager.stop(child.pid, own_group=True)
            child.wait(timeout=30)
        try:
            if paused:
                public = manager.launch(found[1], found[2], found[3], ROOT / 'logs/sampler374-restore.log')
                startup = manager.ready(public, key, llama=found[4])
                if found[4]:
                    slots = manager.http('/slots', key)
                    assert len(slots) >= 16 and {s.get('n_ctx') for s in slots} == {240128}
                    verification = dict(passed=True, slots=len(slots), context=240128)
                else:
                    verification = manager.smoke(key)
                restored = dict(pid=public.pid, startup_s=startup, verification=verification,
                    original_command_reused=True, api_key_preserved=bool(key))
            inputs()
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
    assert restored and restored['verification']['passed'], 'Public service restoration proof is required'
    proof = dict(status='sampler374_gpu_operator_checks_passed' if failure is None else 'sampler374_gpu_operator_checks_failed',
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), script_sha256=sha(__file__),
        pins=pins, terminal_jobs=jobs, restored_runs=runs, completed=completed, failure=failure,
        job_descriptor_sha256=sha(descriptor), fast_latency_ratios=locals().get('fast_latency'),
        restored=restored, source_mutation_performed=False, whole_model_qualified=False,
        scope='Two V100 adaptive sampler operators: fast and forced-cache strict outputs/RNG/graph, fast median latency gate, actual first-OOM and cached pressure replay. No model TPS or whole-service capacity qualification.')
    assert not proof_path.exists()
    proof_path.write_text(json.dumps(proof, indent=2))
    record(proof['status'], proof=str(proof_path), proof_sha256=sha(proof_path),
        source_mutation_performed=False, restored=restored)
    if failure:
        raise SystemExit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        record('sampler374_failed_before_or_during_checks', error_type=type(error).__name__, error=str(error),
            source_mutation_performed=False)
        raise
