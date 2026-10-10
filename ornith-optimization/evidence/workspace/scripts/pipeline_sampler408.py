"""Wait for actual393 restoration, then test inactive noise400 on both V100s."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import manage_onecat71 as manager
from pipeline_layer216 import process
from qualified_profile189 import sha
from test_sampler406_gpu import prerequisites, parent_restoration

ROOT = Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
TARGET = ROOT / 'src/onecat-padded-kv140'
CHECKER = ROOT / 'scripts/test_sampler406_gpu.py'
PINS = {
    CHECKER: '3142ffa84033b75b58d54008f55e5b64557ca915c6ffce01ddb7a0f17213d09d',
    ROOT / 'scripts/topk_topp_sampler400_candidate.py': 'afa1e2cb242649c33c42b58d02709dc3984846d7ea0e89417bd05024148f4322',
    ROOT / 'results/sampler401-noise-cpu-contract-proof.json': 'bccd8ac11e89bda9f7bfc03b3164f4a017a2d09bc584d23d726d655f87fae76b',
    TARGET / 'vllm/v1/sample/ops/topk_topp_sampler.py': 'ab2ee9e52cc229ced835e840cb90cea3a1d7ed2d927d86507b670b532a56d52f',
    TARGET / 'vllm/v1/sample/sampler.py': '71058953620a12947b294a76379446497266bb7e5f4e02d294d28da4e15b5f91',
}


def inputs():
    for path, pin in PINS.items():
        assert sha(path) == pin, f'Changed input: {path.name}'
    return [dict(path=str(p), sha256=s) for p, s in PINS.items()]


def record(status, **values):
    value = dict(status=status, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values)
    (ROOT / 'results/sampler408-status.json').write_text(json.dumps(value, indent=2))
    print(json.dumps(value), flush=True)


def main():
    pins = inputs()
    descriptor = ROOT / 'cache/sampler408-job.json'
    proof_path = ROOT / 'results/sampler408-gpu-operator-proof.json'
    assert not proof_path.exists()
    deadline = time.monotonic() + 15
    while not descriptor.exists():
        assert time.monotonic() < deadline
        time.sleep(.05)
    own = json.loads(descriptor.read_text())
    state, birth = process(os.getpid())
    assert own['pid'] == os.getpid() and str(own['birth']) == birth
    assert own['script_sha256'] == sha(__file__) and own['resource'] == 'gpu_exclusive'
    parent_descriptor = ROOT / 'cache/adaptive393-job.json'
    assert sha(parent_descriptor) == 'b4d778e61c809bd9d30f907af72267b6d20535aa155ca6184b1f660e11bfdb7b'
    parent = json.loads(parent_descriptor.read_text())
    record('waiting_actual393_terminal_and_restoration', parent_pid=parent['pid'], parent_birth=parent['birth'],
        torch_imported='torch' in sys.modules, serving_source_mutation=False, gpu_tests_run=False)
    deadline = time.monotonic() + 24 * 3600
    while True:
        state, birth = process(parent['pid'])
        if state in ('missing', 'Z', 'X') or birth != str(parent['birth']):
            break
        assert time.monotonic() < deadline, 'Parent393 wait deadline'
        time.sleep(10)
    parent_restoration()
    checked = prerequisites(os.getpid())
    inputs()
    lock = (ROOT / 'cache/gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    found = manager.discover()
    assert found is not None, 'Restorable public service required'
    key = manager.api_key(found[1], found[2])
    manager.check_idle(found, key)
    paused, restored, completed, failure, child = False, None, [], None, None
    ratios = []

    def cancel(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    try:
        record('running_idle_gpu_noise400_contracts', prerequisites=checked, serving_source_mutation=False)
        try:
            manager.stop(found[0])
            paused = True
        except BaseException:
            paused = not manager.alive(found[0])
            raise
        for device in (0, 1):
            for mode, variant in (('numeric', None), ('pressure', 'baseline'), ('pressure', 'candidate')):
                suffix = f'{device}-{mode}' + (f'-{variant}' if variant else '')
                output = ROOT / 'results' / f'sampler408-{suffix}.json'
                log = ROOT / 'logs' / f'sampler408-{suffix}.log'
                assert not output.exists() and not log.exists()
                command = [sys.executable, str(CHECKER), '--device', str(device), '--mode', mode,
                    '--lock-fd', str(lock.fileno()), '--output', str(output)]
                if variant:
                    command += ['--variant', variant]
                with log.open('x') as stream:
                    child = subprocess.Popen(command, cwd=ROOT / 'scripts',
                        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '0,1'}, pass_fds=(lock.fileno(),),
                        stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    code = child.wait(timeout=1200)
                child = None
                assert code == 0, f'GPU contract failed: {suffix}; {log.name}'
                value = json.loads(output.read_text())
                assert value['status'] == f'sampler406_gpu_{mode}_completed'
                assert value['script_sha256'] == PINS[CHECKER] and value['vocab'] == 248320
                assert not value['serving_source_mutation'] and not value['whole_model_qualified']
                completed.append(dict(device=device, mode=mode, variant=variant, output=str(output),
                    sha256=sha(output), log=str(log), log_sha256=sha(log), returncode=code))
                inputs()
                if mode == 'numeric':
                    assert len(value['checks']) == 120
                    for count in (16, 160, 176):
                        before = next(x for x in value['measurements'] if x['batch'] == count and x['variant'] == 'baseline')
                        after = next(x for x in value['measurements'] if x['batch'] == count and x['variant'] == 'candidate')
                        ratio = after['gpu_ms_median'] / before['gpu_ms_median']
                        ratios.append(dict(device=device, batch=count, ratio=ratio))
                        assert ratio <= 1.05, f'Latency regression {device}/{count}: {ratio}'
                else:
                    samples = value['measurements'][0]['samples']
                    if variant == 'baseline':
                        assert len(samples) == 1 and samples[0]['oom'], 'Pressure must reproduce actual baseline OOM'
                    else:
                        assert len(samples) == 3 and all(not s['oom'] and s['ids_equal'] and s['rng_equal'] for s in samples)
                record('running_idle_gpu_noise400_contracts', completed=completed, latency_ratios=ratios,
                    serving_source_mutation=False, gpu_tests_run=True)
        assert len(completed) == 6
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
                public = manager.launch(found[1], found[2], found[3], ROOT / 'logs/sampler408-restore.log')
                startup = manager.ready(public, key, llama=found[4])
                if found[4]:
                    slots = manager.http('/slots', key)
                    assert len(slots) == 16 and {s.get('n_ctx') for s in slots} == {240128}
                    verification = dict(passed=True, slots=len(slots), context=240128)
                else:
                    verification = manager.smoke(key)
                restored = dict(pid=public.pid, startup_s=startup, verification=verification,
                    original_command_reused=True, api_key_preserved=bool(key))
            inputs()
            assert sha(TARGET / 'vllm/v1/sample/ops/topk_topp_triton.py') == checked['parent_restoration']['final_source_sha256']
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
            lock.close()
    assert restored and restored['verification']['passed'], 'Restoration must succeed'
    proof = dict(status='sampler408_gpu_operator_checks_passed' if failure is None else 'sampler408_gpu_operator_checks_failed',
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), script_sha256=sha(__file__), pins=pins,
        prerequisites=checked, completed=completed, failure=failure, job_descriptor_sha256=sha(descriptor),
        fast_latency_ratios=ratios, restored=restored, serving_source_mutation=False, whole_model_qualified=False)
    proof_path.write_text(json.dumps(proof, indent=2))
    record(proof['status'], proof=str(proof_path), proof_sha256=sha(proof_path), restored=restored)
    if failure:
        raise SystemExit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        record('sampler408_failed', error_type=type(error).__name__, error=str(error), serving_source_mutation=False)
        raise
