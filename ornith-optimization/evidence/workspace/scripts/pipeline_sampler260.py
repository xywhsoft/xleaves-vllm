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
from deploy_verified228 import restored_runs, terminal_jobs
from pipeline_layer216 import process
from qualified_profile189 import sha

ROOT = Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
TARGET = ROOT / 'src/onecat-padded-kv140'
CHECKER = ROOT / 'scripts/test_sampler258_gpu.py'
PINS = {
    CHECKER: 'cf9c16b6c3500560552332634a549f226f700a40312fa896ee893cd71bb9dcc6',
    ROOT / 'scripts/topk_topp_reference254_candidate.py': 'dbd0c827ebb7ca431e9cb26ceff1f8401abe11a396220023da98a61366e70922',
    ROOT / 'results/sampler255-cpu-bitwise-prerequisites.json': '57de58f126ca8cd64c6916d100942a33e3b9302288e4b424225f26429de803a8',
    TARGET / 'vllm/v1/sample/ops/topk_topp_triton.py': '35c2aa9437444104df8f23f3c1b6abe2efaa6e6909257e6a7cdebd4aeb8b3dd3',
    TARGET / 'vllm/v1/sample/ops/topk_topp_sampler.py': 'ab2ee9e52cc229ced835e840cb90cea3a1d7ed2d927d86507b670b532a56d52f',
}
RUN = 'sampler260'


def inputs():
    result = []
    for path, pin in PINS.items():
        assert sha(path) == pin, f'Changed prerequisite {path.name}'
        result.append(dict(path=str(path), sha256=pin))
    return result


def record(status, **values):
    value = dict(status=status, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values)
    (ROOT / 'results/sampler260-status.json').write_text(json.dumps(value, indent=2))
    print(json.dumps(value), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    pins = inputs()
    if args.validate_only:
        print(json.dumps(dict(status='sampler260_inputs_syntax_checked_only', pins=pins,
            script_sha256=sha(__file__), torch_imported='torch' in sys.modules,
            source_mutation_performed=False, gpu_tests_run=False)), flush=True)
        return
    proof_path = ROOT / 'results/sampler260-gpu-operator-proof.json'
    assert not proof_path.exists()
    parent = json.loads((ROOT / 'cache/layer216-job.json').read_text())
    assert sha(parent['script']) == parent['script_sha256']
    record('waiting_terminal_restored216', parent_pid=parent['pid'], parent_birth=parent['birth'],
        gpu_tests_run=False, source_mutation_performed=False)
    deadline = time.monotonic() + 72 * 3600
    while True:
        state, birth = process(parent['pid'])
        if state in ('missing', 'Z', 'X') or birth != str(parent['birth']):
            break
        assert time.monotonic() < deadline, 'Parent wait timeout'
        time.sleep(10)
    jobs, runs = terminal_jobs(), restored_runs()
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
            for mode, variant in (('numeric', None), ('pressure', 'baseline'), ('pressure', 'candidate')):
                suffix = f'{device}-{mode}' + (f'-{variant}' if variant else '')
                output = ROOT / 'results' / f'sampler260-{suffix}.json'
                log = ROOT / 'logs' / f'sampler260-{suffix}.log'
                assert not output.exists() and not log.exists(), 'Use a fresh run for existing outputs'
                command = [sys.executable, str(CHECKER), '--device', str(device), '--mode', mode,
                    '--lock-fd', str(lock.fileno()), '--output', str(output)]
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
                assert value['status'] == f'sampler258_gpu_{mode}_completed'
                assert value['script_sha256'] == PINS[CHECKER] and value['vocab'] == 248320
                assert not value['source_mutation_performed'] and value['model_requests_sent'] == 0
                if mode == 'numeric':
                    assert value['checks'] and value['measurements']
                    assert all(x.get('output_bitwise_equal', x.get('filtered_logits_bitwise_equal')) for x in value['checks'])
                completed.append(dict(device=device, mode=mode, variant=variant, output=str(output),
                    sha256=sha(output), log=str(log), log_sha256=sha(log), returncode=code))
                inputs()
                record('running_idle_gpu_operator_checks', completed=completed,
                    source_mutation_performed=False, gpu_tests_run=True)
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
                public = manager.launch(found[1], found[2], found[3], ROOT / 'logs/sampler260-restore.log')
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
    proof = dict(status='sampler260_gpu_operator_checks_passed' if failure is None else 'sampler260_gpu_operator_checks_failed',
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), script_sha256=sha(__file__),
        pins=pins, terminal_jobs=jobs, restored_runs=runs, completed=completed, failure=failure,
        restored=restored, source_mutation_performed=False, whole_model_qualified=False,
        scope='Two V100 sampler operators at full 248320 vocabulary, strict outputs and RNG checks plus observed memory pressure. No model TPS or whole-service concurrency qualification.')
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
        record('sampler260_failed_before_or_during_checks', error_type=type(error).__name__, error=str(error),
            source_mutation_performed=False)
        raise
