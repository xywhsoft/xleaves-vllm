"""Queue an exact-token layer baseline after all optimization GPU jobs finish."""
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import socket
import subprocess
import sys
import time

ROOT = Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
LAB = ROOT.parent / 'scripts/lab_runner.py'
SSE = ROOT / 'scripts/llama_matched_runner08.py'
BINARY = Path('/opt/llama.cpp-git/build/bin/llama-server')
PINS = {
    str(LAB): '228a4435b067fbdd7f13dfe12c4fe83ce23050c9c1a51fe19138aef162cf58a5',
    str(SSE): '4ca8efe95fdef1b090602191ec3b83e0de202b553406d7763dda154db007c22c',
    str(BINARY): 'fe196b9dc11fbce357a2efce91e211804ad471b66ec84480dc7b874e15f16beb',
    str(ROOT / 'results/exact-layer215-input-proof.json'): '42e95b5ba635f3ba6028103c89bd11fe57336426fc053070ac4f10366d104525',
}
RUN = 'llama_exact_layer216'
PARENT, BIRTH = 901473, '232463386'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def token_hash(prompts):
    return hashlib.sha256(json.dumps(prompts, separators=(',', ':')).encode()).hexdigest()


def process(pid):
    try:
        parts = Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()
        return parts[0], parts[19]
    except FileNotFoundError:
        return 'missing', None


def record(status, **values):
    result = dict(status=status, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **values)
    (ROOT / 'results/layer216-status.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


def inputs():
    for path, digest in PINS.items():
        assert sha(path) == digest, path
    proof = json.loads((ROOT / 'results/exact-layer215-input-proof.json').read_text())
    assert len(proof['matched_cohorts']) == 8 and all(x['passed'] for x in proof['matched_cohorts'])
    for item in proof['inputs']:
        assert sha(item['path']) == item['sha256']
    short = json.loads(Path(proof['inputs'][0]['path']).read_text())
    long = json.loads(Path(proof['inputs'][1]['path']).read_text())[:16]
    for row in proof['matched_cohorts']:
        prompts = short[f"512-{row['repeat']}"][:row['concurrency']] if row['kind'] == 'cohort' else [x['prompt'] for x in long[:row['concurrency']]]
        assert token_hash(prompts) == row['input_token_arrays_sha256']
        assert all(len(p) == row['input_tokens_per_request'] for p in prompts)
    check_states(proof)
    return proof, short, long


def check_states(proof):
    for item in proof['slot_states']:
        stat = Path(item['path']).stat()
        assert (stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_dev) == (item['bytes'], item['mtime_ns'], item['inode'], item['device'])


class Transform(ast.NodeTransformer):
    def __init__(self):
        self.changes = {}

    def mark(self, key):
        self.changes[key] = self.changes.get(key, 0) + 1

    def visit_FunctionDef(self, node):
        if node.name == 'cohort':
            node.args.args.extend([ast.arg(arg='cache_prompt'), ast.arg(arg='slot_ids')])
            node.args.defaults.extend([ast.Constant(False), ast.Constant(None)])
            self.mark('signature')
        return self.generic_visit(node)

    def visit_Call(self, node):
        self.generic_visit(node)
        if isinstance(node.func, ast.Name) and node.func.id == 'dict':
            if any(k.arg == 'cache_prompt' for k in node.keywords):
                for keyword in node.keywords:
                    if keyword.arg == 'cache_prompt':
                        keyword.value = ast.Name(id='cache_prompt', ctx=ast.Load())
                node.keywords.append(ast.keyword(arg='id_slot', value=ast.parse('index if slot_ids is None else slot_ids[index]', mode='eval').body))
                self.mark('cache_slot')
            for keyword in node.keywords:
                if keyword.arg in {'decode_tps', 'aggregate_common_decode_tps', 'average_common_decode_request_tps'}:
                    condition = 'last > first' if keyword.arg == 'decode_tps' else 'duration > 0'
                    keyword.value = ast.IfExp(test=ast.parse(condition, mode='eval').body, body=keyword.value, orelse=ast.Constant(None))
                    self.mark(keyword.arg)
                if keyword.arg == 'median_request_decode_tps':
                    keyword.value = ast.parse("statistics.median(row['decode_tps'] for row in rows if row['decode_tps'] is not None) if any(row['decode_tps'] is not None for row in rows) else None", mode='eval').body
                    self.mark(keyword.arg)
        if isinstance(node.func, ast.Attribute) and node.func.attr == 'urlopen':
            for keyword in node.keywords:
                if keyword.arg == 'timeout':
                    assert isinstance(keyword.value, ast.Constant) and keyword.value.value == 1800
                    keyword.value = ast.Constant(10800)
                    self.mark('timeout')
        return node


def load_sse():
    assert sha(LAB) == PINS[str(LAB)] and sha(SSE) == PINS[str(SSE)]
    transform = Transform()
    tree = transform.visit(ast.parse(SSE.read_text().split('def main():')[0]))
    assert transform.changes == dict(signature=1, cache_slot=1, decode_tps=1, aggregate_common_decode_tps=1, average_common_decode_request_tps=1, median_request_decode_tps=1, timeout=1)
    namespace = {}
    exec(compile(ast.fix_missing_locations(tree), 'exact_layer216_sse', 'exec'), namespace)
    return namespace, transform.changes


def terminal_jobs():
    checked = []
    for name in ['workspace187', 'numa189', 'boundary200', 'cache210']:
        job = json.loads((ROOT / 'cache' / f'{name}-job.json').read_text())
        state, birth = process(job['pid'])
        assert birth != str(job['birth']) or state in ['missing', 'Z', 'X'], f'Active optimization job {name}'
        checked.append(dict(job=name, pid=job['pid'], state=state, birth_matches=birth == str(job['birth'])))
    return checked


def main():
    output = ROOT / 'results' / f'{RUN}.jsonl'
    assert not output.exists()
    proof, short, long = inputs()
    ns, changes = load_sse()
    record('waiting_terminal_restored210', parent_pid=PARENT, parent_birth=BIRTH, no_gpu_work=True)
    deadline = time.monotonic() + 72 * 3600
    while True:
        state, birth = process(PARENT)
        if state in ['missing', 'Z', 'X'] or birth != BIRTH:
            break
        assert time.monotonic() < deadline
        time.sleep(10)
    jobs = terminal_jobs()
    lock = (ROOT / 'cache/gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    terminal_jobs()
    inputs()
    pid, original, env, cwd = ns['live']()
    key = original[original.index('--api-key') + 1]
    public = 'http://127.0.0.1:8096'
    assert not any(slot.get('is_processing') for slot in ns['http'](public, '/slots', key)), 'Public service busy'
    gpu_pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    assert {int(p) for p in gpu_pids.split() if p.isdigit()} == {pid}, 'Other GPU process exists'
    assert env.get('CUDA_VISIBLE_DEVICES') in ['0,1', '0, 1']
    emit, cohort = ns['emit'], ns['cohort']
    slotdir = Path(proof['slot_states'][0]['path']).parent
    testkey = secrets.token_urlsafe(24)
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    args = list(original)
    args[0] = str(BINARY)
    for flag, value in {'--host': '127.0.0.1', '--port': port, '--api-key': testkey, '--split-mode': 'layer', '-ub': 256, '-b': 512, '-np': 16, '-c': 240128 * 16, '--slot-save-path': str(slotdir)}.items():
        args = ns['option'](args, flag, value)
    for flag, expected in {'--cache-type-k': 'q4_0', '--cache-type-v': 'q4_0', '-fa': 'on', '-ngl': '999'}.items():
        assert args[args.index(flag) + 1] == expected
    native = subprocess.check_output(['ldd', str(BINARY)], text=True)
    libraries = sorted({line.split('=>', 1)[1].strip().split()[0] for line in native.splitlines() if '=>' in line and line.split('=>', 1)[1].strip().startswith('/')})
    pins = [dict(path=p, sha256=sha(p)) for p in [str(BINARY), str(LAB), str(SSE), str(Path(__file__).resolve())] + libraries]
    emit(output, dict(kind='start', run=RUN, terminal_jobs=jobs, proof215_sha256=PINS[str(ROOT / 'results/exact-layer215-input-proof.json')], file_manifest=pins, transform=changes, hardware_changes=False, original_pid=pid, scope='Exact accepted token IDs and greedy 1024-output policy; 236944-token saved states restored before each long cohort. Restoration excluded from timed throughput. No new state saves or 1Cat source mutation.'))
    current = None
    paused = False
    failed = False
    completed = False
    logpath = ROOT / 'logs/layer216-private.log'
    def cancel(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    try:
        paused = True
        ns['stop'](pid)
        emit(output, dict(kind='case_start', case='layer_ub256_b512_exact216', command=[x.replace(testkey, '[REDACTED]') for x in args]))
        with logpath.open('x') as log:
            current = subprocess.Popen(args, cwd=cwd, env=env, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
        base = f'http://127.0.0.1:{port}'
        startup = ns['ready'](base, current, 240)
        slots = ns['http'](base, '/slots', testkey)
        assert len(slots) == 16 and {s['n_ctx'] for s in slots} == {240128}
        emit(output, dict(kind='case_ready', case='layer_ub256_b512_exact216', startup_s=startup, slots=16, context=240128, gpus=ns['gpu']()))
        record('running_actual_exact_layer216', private_pid=current.pid, public_paused=True)
        cohort(base, testkey, [p[:128] for p in short['512-0'][:16]], 32)
        for gate in proof['matched_cohorts']:
            count, repeat, kind = gate['concurrency'], gate['repeat'], gate['kind']
            is_long = kind == 'independent_resident_cohort'
            prompts = [x['prompt'] for x in long[:count]] if is_long else short[f'512-{repeat}'][:count]
            assert token_hash(prompts) == gate['input_token_arrays_sha256']
            if is_long:
                check_states(proof)
                for index in range(count):
                    response = ns['http'](base, f'/slots/{index}?action=restore', testkey, {'filename': f'prefix-{index:02d}.bin'}, timeout=300)
                    emit(output, dict(kind='prefix_restored', index=index, concurrency=count, repeat=repeat, response=response))
            row = cohort(base, testkey, prompts, 1024, cache_prompt=is_long)
            cached = [r['timings']['cache_n'] for r in row['requests']]
            assert cached == [236944 if is_long else 0] * count
            assert all(r['token_ids_available'] and r['timings']['predicted_n'] == 1024 for r in row['requests'])
            needles = [item['slot'] in r['output_text'] and item['needle'] in r['output_text'] for item, r in zip(long[:count], row['requests'])] if is_long else None
            assert not is_long or all(needles)
            row.update(kind=kind, case='layer_ub256_b512_exact216', repeat=repeat, cached_prompt_tokens=cached, needle_passed=needles, input_token_arrays_sha256=token_hash(prompts), exact215_gate=gate, gpus=ns['gpu']())
            emit(output, row)
        check_states(proof)
        emit(output, dict(kind='case_complete', case='layer_ub256_b512_exact216', exact_input_cohorts=8, unchanged_saved_states=True))
        completed = True
    except BaseException as error:
        failed = True
        emit(output, dict(kind='case_failed', case='layer_ub256_b512_exact216', error_type=type(error).__name__, error=str(error).replace(testkey, '[REDACTED]')))
    finally:
        if current is not None:
            ns['stop_group'](current)
        if paused:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            with open('/tmp/crack8096.log', 'a') as log:
                restored = subprocess.Popen(original, cwd=cwd, env=env, stdout=log, stderr=log, stdin=subprocess.DEVNULL, start_new_session=True)
            startup = ns['ready'](public, restored, 240)
            slots = ns['http'](public, '/slots', key)
            assert len(slots) == 16 and {s['n_ctx'] for s in slots} == {240128}
            emit(output, dict(kind='restored', pid=restored.pid, startup_s=startup, verification=dict(passed=True, slots=16, context=240128, health=ns['http'](public, '/health'))))
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
    passed = completed and not failed
    emit(output, dict(kind='run_complete', passed=passed))
    record('complete' if passed else 'failed_qualification', passed=passed, source_mutation_performed=False, public_restored=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        record('failed', error_type=type(error).__name__, error=str(error), source_mutation_performed=False)
        raise
