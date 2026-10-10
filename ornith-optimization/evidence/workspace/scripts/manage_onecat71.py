"""Deploy a measured profile with source checks, smoke checks and rollback."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request

from qualification_artifacts78 import model_identity

ROOT = Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
BASE = 'http://127.0.0.1:8096'


def sha(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256')
    return digest.hexdigest()


def option(args, name, value):
    args = list(args)
    if name in args:
        args[args.index(name) + 1] = str(value)
    else:
        args.extend([name, str(value)])
    return args


def http(path, key='', body=None, timeout=20):
    headers = {'Content-Type': 'application/json'}
    if key:
        headers['Authorization'] = 'Bearer ' + key
    request = urllib.request.Request(BASE + path, headers=headers,
        data=None if body is None else json.dumps(body).encode())
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
        if not raw:
            return {'http_status': response.status}
        if path == '/metrics':
            return raw.decode()
        return json.loads(raw)


def discover():
    found = []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            args = (proc / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
            if '--port' not in args or args[args.index('--port') + 1] != '8096':
                continue
            is_llama = bool(args and 'llama-server' in args[0])
            is_vllm = 'vllm.entrypoints.cli.main' in args and 'serve' in args
            if not is_llama and not is_vllm:
                continue
            env = dict(item.split('=', 1) for item in
                (proc / 'environ').read_bytes().decode().rstrip('\0').split('\0')
                if '=' in item)
            found.append((int(proc.name), args, env, os.readlink(proc / 'cwd'), is_llama))
        except (OSError, UnicodeError, ValueError, IndexError):
            continue
    if len(found) > 1:
        raise RuntimeError('Multiple public serving processes detected')
    return found[0] if found else None


def api_key(args, env):
    if '--api-key' in args:
        return args[args.index('--api-key') + 1]
    return env.get('VLLM_API_KEY') or env.get('ORNITH_API_KEY') or ''


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1][0] != 'Z'
    except OSError:
        return False


def stop(pid, own_group=False):
    # Lab and manager launches use their own process group. Avoid killing an
    # unrelated shell group if the old server was started by another launcher.
    try:
        grouped = own_group or os.getpgid(pid) == pid
    except ProcessLookupError:
        if not own_group:
            return
        grouped = True
    for sig, seconds in [(signal.SIGTERM, 20), (signal.SIGKILL, 10)]:
        try:
            (os.killpg if grouped else os.kill)(pid, sig)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if not alive(pid):
                # A leader can exit before its GPU worker processes. The final
                # group kill prevents workers keeping CUDA allocations alive.
                if grouped:
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                return
            time.sleep(.2)
    raise RuntimeError('Serving process did not stop')


def ready(proc, key, llama=False, timeout=1800):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if proc.poll() is not None:
            raise RuntimeError(f'Server exited with code {proc.returncode}')
        try:
            health = http('/health', key, timeout=3)
            if llama:
                if health.get('status') == 'ok':
                    return time.monotonic() - start
            elif http('/v1/models', key, timeout=3).get('data'):
                return time.monotonic() - start
        except (OSError, ValueError, urllib.error.HTTPError):
            pass
        time.sleep(1)
    raise TimeoutError('Serving readiness timed out')


def launch(args, env, cwd, logfile):
    descriptor = os.open(logfile, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.chmod(logfile, 0o600)
    with os.fdopen(descriptor, 'a') as log:
        return subprocess.Popen(args, cwd=cwd, env=env, stdout=log, stderr=log,
            stdin=subprocess.DEVNULL, start_new_session=True)


def verify_profile(profile):
    if profile.get('status') != 'qualified' or profile['minimum_full_contexts'] < 16:
        raise RuntimeError('Profile does not preserve 16 qualified full contexts')
    command = profile['command']
    if '--api-key' in command:
        raise RuntimeError('Profile must not contain an API key')
    if command[command.index('--max-model-len') + 1] != '240128':
        raise RuntimeError('Profile context length changed')
    evidence = profile['evidence']
    if sha(evidence['path']) != evidence['sha256']:
        raise RuntimeError('Qualification evidence changed')
    rows = [json.loads(line) for line in Path(evidence['path']).read_text().splitlines() if line.strip()]
    selected = [row for row in rows if row.get('case') == profile['case']]
    if not any(row['kind'] == 'case_complete' for row in selected):
        raise RuntimeError('Selected case did not complete qualification')
    if any(row['kind'] == 'case_failed' for row in selected):
        raise RuntimeError('Selected case has a failed qualification')
    start = next(row for row in selected if row['kind'] == 'case_start')
    measured_command = list(start['command'])
    if '--api-key' in measured_command:
        index = measured_command.index('--api-key')
        del measured_command[index:index + 2]
    for flag in ['--port', '--host']:
        measured_command = option(measured_command, flag, command[command.index(flag) + 1])
    measured_env = start.get('runtime_environment', start.get('env_overrides'))
    if measured_command != command or measured_env != profile['env']:
        raise RuntimeError('Deployment differs from the measured command or environment')
    long_rows = [row for row in selected if row['kind'] == 'independent_resident_cohort'
        and row.get('concurrency', 0) >= 16]
    if not long_rows or not all(all(row['needle_passed']) for row in long_rows):
        raise RuntimeError('Independent full-context output checks are missing')
    expected_full = profile['minimum_full_contexts']
    if len([row for row in long_rows if row['concurrency'] == expected_full]) < 2:
        raise RuntimeError('Two full-capacity repetitions are missing')
    spec = start.get('qualification_spec', {})
    required_cached = spec.get('required_cached_tokens', 236944)
    if not isinstance(required_cached, int) or not 235000 <= required_cached < 239104:
        raise RuntimeError('Full-context resident cache requirement is invalid')
    if not all(row['cached_prompt_tokens'] == [required_cached] * row['concurrency']
               and row['output_tokens_per_request'] == 1024 for row in long_rows):
        raise RuntimeError('Matched full-context cache or output lengths changed')
    if spec.get('minimum_full_contexts') != expected_full:
        raise RuntimeError('Measured full-context qualification differs from profile')
    checks = {'natural_quality', 'image_ocr_count_quality',
              'max_image_quality', 'max_image_after_residency'}
    for kind in checks:
        matches = [row for row in selected if row['kind'] == kind]
        if not matches or not all(row.get('all_passed', row.get('passed', False)) for row in matches):
            raise RuntimeError('Missing or failed quality gate: ' + kind)
    if spec.get('require_fresh_images_after_long'):
        images = [row for row in selected if row['kind'] == 'max_image_after_residency']
        if not images or not all(row.get('fresh_encoder_inputs')
                and len(set(row.get('image_sha256', []))) == 2 for row in images):
            raise RuntimeError('Fresh image encoder after full residency was not qualified')
    if spec.get('mixed_short_count'):
        mixed = [row for row in selected if row['kind'] == 'mixed_long_short_cohort']
        if len(mixed) != 2 or not all(all(row['needle_passed'])
                and row['cached_long_tokens'] == [required_cached] * row['long_requests']
                for row in mixed):
            raise RuntimeError('Mixed long and short workload qualification is missing')
    if spec.get('cold_concurrent_inputs'):
        cold = [row for row in selected if row['kind'] == 'cold_concurrent_long_cohort']
        if len(cold) != 1 or not all(cold[0]['needle_passed']) or any(cold[0]['cached_prompt_tokens']):
            raise RuntimeError('Uncached concurrent full-context qualification is missing')
    if model_identity(command) != start.get('model_identity'):
        raise RuntimeError('Measured model weight file metadata changed')
    for entry in profile['file_manifest']:
        if sha(entry['path']) != entry['sha256']:
            raise RuntimeError('Qualified file changed: ' + entry['path'])
    manifest = {entry['path']: entry['sha256'] for entry in profile['file_manifest']}
    for entry in start['source_manifest']:
        source = str(Path(profile['cwd']) / entry['path'])
        if manifest.get(source) != entry['sha256']:
            raise RuntimeError('Measured source is absent from the deployment manifest')
    for entry in start.get('artifact_manifest', []):
        if manifest.get(entry['path']) != entry['sha256']:
            raise RuntimeError('Measured artifact is absent from deployment manifest')
    for name, path in profile['env'].items():
        if name.endswith('_LIBRARY') and manifest.get(path) != sha(path):
            raise RuntimeError('Measured native library is absent from the manifest')
    return {'case': profile['case'], 'minimum_full_contexts': profile['minimum_full_contexts'],
        'max_context_per_request': 240128, 'verified_files': len(profile['file_manifest'])}


def check_idle(found, key):
    if found[4]:
        if any(slot.get('is_processing') for slot in http('/slots', key)):
            raise RuntimeError('Active public requests; deployment deferred')
    else:
        metrics = http('/metrics', key)
        values = re.findall(r'^vllm:num_requests_(?:running|waiting)(?:\{[^\n]*\})?\s+([0-9.eE+-]+)', metrics, re.MULTILINE)
        if not values or any(float(value) > 0 for value in values):
            raise RuntimeError('Public request state is busy or unavailable')
    gpu_pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
        '--format=csv,noheader,nounits'], text=True).split()
    for value in set(gpu_pids):
        if not value.isdigit():
            continue
        if int(value) != found[0] and os.getpgid(int(value)) != found[0]:
            raise RuntimeError('Unrelated GPU process; deployment deferred')


def smoke(key):
    import base64
    import io
    from PIL import Image
    content = []
    for color in ['red', 'blue']:
        buffer = io.BytesIO()
        Image.new('RGB', (768, 768), color).save(buffer, format='PNG')
        content.append({'type': 'image_url', 'image_url': {
            'url': 'data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode()}})
    content.append({'type': 'text', 'text': 'Name the dominant colors of the first and second images in that order. Answer briefly in English.'})
    body = {'model': 'ornith-1.5-35b', 'temperature': 0, 'max_tokens': 100,
        'chat_template_kwargs': {'enable_thinking': False},
        'messages': [{'role': 'user', 'content': content}]}
    result = http('/v1/chat/completions', key, body, timeout=300)
    answer = result['choices'][0]['message'].get('content') or ''
    words = answer.lower()
    if 'red' not in words or 'blue' not in words or words.index('red') > words.index('blue'):
        raise RuntimeError('Public multimodal smoke check failed')
    body['messages'] = [{'role': 'user', 'content': 'Compute 17 * 23. Return only the integer.'}]
    arithmetic = http('/v1/chat/completions', key, body, timeout=180)['choices'][0]['message'].get('content') or ''
    if re.search(r'(?<!\d)391(?!\d)', arithmetic) is None:
        raise RuntimeError('Public arithmetic smoke check failed')
    return {'image_count': 2, 'image_pixels_each': 589824, 'image_answer': answer,
        'arithmetic_answer': arithmetic, 'passed': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('profile', type=Path, nargs='?', default=ROOT / 'results/qualified-serving-profile71.json')
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--rollback', action='store_true')
    args = parser.parse_args()
    profile = None if args.rollback else json.loads(args.profile.read_text())
    checked = {'case': 'previous_serving_profile'} if args.rollback else verify_profile(profile)
    if args.validate_only:
        print(json.dumps({'status': 'validated_only', **checked}), flush=True)
        return
    found = discover()
    key = api_key(found[1], found[2]) if found else os.environ.get('ORNITH_API_KEY', '')
    if not key:
        raise RuntimeError('No runtime API key was supplied')
    if found:
        check_idle(found, key)
    rollback = ROOT / 'results/onecat-rollback-profile71.json'
    if args.rollback:
        saved = json.loads(rollback.read_text())
        command = [item.replace('{RUNTIME_API_KEY}', key) for item in saved['command']]
        env = {**os.environ, **saved['env']}
        if saved.get('environment_key'):
            env['VLLM_API_KEY'] = key
        cwd, llama = saved['cwd'], saved['llama']
    else:
        if found:
            original = [item.replace(key, '{RUNTIME_API_KEY}') for item in found[1]]
            safe_env = {name: value for name, value in found[2].items()
                if name in {'PATH', 'LD_LIBRARY_PATH', 'CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'CUDA_HOME', 'TOKENIZERS_PARALLELISM'}
                or name.startswith(('GGML_', 'VLLM_SM70_', 'PYTORCH_'))}
            rollback.write_text(json.dumps({'command': original, 'env': safe_env,
                'cwd': found[3], 'llama': found[4], 'environment_key': '--api-key' not in found[1]}, indent=2))
        command = option(option(profile['command'], '--port', 8096), '--host', '0.0.0.0')
        env = {**(found[2] if found else os.environ), **profile['env'], 'VLLM_API_KEY': key}
        cwd, llama = profile['cwd'], False
    if found:
        stop(found[0])
    current = None
    logfile = ROOT / 'logs/onecat-public71.log'
    log_offset = logfile.stat().st_size if logfile.exists() else 0
    try:
        current = launch(command, env, cwd, logfile)
        startup = ready(current, key, llama=llama)
        if not llama:
            with logfile.open('rb') as stream:
                stream.seek(log_offset)
                contents = stream.read().decode(errors='replace')
            matches = re.findall(r'Maximum concurrency for [0-9,]+ tokens per request: ([0-9.]+)x', contents)
            minimum = profile['minimum_full_contexts'] if profile else 16
            if not matches or min(float(value) for value in matches) < minimum:
                raise RuntimeError('Public full-context KV capacity changed')
            verification = smoke(key)
        else:
            slots = http('/slots', key)
            if len(slots) < 16 or {slot.get('n_ctx') for slot in slots} != {240128}:
                raise RuntimeError('Rollback context capacity check failed')
            verification = {'passed': True, 'slots': len(slots)}
        result = {'status': 'rolled_back' if args.rollback else 'deployed',
            'pid': current.pid, 'startup_s': startup, 'port': 8096,
            'api_key_preserved': bool(found), **checked, 'smoke': verification,
            'timestamp_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
        (ROOT / 'results/onecat-public-deployment71.json').write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
    except BaseException:
        if current:
            stop(current.pid, own_group=True)
        if found:
            restored = launch(found[1], found[2], found[3], ROOT / 'logs/rollback71.log')
            ready(restored, key, llama=found[4])
        raise


if __name__ == '__main__':
    main()
