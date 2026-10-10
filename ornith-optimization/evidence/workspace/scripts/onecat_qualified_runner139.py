"""Isolated throughput lab; restores production in finally; no saved keys."""
import concurrent.futures
import base64
import fcntl
import hashlib
import json
import io
import os
from pathlib import Path
import secrets
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, '/home/xs/codex-work/ornith-opt-20261006/scripts')
from lab_runner import emit, gpu, http, ready, stop_group
from manage_onecat71 import (
    alive as public_alive, api_key, check_idle, discover,
    http as public_http, launch, ready as public_ready, smoke,
    stop as public_stop,
)
from qualification_artifacts78 import (
    artifact_manifest, effective_environment, model_identity,
)

ROOT = Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')


def cohort(base, key, prompts, output):
    count = len(prompts)
    gate = threading.Barrier(count + 1)

    def one(index):
        body = dict(model='ornith-1.5-35b', prompt=prompts[index],
                    max_tokens=output, temperature=0, seed=1234,
                    ignore_eos=True, return_token_ids=True, stream=True,
                    stream_options={'include_usage': True})
        request = urllib.request.Request(
            base + '/v1/completions', data=json.dumps(body).encode(),
            headers={'Authorization': 'Bearer ' + key,
                     'Content-Type': 'application/json'})
        gate.wait()
        start = time.perf_counter()
        events, ids, chunks, usage = [], [], [], None
        with urllib.request.urlopen(request, timeout=10800) as response:
            for line in response:
                if not line.startswith(b'data: '):
                    continue
                try:
                    item = json.loads(line[6:])
                except ValueError:
                    continue
                if item.get('error'):
                    raise RuntimeError(str(item['error']))
                if item.get('usage'):
                    usage = item['usage']
                for choice in item.get('choices', []):
                    token_ids = choice.get('token_ids') or []
                    if token_ids:
                        now = time.perf_counter()
                        events.append([now, len(token_ids)])
                        ids.extend(token_ids)
                    chunks.append(choice.get('text') or '')
        end = time.perf_counter()
        if len(ids) != output or not usage or usage['completion_tokens'] != output:
            raise RuntimeError(f'token accounting mismatch: ids={len(ids)}, usage={usage}')
        first, last = events[0][0], events[-1][0]
        return dict(start=start, first=first, last=last, end=end,
                    elapsed_s=end-start, ttft_s=first-start,
                    decode_tps=(len(ids)-events[0][1])/(last-first),
                    usage=usage, token_events=events,
                    output_token_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                    output_text=''.join(chunks),
                    output_preview=''.join(chunks)[:100])

    with concurrent.futures.ThreadPoolExecutor(max_workers=count) as pool:
        jobs = [pool.submit(one, index) for index in range(count)]
        started = time.perf_counter()
        gate.wait()
        rows = [job.result() for job in jobs]
        finished = time.perf_counter()
    window_start = max(row['first'] for row in rows)
    window_end = min(row['last'] for row in rows)
    window_tokens = sum(n for row in rows for at, n in row['token_events']
                        if window_start < at <= window_end)
    window_time = window_end-window_start
    total = count*output
    return dict(kind='cohort', concurrency=count,
                input_token_arrays_sha256=hashlib.sha256(
                    json.dumps(prompts, separators=(',', ':')).encode()).hexdigest(),
                input_tokens_per_request=len(prompts[0]) if len({len(p) for p in prompts}) == 1 else None,
                input_tokens_per_request_lengths=[len(p) for p in prompts],
                output_tokens_per_request=output, output_tokens=total,
                wall_s=finished-started,
                aggregate_e2e_output_tps=total/(finished-started),
                average_e2e_request_tps=total/(finished-started)/count,
                common_decode_window_s=window_time,
                common_decode_window_tokens=window_tokens,
                aggregate_common_decode_tps=window_tokens/window_time if window_time>0 else None,
                average_common_decode_request_tps=window_tokens/window_time/count if window_time>0 else None,
                median_request_decode_tps=statistics.median(row['decode_tps'] for row in rows),
                median_ttft_s=statistics.median(row['ttft_s'] for row in rows),
                requests=rows)


def main():
    plan = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'))
    run = plan['run']
    out = ROOT/'results'/(run+'.jsonl')
    lock = (ROOT/'cache/gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX if plan.get('wait_for_gpu_lock') else fcntl.LOCK_EX | fcntl.LOCK_NB)
    # Never overlap the existing matrix, profiler, or finalizer.
    if any(public_alive(pid) for pid in [522542, 522543, 540211]):
        raise RuntimeError('Existing qualification/profile/finalizer is still running')
    found = discover()
    if found is None:
        raise RuntimeError('No restorable public serving process')
    pid, original, env, cwd, original_llama = found
    key = api_key(original, env)
    check_idle(found, key)
    livebase = 'http://127.0.0.1:8096'
    prompts = {}
    for length, label in [(128, 'warm')] + [
            (spec['input_tokens'], f"{spec['input_tokens']}-{repeat}")
            for spec in plan['tests'] for repeat in range(spec.get('repeats', 1))]:
        if label in prompts:
            continue
        prompts[label] = []
        for index in range(max(16, max(spec['concurrency'] for spec in plan['tests']))):
            content = (f'Request {index} benchmark {label}. Continue counting integers with commas.\n'
                       + '1,2,3,4,5,6,7,8,9,10.\n'*max(30,length//10))
            tokenization_body = (
                {'content': content, 'add_special': True} if original_llama else
                {'model': 'ornith-1.5-35b', 'prompt': content,
                 'add_special_tokens': True}
            )
            tokens = http(livebase, '/tokenize', key,
                          tokenization_body, timeout=90)['tokens'][:length]
            if len(tokens) != length:
                raise RuntimeError('Generated prompt too short')
            prompts[label].append(tokens)
    if plan.get('diverse_prompts'):
        diverse = json.loads(Path(plan['diverse_prompts']).read_text())
        for label, items in diverse.items():
            prompts[label] = [item['prompt'] for item in items]
    long_inputs = None
    if plan.get('long_residency') or any(case.get('long_pilot') or case.get('long_residency') for case in plan['cases']):
        long_inputs = json.loads(Path(plan.get('long_inputs', ROOT/'scripts/long_independent_prompts.json')).read_text())
        if any(len(row['prompt']) != 239104 for row in long_inputs):
            raise RuntimeError('Long prompt length gate failed')
        if len({tuple(row['prompt'][:16]) for row in long_inputs}) != len(long_inputs):
            raise RuntimeError('Long requests share their first KV block')
    prompt_path = ROOT / 'results' / (run + '_short_inputs.json')
    if prompt_path.exists():
        raise RuntimeError('Short-input evidence already exists; use a fresh run name')
    prompt_bytes = json.dumps(prompts, sort_keys=True, separators=(',', ':')).encode()
    prompt_path.write_bytes(prompt_bytes)
    emit(out, dict(kind='start', run=run, original_pid=pid,
                   original_engine='llama.cpp' if original_llama else 'vllm',
                   short_inputs=dict(path=str(prompt_path), bytes=len(prompt_bytes),
                                     sha256=hashlib.sha256(prompt_bytes).hexdigest()),
                   timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    current, paused, failed = None, False, False
    qualified_cases = set()

    def cancel(signum, frame):
        raise KeyboardInterrupt(f'signal {signum}')

    signal.signal(signal.SIGTERM, cancel)
    signal.signal(signal.SIGINT, cancel)
    try:
        current_public = discover()
        if current_public is None or current_public[0] != pid:
            raise RuntimeError('Public serving process changed during preparation')
        check_idle(current_public, key)
        try:
            public_stop(pid)
            paused = True
        except BaseException:
            paused = not public_alive(pid)
            raise
        all_long_inputs = long_inputs
        qualified_groups = set()
        for case in plan['cases']:
            admission_group = case.get('admission_group')
            if admission_group and admission_group in qualified_groups:
                emit(out, dict(kind='case_skipped', case=case['name'], reason='Earlier configuration in this fallback group passed every gate'))
                continue
            long_inputs = all_long_inputs[:case.get('long_input_count', len(all_long_inputs))] if all_long_inputs else None
            if long_inputs and max(case.get('resident_concurrencies', [len(long_inputs)])) > len(long_inputs):
                raise RuntimeError('Long residency requests exceed physically prepared prefixes')
            name, testkey = case['name'], 'lab_' + secrets.token_urlsafe(24)
            with socket.socket() as reservation:
                reservation.bind(('127.0.0.1', 0))
                port = reservation.getsockname()[1]
            base = f'http://127.0.0.1:{port}'
            command = [part.replace('{KEY}', testkey).replace('{PORT}', str(port))
                       for part in case['command']]
            logpath = ROOT/'logs'/(run+'_'+name+'.log')
            source_files = subprocess.check_output(
                ['git', 'ls-files', '--modified', '--others', '--exclude-standard'],
                cwd=case['cwd'], text=True).splitlines()
            source_manifest = []
            for relative in sorted(set(source_files)):
                source = Path(case['cwd'])/relative
                if source.is_file():
                    source_manifest.append(dict(path=relative,bytes=source.stat().st_size,
                        sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
            emit(out, dict(kind='case_start', case=name,
                           env_overrides=case.get('env', {}),
                           runtime_environment=effective_environment({**env, **case.get('env', {})}),
                           artifact_manifest=artifact_manifest(case),
                           model_identity=model_identity(command),
                           qualification_spec={k: v for k, v in case.items() if k not in {'command', 'env'}},
                           source_commit=subprocess.check_output(
                               ['git','rev-parse','HEAD'],cwd=case['cwd'],text=True).strip(),
                           source_manifest=source_manifest,
                           command=[part.replace(testkey, '[REDACTED]') for part in command]))
            samples, sample_stop = [], threading.Event()

            def monitor():
                while not sample_stop.is_set():
                    try:
                        samples.append(dict(t=time.time(), gpus=gpu()))
                    except Exception:
                        pass
                    sample_stop.wait(1)

            sampler = None
            try:
                if case.get('gpu_precheck'):
                    for check in case['gpu_precheck']:
                        checked = subprocess.run(check, cwd=case['cwd'],
                            env={**env, **case.get('env', {})}, capture_output=True,
                            text=True, timeout=case.get('gpu_precheck_timeout',300))
                        emit(out,dict(kind='gpu_precheck',case=name,
                            command=check,returncode=checked.returncode,
                            output=checked.stdout+checked.stderr))
                        if checked.returncode:
                            raise RuntimeError('GPU kernel precheck failed')
                if case.get('gqa_tuning_result'):
                    measured=json.loads(Path(case['gqa_tuning_result']).read_text())
                    valid=[row for row in measured if row['batch']==16 and row['context']==239104]
                    fastest=min(valid,key=lambda row:row['gqa_us'])
                    case['env']['VLLM_SM70_TURBOQUANT_GQA_TILE']=str(fastest['tile'])
                    case['env']['VLLM_SM70_TURBOQUANT_GQA_WARPS']=str(fastest['warps'])
                    emit(out,dict(kind='gqa_tuning_selection',case=name,selection=fastest))
                with logpath.open('w') as log:
                    current = subprocess.Popen(command, cwd=case['cwd'],
                        env={**env, **case.get('env', {})}, stdout=log, stderr=log,
                        stdin=subprocess.DEVNULL, start_new_session=True)
                load_start = time.monotonic()
                while time.monotonic()-load_start < case.get('readiness_timeout', 900):
                    if current.poll() is not None:
                        raise RuntimeError(f'server exited {current.returncode}')
                    try:
                        if http(base, '/v1/models', testkey, timeout=3).get('data'):
                            break
                    except Exception:
                        pass
                    time.sleep(1)
                else:
                    raise TimeoutError('Server readiness timeout')
                emit(out, dict(kind='case_ready', case=name,
                    startup_s=time.monotonic()-load_start, idle_gpu=gpu(),
                    capacity_lines=[line for line in logpath.read_text(errors='replace').splitlines()
                                    if any(term in line.lower() for term in
                                           ['kv cache size', 'maximum concurrency', 'gpu kv cache', 'num_gpu_blocks'])]))
                if case.get('minimum_full_contexts'):
                    import re
                    matches = re.findall(r'Maximum concurrency for [0-9,]+ tokens per request: ([0-9.]+)x',
                                         logpath.read_text(errors='replace'))
                    if not matches or min(float(value) for value in matches) < case['minimum_full_contexts']:
                        raise RuntimeError('Full-context capacity gate failed before admission')
                sampler = threading.Thread(target=monitor, daemon=True)
                sampler.start()
                if case.get('image_checks'):
                    from PIL import Image
                    for color in ['red', 'blue']:
                        data = io.BytesIO()
                        Image.new('RGB', (128, 128), color).save(data, format='PNG')
                        url = 'data:image/png;base64,' + base64.b64encode(data.getvalue()).decode()
                        body = dict(model='ornith-1.5-35b', temperature=0, max_tokens=100,
                            chat_template_kwargs={'enable_thinking':False},
                            messages=[dict(role='user',content=[
                                dict(type='image_url',image_url=dict(url=url)),
                                dict(type='text',text='What is the dominant color of the image? Answer with the English color name.')])])
                        result = http(base, '/v1/chat/completions', testkey, body, timeout=300)
                        message = result['choices'][0]['message']
                        content = message.get('content') or ''
                        emit(out, dict(kind='image_quality',case=name, expected=color,
                            passed=color in content.lower(), answer=content,
                            reasoning=message.get('reasoning_content'), usage=result.get('usage')))
                        if color not in content.lower():
                            raise RuntimeError('Image color correctness gate failed')
                    sys.path.insert(0, '/home/xs/codex-work/ornith-opt-20261006/scripts')
                    from long_capacity_runner import natural_checks
                    quality = natural_checks(base, testkey)
                    emit(out, dict(kind='natural_quality',case=name,
                        all_passed=all(row['passed'] for row in quality), results=quality))
                    if not all(row['passed'] for row in quality):
                        raise RuntimeError('Text correctness gate failed')
                if case.get('extra_image_checks'):
                    sys.path.insert(0, str(ROOT / 'scripts'))
                    from mm_quality_checks24 import checks
                    quality = checks(base, testkey, http)
                    emit(out, dict(kind='image_ocr_count_quality', case=name,
                                   all_passed=all(row['passed'] for row in quality),
                                   results=quality))
                    if not all(row['passed'] for row in quality):
                        raise RuntimeError('Image OCR/count correctness gate failed')
                if case.get('max_image_probe'):
                    from PIL import Image
                    content = []
                    for color in ['red', 'blue']:
                        data = io.BytesIO()
                        Image.new('RGB', (768, 768), color).save(data, format='PNG')
                        url = 'data:image/png;base64,' + base64.b64encode(data.getvalue()).decode()
                        content.append(dict(type='image_url', image_url=dict(url=url)))
                    content.append(dict(type='text', text='Name the dominant colors of the first and second images in that order. Answer briefly in English.'))
                    result = http(base, '/v1/chat/completions', testkey,
                        dict(model='ornith-1.5-35b', temperature=0, max_tokens=100,
                            chat_template_kwargs={'enable_thinking':False},
                            messages=[dict(role='user',content=content)]), timeout=300)
                    answer = result['choices'][0]['message'].get('content') or ''
                    words = answer.lower()
                    passed = 'red' in words and 'blue' in words and words.index('red') < words.index('blue')
                    emit(out, dict(kind='max_image_quality',case=name,passed=passed,
                        pixels_per_image=589824,image_count=2,answer=answer,usage=result.get('usage'),gpus=gpu()))
                    if not passed:
                        raise RuntimeError('Maximum image size/count correctness gate failed')
                warm = cohort(base, testkey, prompts['warm'], 32)
                warm.update(kind='warmup', case=name)
                emit(out, warm)
                for spec in case.get('tests', plan['tests']):
                    for repeat in range(spec.get('repeats', 1)):
                        row = cohort(base, testkey,
                            prompts[f"{spec['input_tokens']}-{repeat}"][:spec['concurrency']],
                            spec['output_tokens'])
                        row.update(case=name, repeat=repeat)
                        emit(out, row)
                if case.get('extra_diverse_cohorts'):
                    diverse = json.loads(Path(case['extra_diverse_cohorts']).read_text())
                    for repeat in range(2):
                        row = cohort(base, testkey,
                            [item['prompt'] for item in diverse[f'512-{repeat}']][:case.get('diverse_count', 24)], 1024)
                        row.update(kind='diverse_cohort', case=name, repeat=repeat)
                        emit(out, row)
                if case.get('long_pilot'):
                    item=long_inputs[0]
                    emit(out,dict(kind='long_pilot_start',case=name,input_tokens=len(item['prompt'])))
                    row=cohort(base,testkey,[item['prompt']],128)
                    answer=row['requests'][0]['output_text']
                    correct=item['slot'] in answer and item['needle'] in answer
                    row.update(kind='long_pilot',case=name,needle_passed=correct)
                    emit(out,row)
                    if not correct:
                        raise RuntimeError('Long pilot needle correctness failed')
                if case.get('long_residency',plan.get('long_residency')):
                    for index, item in enumerate(long_inputs):
                        emit(out,dict(kind='independent_cold_prefill_start',case=name,index=index,
                            input_tokens=len(item['prompt'])))
                        row=cohort(base,testkey,[item['prompt']],128)
                        answer=row['requests'][0]['output_text']
                        correct=item['slot'] in answer and item['needle'] in answer
                        row.update(kind='independent_cold_prefill',case=name,index=index,needle_passed=correct)
                        emit(out,row)
                        if not correct:
                            raise RuntimeError(f'Long needle correctness failed at request {index}')
                    for concurrency, repeat in [(n, r) for n in case.get('resident_concurrencies', [len(long_inputs)]) for r in range(2)]:
                        resident_inputs = long_inputs[:concurrency]
                        row=cohort(base,testkey,[item['prompt'] for item in resident_inputs],1024)
                        cached=[(request['usage'].get('prompt_tokens_details') or {}).get('cached_tokens',0)
                                for request in row['requests']]
                        correct=[item['slot'] in request['output_text'] and item['needle'] in request['output_text']
                                 for item,request in zip(resident_inputs,row['requests'])]
                        row.update(kind='independent_resident_cohort',case=name,repeat=repeat,
                            cached_prompt_tokens=cached,needle_passed=correct,
                            methodology=f'{len(long_inputs)} independent resident caches; {concurrency} active requests; concurrent cache hits; no inter-request prefix sharing')
                        emit(out,row)
                        required_cached = case.get('required_cached_tokens')
                        if (min(cached)<235000 or not all(correct) or row['common_decode_window_s']<=0
                            or (required_cached is not None and any(n != required_cached for n in cached))):
                            raise RuntimeError('Independent long residency/cache/correctness gate failed')
                    if case.get('mixed_short_count'):
                        long_count = case.get('mixed_long_count', 16)
                        short_count = case['mixed_short_count']
                        diverse = json.loads(Path(case['extra_diverse_cohorts']).read_text())
                        for repeat in range(2):
                            mixed_prompts = [item['prompt'] for item in long_inputs[:long_count]] + [item['prompt'] for item in diverse[f'512-{repeat}'][:short_count]]
                            row = cohort(base, testkey, mixed_prompts, 1024)
                            requests = row['requests'][:long_count]
                            cached = [(request['usage'].get('prompt_tokens_details') or {}).get('cached_tokens', 0) for request in requests]
                            correct = [item['slot'] in request['output_text'] and item['needle'] in request['output_text'] for item, request in zip(long_inputs[:long_count], requests)]
                            row.update(kind='mixed_long_short_cohort', case=name, repeat=repeat,
                                long_requests=long_count, short_requests=short_count,
                                cached_long_tokens=cached, needle_passed=correct,
                                long_median_request_decode_tps=statistics.median(request['decode_tps'] for request in requests),
                                short_median_request_decode_tps=statistics.median(request['decode_tps'] for request in row['requests'][long_count:]))
                            emit(out, row)
                            if not all(correct) or any(n != case['required_cached_tokens'] for n in cached):
                                raise RuntimeError('Mixed long/short residency/cache/correctness gate failed')
                    if case.get('image_after_long'):
                        from PIL import Image
                        content = []
                        image_hashes = []
                        for color in [(235, 13, 19), (19, 13, 235)]:
                            data = io.BytesIO()
                            Image.new('RGB', (768, 768), color).save(data, format='PNG')
                            image_hashes.append(hashlib.sha256(data.getvalue()).hexdigest())
                            content.append(dict(type='image_url', image_url=dict(url='data:image/png;base64,' + base64.b64encode(data.getvalue()).decode())))
                        content.append(dict(type='text', text='Name the dominant colors of the first and second images in that order. Answer briefly in English.'))
                        result = http(base, '/v1/chat/completions', testkey,
                            dict(model='ornith-1.5-35b', temperature=0, max_tokens=100,
                                chat_template_kwargs={'enable_thinking': False},
                                messages=[dict(role='user', content=content)]), timeout=300)
                        answer = result['choices'][0]['message'].get('content') or ''
                        words = answer.lower()
                        passed = 'red' in words and 'blue' in words and words.index('red') < words.index('blue')
                        emit(out, dict(kind='max_image_after_residency', case=name, passed=passed,
                            resident_contexts=len(long_inputs), pixels_per_image=589824, image_count=2,
                            fresh_encoder_inputs=True, image_sha256=image_hashes,
                            answer=answer, usage=result.get('usage'), gpus=gpu()))
                        if not passed:
                            raise RuntimeError('Maximum images after full residency correctness gate failed')
                    if case.get('cold_concurrent_inputs'):
                        cold = json.loads(Path(case['cold_concurrent_inputs']).read_text())[:16]
                        emit(out, dict(kind='cold_concurrent_prefill_start', case=name,
                            concurrency=len(cold), input_tokens_per_request=239104,
                            output_tokens_per_request=1024))
                        row = cohort(base, testkey, [item['prompt'] for item in cold], 1024)
                        cached = [(request['usage'].get('prompt_tokens_details') or {}).get('cached_tokens', 0) for request in row['requests']]
                        correct = [item['slot'] in request['output_text'] and item['needle'] in request['output_text'] for item, request in zip(cold, row['requests'])]
                        row.update(kind='cold_concurrent_long_cohort', case=name, cached_prompt_tokens=cached,
                            needle_passed=correct, methodology='16 simultaneous independent 239K inputs, uncached; input processing included in E2E output TPS')
                        emit(out, row)
                        if any(cached) or not all(correct):
                            raise RuntimeError('Cold concurrent full-context gate failed')
                if case.get('cuda_profile'):
                    def profile_action(path):
                        request = urllib.request.Request(base+path, data=b'',
                            headers={'Authorization':'Bearer '+testkey},method='POST')
                        with urllib.request.urlopen(request,timeout=60) as response:
                            return response.status
                    emit(out,dict(kind='profile_start',case=name,status=profile_action('/start_profile')))
                    try:
                        if case.get('profile_short_before_long'):
                            short_concurrency = case.get('profile_short_concurrency', 16)
                            row = cohort(base, testkey,
                                prompts['512-0'][:short_concurrency], 128)
                            row.update(kind='profile_short_cohort', case=name)
                            emit(out, row)
                        if case.get('profile_single_before_long'):
                            row = cohort(base, testkey, prompts['512-0'][:1], 128)
                            row.update(kind='profile_single_cohort', case=name)
                            emit(out, row)
                        profile_prompts=([item['prompt'] for item in long_inputs]
                            if case.get('cuda_profile_context')=='resident_long' else prompts['512-0'])
                        row=cohort(base,testkey,profile_prompts,128)
                        row.update(kind='profile_cohort',case=name)
                        emit(out,row)
                    finally:
                        emit(out,dict(kind='profile_stop',case=name,status=profile_action('/stop_profile')))
                emit(out, dict(kind='case_complete', case=name))
                qualified_cases.add(name)
                if admission_group:
                    qualified_groups.add(admission_group)
            except Exception as error:
                emit(out, dict(kind='case_failed', case=name, error=str(error),
                    log_tail=logpath.read_text(errors='replace')[-6000:].replace(testkey, '[REDACTED]') if logpath.exists() else ''))
                if str(error) == 'GPU kernel precheck failed':
                    raise
            finally:
                sample_stop.set()
                if sampler:
                    sampler.join(timeout=6)
                (ROOT/'results'/(run+'_'+name+'_gpu.json')).write_text(json.dumps(samples))
                if current is not None:
                    stop_group(current)
                current = None
    except BaseException as error:
        failed = True
        emit(out, dict(kind='run_interrupted', error=str(error)))
    finally:
        if current is not None:
            stop_group(current)
        if paused:
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            restored = launch(original, env, cwd,
                              ROOT / 'logs' / (run + '_restore.log'))
            startup = public_ready(restored, key, llama=original_llama)
            if original_llama:
                slots = public_http('/slots', key)
                if len(slots) < 16 or {slot.get('n_ctx') for slot in slots} != {240128}:
                    raise RuntimeError('Original full-context capacity was not restored')
                verification = dict(passed=True, slots=len(slots), context=240128)
            else:
                verification = smoke(key)
            emit(out, dict(kind='restored', pid=restored.pid, startup_s=startup,
                           original_command_reused=True,
                           api_key_preserved=bool(key), verification=verification))
    groups = {case['admission_group'] for case in plan['cases'] if case.get('admission_group')}
    required_cases = {case['name'] for case in plan['cases'] if not case.get('admission_group')}
    passed = not failed and groups <= qualified_groups and required_cases <= qualified_cases
    emit(out, dict(kind='run_complete', run=run, passed=passed,
                   qualified_cases=sorted(qualified_cases)))
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
