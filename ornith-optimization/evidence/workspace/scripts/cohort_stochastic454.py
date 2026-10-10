"""Closed SSE cohort, same139 accounting plus explicit stochastic sampling."""
import concurrent.futures
import hashlib
import json
import statistics
import threading
import time
import urllib.request


def body_temperature(sampling):
    return sampling.get('temperature',0)


def cohort(base, key, prompts, output, sampling=None):
    sampling = dict(sampling or {})
    assert set(sampling) <= {"temperature", "top_k", "top_p"}
    assert output == 1024, "Use the same closed1024-token protocol"
    count = len(prompts)
    gate = threading.Barrier(count + 1)

    def one(index):
        body = dict(model='ornith-1.5-35b', prompt=prompts[index],
                    max_tokens=output, temperature=0, seed=1234,
                    ignore_eos=True, return_token_ids=True, stream=True,
                    stream_options={'include_usage': True})
        body.update(sampling)
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
                    decode_tps=(len(ids)-events[0][1])/(last-first) if last > first else None,
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
    return dict(kind='cohort', concurrency=count, sampling_parameters=dict(temperature=body_temperature(sampling), top_k=sampling.get('top_k',-1), top_p=sampling.get('top_p',1.0)),
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
                median_request_decode_tps=statistics.median(row['decode_tps'] for row in rows if row['decode_tps'] is not None) if any(row['decode_tps'] is not None for row in rows) else None,
                median_ttft_s=statistics.median(row['ttft_s'] for row in rows),
                requests=rows)
