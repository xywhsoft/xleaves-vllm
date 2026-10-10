"""Compare completed exact-input layer216 with current-source qualified profiles."""
import argparse
import json
from pathlib import Path
import statistics
import time

from pipeline_layer216 import PINS, RUN
from profile_locality201 import checked_locality
from qualified_profile189 import audit, sha

KINDS = ('cohort', 'independent_resident_cohort')
METRICS = ('aggregate_common_decode_tps', 'average_common_decode_request_tps',
           'aggregate_e2e_output_tps', 'average_e2e_request_tps',
           'median_request_decode_tps', 'median_ttft_s')


def close(actual, expected):
    assert actual is not None and abs(actual - expected) <= 1e-8 * max(1, abs(expected))


def audit_row(row):
    count = row['concurrency']
    assert row['output_tokens_per_request'] == 1024 and row['output_tokens'] == count * 1024
    requests = row['requests']
    assert len(requests) == count
    for request in requests:
        events = request['token_events']
        assert events and all(n > 0 for _, n in events) and sum(n for _, n in events) == 1024
        assert all(a[0] <= b[0] for a, b in zip(events, events[1:]))
        assert request['first'] == events[0][0]
        assert request['last'] == events[-1][0]
        assert request['start'] <= request['first'] <= request['last'] <= request['end']
        close(request['ttft_s'], request['first'] - request['start'])
        duration = request['last'] - request['first']
        if duration > 0:
            close(request['decode_tps'], (1024 - events[0][1]) / duration)
        else:
            assert request['decode_tps'] is None
    begin = max(r['first'] for r in requests)
    end = min(r['last'] for r in requests)
    duration = end - begin
    tokens = sum(n for r in requests for at, n in r['token_events'] if begin < at <= end)
    close(row['common_decode_window_s'], duration)
    assert row['common_decode_window_tokens'] == tokens
    assert row['wall_s'] > 0
    assert row['wall_s'] + 1e-7 >= max(r['end'] for r in requests) - min(r['start'] for r in requests)
    close(row['aggregate_e2e_output_tps'], count * 1024 / row['wall_s'])
    close(row['average_e2e_request_tps'], row['aggregate_e2e_output_tps'] / count)
    if duration > 0:
        close(row['aggregate_common_decode_tps'], tokens / duration)
        close(row['average_common_decode_request_tps'], tokens / duration / count)
    else:
        assert row['aggregate_common_decode_tps'] is None and row['average_common_decode_request_tps'] is None
    valid = [r['decode_tps'] for r in requests if r['decode_tps'] is not None]
    if valid:
        close(row['median_request_decode_tps'], statistics.median(valid))
    else:
        assert row['median_request_decode_tps'] is None
    close(row['median_ttft_s'], statistics.median(r['ttft_s'] for r in requests))
    if row['kind'] == 'independent_resident_cohort':
        assert row['cached_prompt_tokens'] == [236944] * count and all(row['needle_passed'])


def groups(rows, gates):
    selected = []
    for gate in gates:
        found = [r for r in rows if r['kind'] == gate['kind'] and r.get('concurrency') == gate['concurrency'] and r.get('repeat') == gate['repeat']]
        assert len(found) == 1
        row = found[0]
        assert row['input_token_arrays_sha256'] == gate['input_token_arrays_sha256'], 'Input token arrays differ; no strict speedup comparison permitted'
        assert row['input_tokens_per_request'] == gate['input_tokens_per_request']
        audit_row(row)
        selected.append(row)
    return selected


def stats(rows, field):
    values = [r[field] for r in rows]
    if any(value is None for value in values):
        return None
    return dict(mean=statistics.mean(values), minimum=min(values), maximum=max(values), samples=values)


def baseline(path, input_path):
    assert sha(input_path) == PINS[str(input_path)]
    proof = json.loads(input_path.read_text())
    gates = proof['matched_cohorts']
    assert len(gates) == 8 and all(g['passed'] for g in gates)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[-1]['kind'] == 'run_complete' and rows[-1]['passed']
    assert any(r['kind'] == 'case_complete' and r['exact_input_cohorts'] == 8 and r['unchanged_saved_states'] for r in rows)
    assert not any(r['kind'] == 'case_failed' for r in rows)
    assert any(r['kind'] == 'restored' and r['verification']['passed'] for r in rows)
    start = next(r for r in rows if r['kind'] == 'start')
    assert start['run'] == RUN and start['proof215_sha256'] == sha(input_path)
    assert not start['hardware_changes']
    for pin in start['file_manifest']:
        assert sha(pin['path']) == pin['sha256']
    command = next(r['command'] for r in rows if r['kind'] == 'case_start')
    for flag, value in [('--split-mode', 'layer'), ('-ub', '256'), ('-b', '512'), ('-np', '16'), ('-c', str(16 * 240128)), ('--cache-type-k', 'q4_0'), ('--cache-type-v', 'q4_0'), ('-fa', 'on')]:
        assert command[command.index(flag) + 1] == value
    ready = next(r for r in rows if r['kind'] == 'case_ready')
    assert ready['slots'] == 16 and ready['context'] == 240128
    selected = groups(rows, gates)
    for row in selected:
        assert all(r['token_ids_available'] and r['timings']['predicted_n'] == 1024 for r in row['requests'])
        expected = 236944 if row['kind'] == 'independent_resident_cohort' else 0
        assert row['cached_prompt_tokens'] == [expected] * row['concurrency']
    return selected, gates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True, type=Path)
    parser.add_argument('--input-proof', required=True, type=Path)
    parser.add_argument('--profile', required=True, type=Path, action='append')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    original, gates = baseline(args.baseline, args.input_proof)
    comparisons = []
    for path in args.profile:
        profile = json.loads(path.read_text())
        validation = audit(profile)
        assert 'input_replay192' in profile
        replay = profile['input_replay192']
        assert sha(replay['proof']) == replay['sha256']
        assert json.loads(Path(replay['proof']).read_text())['whole_case_qualified']
        binding = profile['command_locality201']
        assert sha(binding['path']) == binding['sha256']
        observation = json.loads(Path(binding['path']).read_text())
        locality = checked_locality(profile, observation['locality'] if binding['enabled'] else None)
        assert sha(profile['evidence']['path']) == profile['evidence']['sha256']
        rows = [json.loads(line) for line in Path(profile['evidence']['path']).read_text().splitlines()]
        assert any(r['kind'] == 'case_complete' for r in rows) and not any(r['kind'] == 'case_failed' for r in rows)
        updated = groups(rows, gates)
        results = []
        for kind in KINDS:
            for count in (1, 16):
                before = [r for r in original if r['kind'] == kind and r['concurrency'] == count]
                after = [r for r in updated if r['kind'] == kind and r['concurrency'] == count]
                assert len(before) == len(after) == 2
                measurements = {}
                for field in METRICS:
                    left, right = stats(before, field), stats(after, field)
                    ratio = right['mean'] / left['mean'] if left and right and left['mean'] > 0 else None
                    measurements[field] = dict(layer=left, optimized=right, ratio=ratio, change_pct=(ratio - 1) * 100 if ratio is not None else None)
                results.append(dict(kind=kind, concurrency=count, repeats=2, measurements=measurements, exact_input_token_arrays_matched=True))
        comparisons.append(dict(case=profile['case'], profile=dict(path=str(path), sha256=sha(path)), validation=validation, locality=locality, groups=results))
    result = dict(status='exact_layer229_completed_paired_comparison', utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        baseline=dict(path=str(args.baseline), sha256=sha(args.baseline)), input_proof=dict(path=str(args.input_proof), sha256=sha(args.input_proof)),
        comparisons=comparisons, script_sha256=sha(Path(__file__).resolve()), model_requests_sent=0, source_mutation=False,
        scope='Same input token arrays, concurrency, greedy 1024-output policy and cached236944 long context. Warm long excludes initial prefill. Common-window decode and complete-wall output TPS differ. Different backend quantization is permitted and documented by source profiles; this is not a bit-identical weight/output comparison. Two repeats do not establish statistical significance. Selected candidate source must match current audited source.')
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(status=result['status'], output=str(args.output), sha256=sha(args.output), profiles=len(comparisons))), flush=True)


if __name__ == '__main__':
    main()
