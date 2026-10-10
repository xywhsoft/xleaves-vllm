"""Verify actual token arrays and bind immutable replay inputs to qualified profiles."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def token_hash(prompts):
    return hashlib.sha256(json.dumps(prompts, separators=(',', ':')).encode()).hexdigest()


def committed_rows(path):
    raw = Path(path).read_bytes()
    prefix = raw[:raw.rfind(b'\n') + 1]
    values = [json.loads(line) for line in prefix.splitlines() if line.strip()]
    starts = [index for index, row in enumerate(values) if row['kind'] == 'start']
    assert len(starts) == 1, 'Use one exact non-restarted qualification journal'
    return values[starts[0]:], dict(path=str(path), committed_bytes=len(prefix),
        committed_byte_prefix_sha256=hashlib.sha256(prefix).hexdigest())


def replay(plan_path, journal_path, case):
    plan = json.loads(plan_path.read_text())
    rows, journal = committed_rows(journal_path)
    assert rows[0]['run'] == plan['run']
    specs = [item for item in plan['cases'] if item['name'] == case]
    assert len(specs) == 1
    spec = specs[0]
    selected = [row for row in rows if row.get('case') == case]
    assert sum(row['kind'] == 'case_start' for row in selected) == 1
    measured_spec = next(row['qualification_spec'] for row in selected if row['kind'] == 'case_start')
    assert measured_spec == {key: value for key, value in spec.items() if key not in {'command', 'env'}}
    short_info = rows[0]['short_inputs']
    short_path = Path(short_info['path'])
    assert sha(short_path) == short_info['sha256']
    assert short_path.stat().st_size == short_info['bytes']
    short = json.loads(short_path.read_text())
    long_path = Path(plan['long_inputs'])
    long_items = json.loads(long_path.read_text())
    long = [item['prompt'] for item in long_items[:spec['long_input_count']]]
    assert len(long) == spec['minimum_full_contexts']
    assert all(len(prompt) == 239104 for prompt in long)
    assert len({tuple(prompt[:16]) for prompt in long}) == len(long)
    diverse_path = Path(spec['extra_diverse_cohorts'])
    diverse = json.loads(diverse_path.read_text())
    cold_path = Path(spec['cold_concurrent_inputs'])
    cold = [item['prompt'] for item in json.loads(cold_path.read_text())[:16]]
    assert len(cold) == 16 and all(len(prompt) == 239104 for prompt in cold)
    assert len({tuple(prompt[:16]) for prompt in cold}) == 16
    assert not {tuple(prompt[:16]) for prompt in long} & {tuple(prompt[:16]) for prompt in cold}
    checks = []
    for row in selected:
        kind = row['kind']
        if kind == 'warmup':
            prompts = short['warm']
        elif kind == 'cohort':
            prompts = short[f'{row["input_tokens_per_request"]}-{row["repeat"]}'][:row['concurrency']]
        elif kind == 'diverse_cohort':
            prompts = [item['prompt'] for item in diverse[f'512-{row["repeat"]}']][:row['concurrency']]
        elif kind == 'long_pilot':
            prompts = long[:1]
        elif kind == 'independent_cold_prefill':
            prompts = long[row['index']:row['index'] + 1]
        elif kind == 'independent_resident_cohort':
            prompts = long[:row['concurrency']]
        elif kind == 'mixed_long_short_cohort':
            prompts = long[:row['long_requests']] + [item['prompt'] for item in diverse[f'512-{row["repeat"]}']][:row['short_requests']]
        elif kind == 'cold_concurrent_long_cohort':
            prompts = cold
        else:
            continue
        expected = token_hash(prompts)
        assert expected == row['input_token_arrays_sha256'], (kind, row.get('repeat'), 'Input hash mismatch')
        assert len(prompts) == row['concurrency']
        assert [len(prompt) for prompt in prompts] == row['input_tokens_per_request_lengths']
        assert row['output_tokens'] == row['concurrency'] * row['output_tokens_per_request']
        assert len(row['requests']) == row['concurrency']
        assert all(request['usage']['completion_tokens'] == row['output_tokens_per_request'] for request in row['requests'])
        checks.append(dict(kind=kind, repeat=row.get('repeat'), index=row.get('index'),
                           concurrency=len(prompts), expected_and_measured_input_sha256=expected, passed=True))
    assert checks
    paths = [plan_path, short_path, long_path, diverse_path, cold_path, Path(__file__).resolve()]
    manifest = [dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)) for path in paths]
    complete = any(row['kind'] == 'case_complete' for row in selected)
    failed = any(row['kind'] == 'case_failed' for row in selected)
    return dict(status='actual_materialized_inputs_replayed192', case=case,
        case_complete=complete, case_failed=failed, whole_case_qualified=complete and not failed,
        journal=journal, plan_sha256=sha(plan_path), input_manifest=manifest,
        checks=checks, checked_cohort_kinds=dict(Counter(row['kind'] for row in checks)),
        model_requests_sent=0, hardware_changes=False,
        scope='CPU-only reconstruction of recorded token arrays. Pending rows are not measurements. Partial replay does not qualify a model or accept staged TPS.')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--journal', required=True, type=Path)
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--qualified-output', type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    assert bool(args.profile) == bool(args.qualified_output)
    if args.qualified_output:
        assert not args.qualified_output.exists()
    started = time.perf_counter()
    proof = replay(args.plan, args.journal, args.case)
    profile = None
    if args.profile:
        from qualified_profile189 import audit
        profile = json.loads(args.profile.read_text())
        assert proof['whole_case_qualified'] and profile['case'] == args.case
        proof['profile_validation'] = audit(profile)
        if args.case.endswith('_numa189'):
            locality_path = args.plan.parent.parent / 'results/numa189-worker-locality.json'
            locality = json.loads(locality_path.read_text())[args.case]
            assert locality['passed'] and len(locality['workers']) == 2
            assert all(worker['passed'] for worker in locality['workers'])
            proof['input_manifest'].append(dict(path=str(locality_path), bytes=locality_path.stat().st_size, sha256=sha(locality_path)))
    proof['cpu_audit_elapsed_s'] = time.perf_counter() - started
    proof['utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    args.output.write_text(json.dumps(proof, indent=2))
    if profile:
        manifest = {entry['path']: entry for entry in profile['file_manifest']}
        for entry in proof['input_manifest']:
            manifest[entry['path']] = entry
        manifest[str(args.output)] = dict(path=str(args.output), bytes=args.output.stat().st_size, sha256=sha(args.output))
        profile['file_manifest'] = sorted(manifest.values(), key=lambda entry: entry['path'])
        profile['input_replay192'] = dict(proof=str(args.output), sha256=sha(args.output), checks=len(proof['checks']))
        from qualified_profile189 import audit
        audit(profile)
        args.qualified_output.write_text(json.dumps(profile, indent=2))
    print(json.dumps(dict(status=proof['status'], case=proof['case'], whole_case_qualified=proof['whole_case_qualified'],
        checked_cohorts=len(proof['checks']), checked_kinds=proof['checked_cohort_kinds'],
        cpu_audit_elapsed_s=proof['cpu_audit_elapsed_s'], output=str(args.output), sha256=sha(args.output),
        qualified_profile_written=bool(profile))), flush=True)


if __name__ == '__main__':
    main()
