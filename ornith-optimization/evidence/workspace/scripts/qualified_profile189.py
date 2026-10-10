"""Freeze qualified187/189 inputs, including Python dependencies and admission gates."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from manage_onecat71 import verify_profile
from review_source_drift180 import compare

ROOT = Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(profile):
    checked = verify_profile(profile)
    source = compare(Path(profile['cwd']), profile['file_manifest'])
    assert source['source_matches'], source
    rows = [json.loads(line) for line in Path(profile['evidence']['path']).read_text().splitlines()]
    start = next(row for row in rows if row['kind'] == 'case_start')
    spec = start['qualification_spec']
    maximum, full = profile['maximum_active_sequences'], profile['minimum_full_contexts']
    prefixes = [row for row in rows if row['kind'] == 'independent_cold_prefill']
    assert len(prefixes) == full and [row['index'] for row in prefixes] == list(range(full))
    assert all(row['needle_passed'] and row['input_tokens_per_request'] == 239104 for row in prefixes)
    diverse = [row for row in rows if row['kind'] == 'diverse_cohort']
    assert len(diverse) == 2 and {row['repeat'] for row in diverse} == {0, 1}
    assert all(row['concurrency'] == maximum for row in diverse)
    short = [row for row in rows if row['kind'] == 'cohort' and row['concurrency'] == maximum]
    assert len(short) == 2 and {row['repeat'] for row in short} == {0, 1}
    mixed = [row for row in rows if row['kind'] == 'mixed_long_short_cohort']
    assert len(mixed) == 2 and {row['repeat'] for row in mixed} == {0, 1}
    assert all(row['long_requests'] == spec['mixed_long_count'] == 16
               and row['short_requests'] == spec['mixed_short_count'] == maximum - 16
               and row['concurrency'] == maximum for row in mixed)
    for row in rows:
        if row['kind'] not in {'cohort', 'diverse_cohort', 'independent_resident_cohort',
                               'mixed_long_short_cohort', 'cold_concurrent_long_cohort'}:
            continue
        assert row['output_tokens_per_request'] == 1024
        assert row['output_tokens'] == row['concurrency'] * 1024
        assert len(row['requests']) == row['concurrency']
        assert all(request['usage']['completion_tokens'] == 1024 for request in row['requests'])
    images = [row for row in rows if row['kind'] == 'max_image_after_residency']
    assert len(images) == 1 and images[0]['resident_contexts'] == full
    assert images[0]['image_count'] == 2 and images[0]['pixels_per_image'] == 589824
    return dict(**checked, source_coverage=source, verified_mixed_active_sequences=maximum,
                verified_full_contexts=full, uncached_full_context_concurrency=16)


def freeze(journal, case):
    bare = ROOT / 'results' / f'qualified189-{case}-bare156.json'
    output = ROOT / 'results' / f'qualified189-{case}-profile.json'
    assert not bare.exists() and not output.exists()
    subprocess.run([sys.executable, str(ROOT / 'scripts/make_serving_profile156.py'),
                    str(journal), case, '--output', str(bare)], cwd=ROOT / 'scripts', check=True)
    profile = json.loads(bare.read_text())
    rows = [json.loads(line) for line in Path(profile['evidence']['path']).read_text().splitlines()]
    spec = next(row for row in rows if row['kind'] == 'case_start')['qualification_spec']
    prerequisite = spec['workspace186_prerequisite']
    manifest = {entry['path']: entry for entry in profile['file_manifest']}
    for field, hash_field in [('proof', 'proof_sha256'), ('hooks', 'hooks_sha256'),
                              ('external_interface', 'external_interface_sha256')]:
        path = Path(prerequisite[field])
        assert sha(path) == prerequisite[hash_field]
        manifest[str(path)] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
    for field in ['extra_diverse_cohorts', 'cold_concurrent_inputs']:
        path = Path(spec[field])
        manifest[str(path)] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
    path = Path(__file__).resolve()
    manifest[str(path)] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
    profile['file_manifest'] = sorted(manifest.values(), key=lambda entry: entry['path'])
    profile['extended_validation189'] = audit(profile)
    output.write_text(json.dumps(profile, indent=2))
    return dict(path=str(output), sha256=sha(output), **profile['extended_validation189'])
