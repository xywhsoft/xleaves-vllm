"""Freeze one fully qualified serving case and its exact deployment inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from manage_onecat71 import option, verify_profile

ROOT = Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('journal', type=Path)
    parser.add_argument('case')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'results/qualified-serving-profile71.json')
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.journal.read_text().splitlines()
            if line.strip()]
    starts = [index for index, row in enumerate(rows) if row.get('kind') == 'start']
    if starts:
        rows = rows[starts[-1]:]
    selected = [row for row in rows if row.get('case') == args.case]
    case_starts = [row for row in selected if row['kind'] == 'case_start']
    if len(case_starts) != 1:
        raise RuntimeError('Select exactly one case from the latest lab run')
    start = case_starts[0]
    spec = start['qualification_spec']
    if spec.get('bootstrap_only'):
        raise RuntimeError('A compatibility bootstrap cannot become a serving profile')
    command = list(start['command'])
    index = command.index('--api-key')
    del command[index:index + 2]
    command = option(option(command, '--host', '0.0.0.0'), '--port', 8096)
    manifest = {entry['path']: entry for entry in start['artifact_manifest']}
    for entry in start['source_manifest']:
        path = str(Path(spec['cwd']) / entry['path'])
        manifest[path] = {**entry, 'path': path}
    # Bind the validation and deployment programs as well as serving code.
    for name in ['qualification_artifacts78.py', 'manage_onecat71.py',
                 'onecat_qualified_runner139.py', 'make_serving_profile156.py']:
        path = ROOT / 'scripts' / name
        manifest[str(path)] = {'path': str(path), 'bytes': path.stat().st_size,
                               'sha256': sha(path)}
    evidence = ROOT / 'results' / ('frozen-qualification78-' + args.case + '.jsonl')
    # Retain raw token accounting and gates. Exclude unrelated failed fallback cases.
    encoded = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected)
    if evidence.exists() and evidence.read_text() != encoded:
        raise RuntimeError('An existing frozen evidence file differs; use a new case name')
    evidence.write_text(encoded)
    profile = {'status': 'qualified', 'case': args.case, 'cwd': spec['cwd'],
               'command': command, 'env': start['runtime_environment'],
               'minimum_full_contexts': spec['minimum_full_contexts'],
               'maximum_active_sequences': int(command[command.index('--max-num-seqs') + 1]),
               'max_context_per_request': 240128,
               'image_count': 2, 'maximum_pixels_per_image': 589824,
               'source_commit': start['source_commit'],
               'model_identity': start['model_identity'],
               'evidence': {'path': str(evidence), 'sha256': sha(evidence),
                            'source_journal': str(args.journal),
                            'source_journal_sha256_at_freeze': sha(args.journal)},
               'file_manifest': sorted(manifest.values(), key=lambda entry: entry['path']),
               'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    checked = verify_profile(profile)
    args.output.write_text(json.dumps(profile, indent=2))
    print(json.dumps({'status': 'profile_frozen_and_validated',
                      'output': str(args.output), **checked}))


if __name__ == '__main__':
    main()
