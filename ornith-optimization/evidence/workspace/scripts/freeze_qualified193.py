"""Freeze a fully qualified case without colliding with the live189 pipeline."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

from qualified_profile189 import audit, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--journal', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--case', required=True)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = args.plan.parent.parent
    output = args.output
    bare = output.with_name(output.stem + '-bare156.json')
    dependencies = output.with_name(output.stem + '-dependencies.json')
    proof = output.with_name(output.stem + '-inputs192.json')
    assert all(not path.exists() for path in [output, bare, dependencies, proof])
    subprocess.run([sys.executable, str(root / 'scripts/make_serving_profile156.py'),
                    str(args.journal), args.case, '--output', str(bare)], cwd=root / 'scripts', check=True)
    profile = json.loads(bare.read_text())
    rows = [json.loads(line) for line in Path(profile['evidence']['path']).read_text().splitlines()]
    spec = next(row['qualification_spec'] for row in rows if row['kind'] == 'case_start')
    prerequisite = spec['workspace186_prerequisite']
    manifest = {entry['path']: entry for entry in profile['file_manifest']}
    for field, pin in [('proof', 'proof_sha256'), ('hooks', 'hooks_sha256'),
                       ('external_interface', 'external_interface_sha256')]:
        path = Path(prerequisite[field])
        assert sha(path) == prerequisite[pin]
        manifest[str(path)] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
    for path in [Path(__file__).resolve(), root / 'scripts/qualified_profile189.py',
                 root / 'scripts/review_source_drift180.py']:
        manifest[str(path)] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
    profile['file_manifest'] = sorted(manifest.values(), key=lambda entry: entry['path'])
    profile['extended_validation189'] = audit(profile)
    dependencies.write_text(json.dumps(profile, indent=2))
    subprocess.run([sys.executable, str(root / 'scripts/replay_inputs192.py'),
                    '--plan', str(args.plan), '--journal', str(args.journal), '--case', args.case,
                    '--profile', str(dependencies), '--output', str(proof),
                    '--qualified-output', str(output)], cwd=root / 'scripts', check=True)
    final = json.loads(output.read_text())
    checked = audit(final)
    print(json.dumps(dict(status='qualified_profile193_frozen_and_inputs_replayed',
        output=str(output), sha256=sha(output), input_proof=str(proof),
        input_proof_sha256=sha(proof), **checked)), flush=True)


if __name__ == '__main__':
    main()
