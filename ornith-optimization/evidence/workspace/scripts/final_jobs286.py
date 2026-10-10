"""Final terminal barrier including sampler260 and subsequently registered GPU jobs."""
import argparse
import json
from pathlib import Path
import sys

from deploy_verified228 import JOBS, ROOT
from pipeline_layer216 import process
from qualified_profile189 import sha

REQUIRED = (*JOBS, 'sampler260')
LEGACY = {
    'workspace187': ('pipeline_workspace187.py', 'ef7376c18e5c2dd3153ed96cf9811874363d53947f8449e37fdd4dcf6ca65d48'),
    'numa189': ('pipeline_numa189.py', '7382ec1e38a71ef84e9493e17d7b96ae3f30a95b526203633a31ec33c4c60882'),
    'boundary200': ('pipeline_boundary200.py', '567c1f1cbea4e6e7b0b0c0e43e27f80774d46684169cf4c77fa1c26e1412e327'),
}


def check_job(name):
    assert name and all(c.isalnum() or c in '-_' for c in name), 'Invalid job name'
    path = ROOT / 'cache' / f'{name}-job.json'
    job = json.loads(path.read_text())
    if name in LEGACY:
        filename, pin = LEGACY[name]
        script = ROOT / 'scripts' / filename
        assert job.get('script', str(script)) == str(script), f'Changed legacy driver path {name}'
        assert job.get('script_sha256', job.get('driver_sha256', pin)) == pin, f'Changed legacy driver pin {name}'
    else:
        script, pin = Path(job['script']), job['script_sha256']
    assert script.is_relative_to(ROOT / 'scripts') and script.is_file(), 'Unexpected GPU driver path'
    assert sha(script) == pin, f'Changed GPU driver {name}'
    state, birth = process(job['pid'])
    same = birth == str(job['birth'])
    assert not same or state in ('missing', 'Z', 'X'), f'Active optimization job {name}: {job["pid"]}'
    return dict(job=name, pid=job['pid'], same_birth=same, state=state,
        descriptor=str(path), descriptor_sha256=sha(path), script=str(script), script_sha256=sha(script))


def terminal_jobs():
    names = set(REQUIRED)
    # New whole-model/operator drivers must register resource=gpu_exclusive.
    # Read-only thermal observers have no such marker and do not occupy GPUs.
    for path in (ROOT / 'cache').glob('*-job.json'):
        job = json.loads(path.read_text())
        if job.get('resource') == 'gpu_exclusive':
            names.add(path.name.removesuffix('-job.json'))
    return [check_job(name) for name in sorted(names)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-job')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    try:
        jobs = [check_job(args.check_job)] if args.check_job else terminal_jobs()
        value = dict(status='final286_registered_gpu_jobs_terminal_only', jobs=jobs)
    except AssertionError as error:
        value = dict(status='final286_cpu_admission_rejected', reason=str(error))
    value.update(script_sha256=sha(__file__), torch_imported='torch' in sys.modules,
        source_mutation_performed=False, gpu_tests_run=False, public_deployment_performed=False,
        scope='Actual process birth/state and driver pins only. Does not validate model quality, restoration, profile, performance or authorize deployment.')
    args.output.write_text(json.dumps(value, indent=2))
    print(json.dumps(dict(status=value['status'], output=str(args.output), sha256=sha(args.output))), flush=True)


if __name__ == '__main__':
    main()
