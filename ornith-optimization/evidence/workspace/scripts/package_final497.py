"""Immutable final source/evidence package after verified deployment and comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import time

from deploy_verified496 import ROOT, manager, restored_runs, terminal_jobs, validate
from qualified_profile189 import sha

EXTERNAL = Path('/home/xs/codex-work/ornith-opt-20261006/src/onecat/flash-attention-v100')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--deployment', required=True, type=Path)
    parser.add_argument('--comparison', required=True, type=Path)
    parser.add_argument('--extra', action='append', default=[], type=Path)
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--proof', required=True, type=Path)
    args = parser.parse_args()
    assert all(not path.exists() for path in [args.directory, args.archive, args.proof])
    assert args.directory.is_relative_to(ROOT / 'reports') and args.archive.is_relative_to(ROOT / 'reports')
    assert args.proof.is_relative_to(ROOT / 'results')
    assert shutil.disk_usage(ROOT).free > 2_000_000_000
    jobs = terminal_jobs()
    runs = restored_runs()
    profile = json.loads(args.profile.read_text())
    validation = validate(profile)
    deployment = json.loads(args.deployment.read_text())
    assert deployment['status'] == 'verified_public_deployment496'
    assert deployment['profile_sha256'] == sha(args.profile) and deployment['case'] == profile['case']
    assert deployment['smoke']['passed'] and deployment['api_key_preserved']
    assert deployment['registered_jobs_barrier_sha256'] == sha(ROOT / 'scripts/final_jobs286.py')
    assert deployment['wrapper_sha256'] == sha(ROOT / 'scripts/deploy_verified496.py')
    public = manager.discover()
    assert public is not None and not public[4] and public[0] == deployment['pid']
    public_key = manager.api_key(public[1], public[2]).encode()
    assert len(public_key) >= 8
    comparison = json.loads(args.comparison.read_text())
    assert comparison['status'] == 'exact_layer229_completed_paired_comparison'
    assert comparison['script_sha256'] == sha(ROOT / 'scripts/compare_exact_layer229.py')
    assert any(item['case'] == profile['case'] and item['profile']['sha256'] == sha(args.profile) for item in comparison['comparisons'])
    args.directory.mkdir(parents=True)
    copied = {}

    def mapped(path):
        path = Path(path)
        if path.is_relative_to(ROOT):
            return Path('workspace') / path.relative_to(ROOT)
        assert path.is_relative_to(EXTERNAL), f'Unexpected external source path: {path}'
        return Path('external/flash-attention-v100') / path.relative_to(EXTERNAL)

    def copy(path):
        path = Path(path)
        relative = mapped(path)
        target = args.directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            assert sha(target) == sha(path)
        else:
            shutil.copyfile(path, target)
        assert sha(target) == sha(path)
        copied[str(path)] = str(relative)

    for entry in profile['file_manifest']:
        assert sha(entry['path']) == entry['sha256'] and Path(entry['path']).stat().st_size == entry['bytes']
        copy(entry['path'])
    assert sha(profile['evidence']['path']) == profile['evidence']['sha256']
    for path in [args.profile, args.deployment, args.comparison, Path(profile['evidence']['path']),
                 Path(comparison['baseline']['path']), Path(comparison['input_proof']['path'])] + args.extra:
        copy(path)
    for name in ['baseline', 'input_proof']:
        assert sha(comparison[name]['path']) == comparison[name]['sha256']
    for path in [
        'scripts/deploy_verified214.py', 'scripts/deploy_verified228.py', 'scripts/manage_onecat71.py',
        'scripts/compare_exact_layer229.py', 'scripts/package_final438.py',
        'scripts/deploy_verified437.py', 'scripts/deploy_verified383.py', 'scripts/final_jobs286.py',
        'scripts/adaptive_profile413.py', 'scripts/noise_profile436.py', 'scripts/freeze_adaptive416.py',
        'scripts/pipeline_adaptive393.py', 'scripts/baseline_sampler_adaptive393.py',
        'scripts/test_sampler401_cpu.py', 'scripts/test_sampler406_gpu.py', 'scripts/pipeline_sampler408.py',
        'scripts/topk_topp_sampler400_candidate.py',
        'results/sampler401-noise-cpu-contract-proof.json', 'results/sampler407-cpu-admission.json',
        'results/sampler408-gpu-operator-proof.json', 'results/adaptive393-source-hooks.json',
        'results/adaptive393-whole-model-proof.json', 'results/freeze416-qualified-profiles.json',
        'cache/adaptive393-job.json', 'cache/sampler408-job.json', 'cache/freeze416-job.json',
        'scripts/deploy_verified380.py', 'scripts/deploy_verified297.py',
        'scripts/adaptive_profile382.py', 'scripts/pipeline_adaptive381.py',
        'scripts/topk_topp_adaptive370_candidate.py', 'scripts/baseline_sampler_adaptive381.py',
        'scripts/test_sampler371_cpu.py', 'scripts/test_sampler373_gpu.py', 'scripts/pipeline_sampler374.py',
        'results/sampler371-adaptive-cpu-bitwise-proof.json', 'results/sampler374-cpu-prerequisites.json',
        'results/sampler374-gpu-operator-proof.json', 'results/adaptive381-whole-model-proof.json',
        'results/adaptive381-source-hooks.json', 'cache/adaptive381-job.json', 'cache/sampler374-job.json',
        'scripts/pipeline_sampler260.py', 'scripts/test_sampler258_gpu.py',
        'scripts/topk_topp_reference254_candidate.py',
        'results/sampler260-gpu-operator-proof.json',
        'scripts/qualified_profile189.py', 'scripts/review_source_drift180.py',
        'scripts/profile_locality201.py', 'scripts/guard_profile213.py',
        'scripts/query_numa_anon221.py', 'scripts/test_numa_query222.py',
        'results/numa222-query-cpu-proof.json', 'results/deploy228-cpu-prerequisites.json',
        'results/compare229-cpu-prerequisites.json',
        'portability40/native-provenance.json', 'portability40/flash-attention-sm70.patch',
        'portability40/cutlass-header-digests.json', 'build/from-qualified-source60/ornith-native-manifest.json',
    ]:
        copy(ROOT / path)
    for relative in ('scripts/deploy_verified496.py','scripts/package_final472.py','scripts/pipeline_chain447.py','cache/chain447-job.json','results/chain447-status.json',
        'scripts/pipeline_pp445.py','cache/pp445-job.json','results/pp445-short-screen-proof.json',
        'scripts/pipeline_stoch490.py','scripts/run_stochastic493.py','scripts/cohort_stochastic454.py','cache/stoch490-job.json','results/stoch490-short-sampling-proof.json','scripts/pipeline_stoch465.py','scripts/run_stochastic467.py','cache/stoch465-job.json','results/stoch465-closure492.json','results/stoch468-cpu-admission.json','results/stochastic455-cpu-protocol.json'):
        copy(ROOT/relative)
    for relative in ('scripts/deploy_verified496.py','scripts/package_final497.py','scripts/pipeline_pp491.py','scripts/pipeline_chain482.py','cache/pp491-job.json','cache/chain482-job.json','results/chain482-status.json','results/pp491-short-screen-proof.json'):
        copy(ROOT/relative)
    for proof_name in ('pp491-short-screen-proof.json','pp445-short-screen-proof.json','stoch490-short-sampling-proof.json','stoch465-closure492.json'):
        screen=json.loads((ROOT/'results'/proof_name).read_text())
        for field in ('journal','plan','sampling_journal'):
            if field in screen:
                assert sha(screen[field])==screen[field+'_sha256'];copy(screen[field])
    for name in ('sampler260-gpu-operator-proof.json', 'sampler374-gpu-operator-proof.json', 'sampler408-gpu-operator-proof.json'):
        operator = json.loads((ROOT / 'results' / name).read_text())
        for entry in operator['completed']:
            assert sha(entry['output']) == entry['sha256'] and sha(entry['log']) == entry['log_sha256']
            copy(entry['output'])
            copy(entry['log'])
    whole = json.loads((ROOT / 'results/adaptive381-whole-model-proof.json').read_text())
    if whole['journal']:
        assert sha(whole['journal']) == whole['journal_sha256']
        copy(whole['journal'])
    if whole['hooks']:
        assert sha(whole['hooks']['log']) == whole['hooks']['log_sha256']
        copy(whole['hooks']['log'])
    actual = json.loads((ROOT/'results/adaptive393-whole-model-proof.json').read_text())
    assert sha(actual['journal']) == actual['journal_sha256']; copy(actual['journal'])
    assert sha(actual['hooks']['log']) == actual['hooks']['log_sha256']; copy(actual['hooks']['log'])
    copy(ROOT/'scripts/adaptive393-plan.json')
    frozen = json.loads((ROOT/'results/freeze416-qualified-profiles.json').read_text())
    assert sha(frozen['comparison']) == frozen['comparison_sha256']; copy(frozen['comparison'])
    for item in frozen['profiles']:
        assert sha(item['profile']) == item['sha256']; copy(item['profile'])
        parent = json.loads(Path(item['profile']).read_text())
        assert sha(parent['evidence']['path']) == parent['evidence']['sha256']; copy(parent['evidence']['path'])
    noise_path = ROOT/'results/noise420-whole-model-proof.json'
    if noise_path.exists():
        copy(noise_path)
        for relative in ('scripts/pipeline_noise420.py','scripts/baseline_sampler_noise420.py','scripts/launch_noise420.sh',
                         'cache/noise420-job.json','results/noise420-source-hooks.json'):
            copy(ROOT/relative)
        noise = json.loads(noise_path.read_text())
        if noise['hooks']:
            assert sha(noise['hooks']['log']) == noise['hooks']['log_sha256']; copy(noise['hooks']['log'])
        for stage in noise['stages']:
            assert sha(stage['journal']) == stage['journal_sha256'] and sha(stage['plan']) == stage['plan_sha256']
            copy(stage['journal']);copy(stage['plan'])
    native = json.loads((ROOT / 'build/from-qualified-source60/ornith-native-manifest.json').read_text())
    for item in native:
        assert sha(item['library']) == item['library_sha256']
        copy(item['library'])
    repo = Path(profile['cwd'])
    changed = subprocess.check_output(['git', 'ls-files', '-m', '-o', '--exclude-standard', '-z'], cwd=repo).split(b'\0')
    for name in sorted(set(name for name in changed if name)):
        copy(repo / name.decode())
    patch = subprocess.check_output(['git', 'diff', '--binary', 'HEAD'], cwd=repo)
    (args.directory / 'onecat-serving.patch').write_bytes(patch)
    uv = shutil.which('uv') or str(Path.home() / '.local/bin/uv')
    requirements = subprocess.check_output([uv, 'pip', 'freeze', '--python', str(repo / '.venv/bin/python')], text=True)
    (args.directory / 'runtime-requirements.txt').write_text(requirements)
    metadata = dict(case=profile['case'], base_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip(),
        generated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), validation=validation,
        profile_sha256=sha(args.profile), deployment_sha256=sha(args.deployment), comparison_sha256=sha(args.comparison),
        context_per_request=240128, verified_full_contexts=profile['minimum_full_contexts'], maximum_active_sequences=profile['maximum_active_sequences'],
        original_path_to_snapshot=copied, terminal_jobs=jobs, restored_runs=runs,
        inherited_core_engine_binary_reused=True, model_weights_included=False,
        public_api_credential_absence_verified=True, hardware_changes=False)
    (args.directory / 'source-environment.json').write_text(json.dumps(metadata, indent=2))
    (args.directory / 'REPRODUCTION.txt').write_text('''Qualified 1Cat-vLLM / Ornith1.5 / dual V100 source and evidence
Base repository: https://github.com/1CatAI/1Cat-vLLM.git at source-environment.json base_commit.
Apply onecat-serving.patch and restore the changed/untracked files under workspace/src.
Use uv-managed Python3.12, PyTorch2.10 cu128 and CUDA12.8; update local editable
paths in runtime-requirements.txt to the restored paths. Do not use system pip.
Eight native modules, final qualified sampler source and numerical/operator/whole-model evidence are included. Original core engine/attention
binaries were reused; the complete engine has not been qualified from a clean build.
Portable module rebuild uses tools/build_ornith_sm70.py, pinned CUTLASS4.4.2 and
patched FA2; rerun numerical and model gates after source/dependency changes.
The selected profile records exact command, environment, source and model gates.
Original absolute paths map to packaged files in source-environment.json.
Weights are not bundled. Supply API credentials at runtime.
All requests keep240128 total context. Configured active sequences and verified
full-length contexts are separate limits; do not claim all active slots are240K.
Long warm measurements reuse236944 input tokens and exclude initial prefill.
Cold end-to-end TPS includes input processing, waits and output; common-window
decode TPS is a different metric and can be undefined or a brief overlap sample.
Layer comparison matches exact token arrays/concurrency/greedy1024output policy;
different quantization backends are documented. Two repeats are not significance.
The deployment proof binds public smoke checks, precise worker locality ifenabled,
source hashes and terminal restored test runs. No cooling/power/clock modifications.
Snapshot hashes prove the copied files; they do not guarantee future hardware
conditions, changed dependency behavior or a universal theoretical throughput limit.
''')
    manifest = [dict(path=str(path.relative_to(args.directory)), bytes=path.stat().st_size, sha256=sha(path))
                for path in sorted(args.directory.rglob('*')) if path.is_file()]
    for entry in manifest:
        tail = b''
        with (args.directory / entry['path']).open('rb') as stream:
            while chunk := stream.read(1 << 20):
                combined = tail + chunk
                assert public_key not in combined, 'Public API credential found in package; refusing archive'
                tail = combined[-len(public_key):]
    (args.directory / 'snapshot-manifest.json').write_text(json.dumps(manifest, indent=2))
    args.archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.archive, 'w:gz', compresslevel=3) as archive:
        archive.add(args.directory, arcname=args.directory.name)
    with tarfile.open(args.archive, 'r:gz') as archive:
        packaged_manifest = archive.extractfile(args.directory.name + '/snapshot-manifest.json')
        assert packaged_manifest is not None
        assert packaged_manifest.read() == (args.directory / 'snapshot-manifest.json').read_bytes()
        for entry in manifest:
            stream = archive.extractfile(args.directory.name + '/' + entry['path'])
            assert stream is not None and hashlib.file_digest(stream, 'sha256').hexdigest() == entry['sha256']
    result = dict(status='final497_deployed_source_and_exact_comparison_packaged_verified',
        archive=str(args.archive), sha256=sha(args.archive), bytes=args.archive.stat().st_size,
        verified_members=len(manifest) + 1, public_api_credential_absence_verified=True,
        case=profile['case'], profile_sha256=sha(args.profile),
        deployment_sha256=sha(args.deployment), comparison_sha256=sha(args.comparison),
        script_sha256=sha(Path(__file__).resolve()), model_requests_sent=0, source_mutation=False,
        model_weights_included=False, hardware_changes=False)
    args.proof.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
