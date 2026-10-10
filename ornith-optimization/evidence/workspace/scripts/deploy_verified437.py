"""Final deployment including registered GPU jobs and sampler restoration."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import time

from deploy_verified214 import ROOT, invoke, manager, validate as original_validate
from pipeline_numa189 import process
from profile_locality201 import checked_locality
from qualified_profile189 import sha
from query_numa_anon221 import inspect
from final_jobs286 import terminal_jobs
from adaptive_profile413 import extend as adaptive_profile
from noise_profile436 import extend as noise_profile
from deploy_verified383 import restored_runs as previous_restored_runs
from final_jobs286 import check_job
from deploy_verified228 import restored_runs as original_restored_runs

READER = '765e24f4c974f6eb95bd43f356866449f08baa7dfa8aacfad9ba3a932d2c48cf'
CPU = '29eb36ccd2ba941182c4923ecca822d99e18b330241982b426ccdb5b6fe2c27e'
JOBS = ('workspace187', 'numa189', 'boundary200', 'cache210', 'layer216',
        'numa223-observer', 'numa226-freezer')


def restored_runs():
    result = previous_restored_runs()
    check_job('freeze416')
    frozen = ROOT/'results/freeze416-qualified-profiles.json'
    bound = json.loads(frozen.read_text())
    assert bound['status'] == 'freeze416_actual_complete_profiles_compared'
    assert sha(ROOT/'scripts/freeze_adaptive416.py') == bound['script_sha256']
    result.append(dict(journal=str(frozen),sha256=sha(frozen),status=bound['status']))
    for name, script, pin, statuses in (
        ('adaptive393-whole-model-proof.json','pipeline_adaptive393.py','a363ab7afa2ede5426d4b9ac354956741aa2eac4a8f3635b4166367e05f2523a',('adaptive393_whole_model_completed',)),
        ('sampler408-gpu-operator-proof.json','pipeline_sampler408.py','3c90ea18fd9bf7266ef7220545f7f5431300e538091ff280098eb4c9276edfad',('sampler408_gpu_operator_checks_passed','sampler408_gpu_operator_checks_failed'))):
        path = ROOT/'results'/name; value = json.loads(path.read_text())
        assert value['status'] in statuses
        assert sha(ROOT/'scripts'/script) == value['script_sha256'] == pin
        job_name = 'adaptive393' if name.startswith('adaptive393') else 'sampler408'
        check_job(job_name)
        assert sha(ROOT/'cache'/f'{job_name}-job.json') == value['job_descriptor_sha256']
        restored = value['restored']
        assert restored['verification'] == dict(passed=True,slots=16,context=240128)
        assert restored['original_command_reused'] and restored['api_key_preserved']
        if name.startswith('adaptive393'):
            assert value['failure'] is None and value['qualified_cases'] and value['source_retained_for_final_review']
            assert value['final_source_sha256'] == '7e1e04a7af6badc8012cfbcae920495672a57d3a30b07d547adc82f8d8177c52'
            assert sha(value['journal']) == value['journal_sha256']
            assert value['current_public']['original_command_reused'] and value['current_public']['api_key_preserved']
        else:
            assert not value['serving_source_mutation']
            for entry in value['completed']:
                assert sha(entry['output']) == entry['sha256'] and sha(entry['log']) == entry['log_sha256']
        result.append(dict(journal=str(path),sha256=sha(path),status=value['status'],restored=restored))
    noise = ROOT/'results/noise420-whole-model-proof.json'
    if noise.exists():
        check_job('noise420'); value = json.loads(noise.read_text())
        assert value['status'] in ('noise420_whole_model_completed','noise420_whole_model_failed')
        assert value['script_sha256'] == sha(ROOT/'scripts/pipeline_noise420.py') == 'd48f4ec8d3fc09dca8bf8f37d5f9442762f8dcab57f977f1733ebb0c9eeb5cf4'
        assert value['job_descriptor_sha256'] == sha(ROOT/'cache/noise420-job.json')
        for stage in value['stages']:
            assert sha(stage['journal']) == stage['journal_sha256'] and sha(stage['plan']) == stage['plan_sha256']
            restored = stage['restored']
            assert restored['verification']['passed'] and restored['verification']['context'] == 240128
            assert restored['original_command_reused'] and restored['api_key_preserved']
        assert value['current_public']['original_command_reused'] and value['current_public']['api_key_preserved']
        helper = ROOT/'src/onecat-padded-kv140/vllm/v1/sample/ops/topk_topp_sampler.py'
        assert sha(helper) == value['final_source_sha256']
        assert sha(ROOT/'src/onecat-padded-kv140/vllm/v1/sample/ops/topk_topp_triton.py') == value['final_topk_source_sha256']
        if value['source_retained_for_final_review']:
            assert value['failure'] is None and value['promotion']['passed'] and value['qualified_cases']
            assert value['final_source_sha256'] == 'afa1e2cb242649c33c42b58d02709dc3984846d7ea0e89417bd05024148f4322'
        else:
            assert value['final_source_sha256'] == 'ab2ee9e52cc229ced835e840cb90cea3a1d7ed2d927d86507b670b532a56d52f'
        result.append(dict(journal=str(noise),sha256=sha(noise),status=value['status'],retained=value['source_retained_for_final_review']))
    else:
        assert not (ROOT/'cache/noise420-job.json').exists()
    return result


def validate(profile):
    checked = original_validate(profile)
    evidence = [json.loads(x) for x in Path(profile['evidence']['path']).read_text().splitlines()]
    spec = next(x['qualification_spec'] for x in evidence if x['kind'] == 'case_start')
    assert 'adaptive370_prerequisite' in spec, 'Selected final profile must carry current370 source qualification'
    if 'noise400_prerequisite' in spec:
        assert 'noise_validation436' in profile and 'adaptive_validation436' in profile
        extended = noise_profile(profile)
        checked['noise_source_provenance'] = extended['noise_validation436']
        checked['adaptive_source_provenance'] = extended['adaptive_validation436']
    else:
        assert 'adaptive_validation413' in profile
        assert sha(Path(profile['cwd'])/'vllm/v1/sample/ops/topk_topp_sampler.py') == 'ab2ee9e52cc229ced835e840cb90cea3a1d7ed2d927d86507b670b532a56d52f'
        extended = adaptive_profile(profile)
        checked['adaptive_source_provenance'] = extended['adaptive_validation413']
    manifest = {x['path']:x for x in profile['file_manifest']}
    assert all(manifest.get(x['path']) == x for x in extended['file_manifest'])
    return checked


def reader_proof():
    path = ROOT / 'results/numa222-query-cpu-proof.json'
    assert sha(ROOT / 'scripts/query_numa_anon221.py') == READER
    assert sha(path) == CPU
    value = json.loads(path.read_text())
    assert value['reader_sha256'] == READER
    assert sha(ROOT / 'scripts/test_numa_query222.py') == value['script_sha256']
    assert value['status'] == 'numa222_cpu_mixed_mapping_and_real_remote_negative_passed'
    positive, negative = value['fixtures']
    assert positive['passed'] and positive['remote_anon'] == 0
    assert negative['negative'] and not negative['passed'] and negative['remote_anon'] >= 4096
    return dict(reader_sha256=READER, cpu_proof=str(path), cpu_proof_sha256=CPU)


def public_locality(profile, snapshot, public_pid):
    proof = reader_proof()
    text = snapshot.read_text(errors='replace')
    workers = []
    for rank in (0, 1):
        pids = re.findall(r'\(Worker_TP' + str(rank) + r' pid=(\d+)\)', text)
        assert pids, f'No public rank {rank} PID in fresh log'
        pid = int(pids[-1])
        state, birth = process(pid)
        assert state not in ['missing', 'Z', 'X'] and birth is not None
        assert os.getpgid(pid) == public_pid, 'Worker is outside deployed public process group'
        worker = inspect(pid, rank, birth)
        assert worker['passed'] and worker['ambiguous_anonymous_pages'] == 0
        assert worker['query_nodes_argument'] is None and worker['query_flags'] == 0
        assert os.getpgid(pid) == public_pid
        workers.append(worker)
    checked = checked_locality(profile, dict(passed=True, workers=workers))
    return dict(**checked, **proof, log_snapshot=str(snapshot), log_snapshot_sha256=sha(snapshot))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--validate-only', action='store_true')
    args = parser.parse_args()
    profile = json.loads(args.profile.read_text())
    checked = validate(profile)
    proof = reader_proof()
    if args.validate_only:
        print(json.dumps(dict(status='deployment437_profile_audited_only', case=profile['case'], reader_proof=proof, **checked)), flush=True)
        return
    assert args.output is not None and not args.output.exists()
    terminal_jobs()
    lock = (ROOT / 'cache/gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        jobs = terminal_jobs()
        runs = restored_runs()
        checked = validate(profile)
        before = manager.discover()
        assert before is not None
        manager.check_idle(before, manager.api_key(before[1], before[2]))
        log = ROOT / 'logs/onecat-public71.log'
        offset = log.stat().st_size if log.exists() else 0
        invoke([str(args.profile)])
        try:
            current = manager.discover()
            assert current is not None and not current[4]
            result_path = ROOT / 'results/onecat-public-deployment71.json'
            result = json.loads(result_path.read_text())
            assert result['status'] == 'deployed' and result['case'] == profile['case']
            assert result['pid'] == current[0] and result['smoke']['passed']
            observed = dict(enabled=False)
            if checked['command_locality']['enabled']:
                snapshot = ROOT / 'logs' / f'public437-{current[0]}-workers.log'
                assert not snapshot.exists()
                with log.open('rb') as stream:
                    stream.seek(offset)
                    snapshot.write_bytes(stream.read())
                observed = public_locality(profile, snapshot, current[0])
            record = dict(status='verified_public_deployment437', utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                profile=str(args.profile), profile_sha256=sha(args.profile), case=profile['case'], pid=current[0],
                port=8096, api_key_preserved=result['api_key_preserved'], smoke=result['smoke'],
                public_deployment_proof=dict(path=str(result_path), sha256=sha(result_path)),
                actual_public_locality=observed, reader_proof=proof, terminal_optimization_jobs=jobs,
                terminal_restored_runs=runs, registered_jobs_barrier_sha256=sha(ROOT / 'scripts/final_jobs286.py'), wrapper_sha256=sha(Path(__file__).resolve()), **checked)
            args.output.write_text(json.dumps(record, indent=2))
            print(json.dumps(dict(status=record['status'], output=str(args.output), sha256=sha(args.output), case=profile['case'], pid=current[0])), flush=True)
        except BaseException:
            invoke(['--rollback'])
            raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
