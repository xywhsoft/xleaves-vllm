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


def inherited_restored_runs():
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


def restored_runs():
    result=inherited_restored_runs()
    check_job('chain447')
    chain_path=ROOT/'results/chain447-status.json';chain=json.loads(chain_path.read_text())
    assert chain['status']=='chain447_stopped_requires_actual_result_review'
    assert chain['error']=='Actual408 result requires review before installing helper400'
    chain_job=json.loads((ROOT/'cache/chain447-job.json').read_text())
    assert sha(chain_job['script'])==chain_job['script_sha256']=='090df3394904b09578157a746bca4d39e986ee9ae5c1d9cad507f97ebed5f42b'
    sampler=json.loads((ROOT/'results/sampler408-gpu-operator-proof.json').read_text())
    assert sampler['failure']['error']=='Latency regression 0/176: 1.066268661037099'
    assert not (ROOT/'cache/noise420-job.json').exists() and not (ROOT/'cache/freeze440-job.json').exists()
    result.append(dict(journal=str(chain_path),sha256=sha(chain_path),status=chain['status'],review='Sequential408 failed; later paired511 passed. Noise529 reversible whole-model shortscreen reviewed separately.'))
    check_job('chain482')
    follow=json.loads((ROOT/'results/chain482-status.json').read_text())
    assert follow['status']=='chain482_stopped_requires_actual_result_review'
    assert follow['error_type']=='FileNotFoundError' and 'stoch465-short-sampling-proof.json' in follow['error']
    assert not (ROOT/'cache/pp481-job.json').exists()
    closure_path=ROOT/'results/stoch465-closure492.json';closure=json.loads(closure_path.read_text())
    check_job('stoch465')
    assert sha(closure_path)=='50c5c9814a89b35c7e157c3ca64adac21d09341cf2ee9f7ad48f5fb84904e99a'
    assert closure['status']=='stoch465_actual_failed404_and_emit_contract_restoration_closed492'
    assert not closure['screen_completed'] and closure['sampling_cohort_count']==0
    assert sha(closure['journal'])==closure['journal_sha256'] and sha(closure['plan'])==closure['plan_sha256']
    assert sha(closure['pipeline_log'])==closure['pipeline_log_sha256']
    assert closure['restored']['verification']==dict(passed=True,slots=16,context=240128)
    assert closure['restored']['original_command_reused'] and closure['restored']['api_key_preserved']
    result.append(dict(journal=str(closure_path),sha256=sha(closure_path),status=closure['status'],restored=closure['restored']))
    assert sha(ROOT/'scripts/pipeline_chain482.py')=='78f738060fdb8cbc9f379125ab6483568e84489bd1e615332ef7b8d3f1293a55'
    result.append(dict(journal=str(ROOT/'results/chain482-status.json'),sha256=sha(ROOT/'results/chain482-status.json'),status=follow['status']))
    for job_name,proof_name,script,pin in (
        ('pp491','pp491-short-screen-proof.json','pipeline_pp491.py','32c4719e8c58fd73f532a7a50c50c06d861285b4f7f7491f34d366ed8b063941'),
        ('pp445','pp445-short-screen-proof.json','pipeline_pp445.py','a08d27c522f5b9c0e9fe4d04cb6902409bd2cacbaab577069ea5eb931172bfa1'),
        ('stoch490','stoch490-short-sampling-proof.json','pipeline_stoch490.py','9278d1307d0a4d9f06f4e50b4d456a2dac1238f62524903899142c659dbd89d6')):
        check_job(job_name)
        path=ROOT/'results'/proof_name;value=json.loads(path.read_text())
        assert value['script_sha256']==sha(ROOT/'scripts'/script)==pin
        assert value['job_descriptor_sha256']==sha(ROOT/'cache'/f'{job_name}-job.json')
        assert sha(value['journal'])==value['journal_sha256'] and sha(value['plan'])==value['plan_sha256']
        assert value['restored']['verification']==dict(passed=True,slots=16,context=240128)
        assert value['restored']['original_command_reused'] and value['restored']['api_key_preserved']
        assert value['current_public']['original_command_reused'] and value['current_public']['api_key_preserved']
        assert not value['serving_source_mutation'] and not value['public_deployment_performed']
        if job_name=='stoch490':
            assert value['status']=='stoch490_short_sampling_completed' and value['failure'] is None and value['screen_completed']
            assert value['sampling_cohort_count']==48
            assert sha(value['sampling_journal'])==value['sampling_journal_sha256']
            assert sha(ROOT/'scripts/run_stochastic493.py')=='a138032251c76a822f9d0aad469961fad721b1703995062ddb5119dfd48f7c14'
            from compare_exact_layer229 import audit_row
            sampling=[json.loads(x) for x in Path(value['sampling_journal']).read_text().splitlines()]
            assert sampling[-1]['kind']=='sampling_wrapper_terminal'
            cohorts=[x for x in sampling if x['kind']=='cohort']
            assert len(cohorts)==48
            for row in cohorts:
                audit_row(row)
                assert row['cached_prompt_tokens']==[0]*row['concurrency']
                assert row['sampling_mode'] in ('greedy','random_full','random_filtered')
        result.append(dict(journal=str(path),sha256=sha(path),status=value['status'],screen_completed=value['screen_completed'],restored=value['restored'],full_context_qualified=False))
    return result


def final_closed_runs534():
    result=restored_runs()
    check_job('chain521');check_job('sampler511');check_job('noise529');check_job('noise530')
    operator_path=ROOT/'results/sampler511-gpu-operator-proof.json'
    op=json.loads(operator_path.read_text())
    assert sha(operator_path)=='6d0f07f9cac1bb8ac3c74a071fe4830ba2528f589961c3b2589b4567168d7e2e'
    assert op['status']=='sampler511_gpu_operator_checks_passed' and op['failure'] is None
    assert len(op['completed'])==6 and all(x['ratio']<=1.05 for x in op['fast_latency_ratios'])
    assert op['restored']['verification']==dict(passed=True,slots=16,context=240128)
    assert sha(ROOT/'scripts/pipeline_sampler511.py')==op['script_sha256']
    assert sha(ROOT/'cache/sampler511-job.json')==op['job_descriptor_sha256']
    for entry in op['completed']:
        assert sha(entry['output'])==entry['sha256'] and sha(entry['log'])==entry['log_sha256']
    chain_path=ROOT/'results/chain521-status.json';chain=json.loads(chain_path.read_text())
    assert chain['status']=='chain521_completed_actual_balanced_result_requires_wholemodel_decision'
    old_noise=ROOT/'results/noise529-short-sampling-proof.json';old=json.loads(old_noise.read_text());assert sha(old_noise)=='e5c6a46b9ca8f3bfdd7357a85f660e119f9d67bf20def762644e677f67df989e'
    assert not old['screen_completed'] and old['sampling_cohort_count']==0 and not old['source_retained']
    root_cause=ROOT/'results/noise529-root-cause538.json';cause=json.loads(root_cause.read_text());assert sha(root_cause)=='45becf1cbca22c98c1925325db683039c80b91064083c2adc6571965754565c7' and sha(cause['log'])==cause['log_sha256']
    noise_path=ROOT/'results/noise530-short-sampling-proof.json';noise=json.loads(noise_path.read_text())
    assert not noise['source_retained'] and not noise['full_context_qualified'] and not noise['public_deployment_performed']
    assert not noise['short_promotion_passed'],'176 promotion passed: whole-model qualification must precede final selection'
    assert noise['final_helper_sha256']=='ab2ee9e52cc229ced835e840cb90cea3a1d7ed2d927d86507b670b532a56d52f'
    assert sha(ROOT/'scripts/pipeline_noise530.py')==noise['script_sha256']=='685b18e924d69619ecc69e4bb344155a7673d341143e510bf2b21d2ea1c62040'
    assert sha(ROOT/'cache/noise530-job.json')==noise['job_descriptor_sha256']
    for field in ('journal','plan','sampling_journal','source_hooks_log'):
        assert sha(noise[field])==noise[field+'_sha256']
    assert noise['restored']['verification']==dict(passed=True,slots=16,context=240128)
    assert noise['current_public']['original_command_reused'] and noise['current_public']['api_key_preserved']
    if noise['screen_completed']:
        assert noise['sampling_cohort_count']==24 and len(noise['promotion_gates'])==9
        assert any(not x['passed'] for x in noise['promotion_gates'])
    result.extend(dict(journal=str(p),sha256=sha(p),status=v['status']) for p,v in ((operator_path,op),(chain_path,chain),(old_noise,old),(root_cause,cause),(noise_path,noise)))
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
        print(json.dumps(dict(status='deployment534_profile_audited_only', case=profile['case'], reader_proof=proof, **checked)), flush=True)
        return
    assert args.output is not None and not args.output.exists()
    terminal_jobs()
    lock = (ROOT / 'cache/gpu.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        jobs = terminal_jobs()
        runs = final_closed_runs534()
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
                snapshot = ROOT / 'logs' / f'public534-{current[0]}-workers.log'
                assert not snapshot.exists()
                with log.open('rb') as stream:
                    stream.seek(offset)
                    snapshot.write_bytes(stream.read())
                observed = public_locality(profile, snapshot, current[0])
            record = dict(status='verified_public_deployment534', utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
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
