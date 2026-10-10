"""Reversible helper400 TP176 screen after actual paired511; always restoreBASE."""
import copy
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from deploy_verified437 import restored_runs
from final_jobs286 import REQUIRED, check_job
from manage_onecat71 import api_key, check_idle, discover
from pipeline_layer216 import process
from qualified_profile189 import sha

ROOT = Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
TARGET = ROOT/'src/onecat-padded-kv140'
RUNNER = ROOT/'scripts/run_stochastic493.py'


def make_plan(helper):
    parent=json.loads((ROOT/'scripts/adaptive393-plan.json').read_text())
    cases=[]
    for count in (176,):
        case=copy.deepcopy(next(x for x in parent['cases'] if x['diverse_count']==count))
        case['name']=f'noise529_tp2_c{count}_ctx240128_short_sampling_screen'
        case['screen529_scope']='Newhelper400, sameTP176 configuration; shortsampling screen only. Actual511 numerical/pressure checks passed; no whole240Kqualification.'
        for key in ('long_pilot','long_residency','image_after_long','require_fresh_images_after_long','cold_concurrent_inputs','cuda_profile'):
            case[key]=False
        case.pop('admission_group',None)
        case.pop('adaptive370_prerequisite',None)
        case['env']['VLLM_SERVER_DEV_MODE']='1'
        case['tests']=[dict(concurrency=n,input_tokens=512,output_tokens=1024,repeats=2) for n in (1,16,count)]
        cases.append(case)
    plan=copy.deepcopy(parent)
    plan.update(run='noise529_short_sampling',cases=cases,wait_for_gpu_lock=False,long_residency=False,
        tests=[dict(concurrency=n,input_tokens=512,output_tokens=1024,repeats=2) for n in (1,16,176)])
    plan['candidate_source_sha256']['vllm/v1/sample/ops/topk_topp_sampler.py']=helper
    plan['qualification_scope']='ShortTP176 helper400 screen, threebalancedmodes. Always reverthelperBASE, originalpublic restored. Whole176/16full context gates required beforedeploy ifallmode gains pass.'
    return plan


def prerequisites():
    assert sha(RUNNER)=='a138032251c76a822f9d0aad469961fad721b1703995062ddb5119dfd48f7c14'
    assert sha(ROOT/'scripts/cohort_stochastic454.py')=='51327f0bc7372d5a752a5c48d4a2a3e1a908520a5df6d4fd268005235d8af625'
    assert sha(ROOT/'results/stochastic455-cpu-protocol.json')=='3a1411720d62b6881a524c741237cbb7b27e1f8e4ab5df3039c4892e4f5d3353'
    check_job('chain447');check_job('pp445');check_job('stoch465');check_job('chain482');check_job('stoch490');check_job('pp491');check_job('sampler511');check_job('chain521')
    gpu_path=ROOT/'results/sampler511-gpu-operator-proof.json'
    assert sha(gpu_path)=='6d0f07f9cac1bb8ac3c74a071fe4830ba2528f589961c3b2589b4567168d7e2e'
    gpu=json.loads(gpu_path.read_text())
    assert gpu['status']=='sampler511_gpu_operator_checks_passed' and gpu['failure'] is None and len(gpu['completed'])==6
    assert all(x['ratio']<=1.05 for x in gpu['fast_latency_ratios'])
    assert gpu['restored']['verification']==dict(passed=True,slots=16,context=240128)
    for entry in gpu['completed']:
        assert sha(entry['output'])==entry['sha256'] and sha(entry['log'])==entry['log_sha256']
    previous=json.loads((ROOT/'results/stoch465-closure492.json').read_text())
    assert not previous['screen_completed'] and previous['sampling_cohort_count']==0
    assert previous['restored']['verification']==dict(passed=True,slots=16,context=240128)
    assert previous['current_public']['original_command_reused'] and previous['current_public']['api_key_preserved']
    pp=json.loads((ROOT/'results/pp445-short-screen-proof.json').read_text())
    assert pp['restored']['verification']==dict(passed=True,slots=16,context=240128)
    assert pp['current_public']['original_command_reused'] and pp['current_public']['api_key_preserved']
    assert not pp['serving_source_mutation']
    assert sha(ROOT/'scripts/adaptive393-plan.json')=='7b2fbf9cdd9a3b2fe823bee0e834ac83c6c0a3b0b150f4948930f6db0fd6bb9c'
    check_job('freeze416')
    prior = restored_runs()
    helper = sha(TARGET/'vllm/v1/sample/ops/topk_topp_sampler.py')
    assert helper in ('ab2ee9e52cc229ced835e840cb90cea3a1d7ed2d927d86507b670b532a56d52f',
                      'afa1e2cb242649c33c42b58d02709dc3984846d7ea0e89417bd05024148f4322')
    assert sha(TARGET/'vllm/v1/sample/ops/topk_topp_triton.py')=='7e1e04a7af6badc8012cfbcae920495672a57d3a30b07d547adc82f8d8177c52'
    if (ROOT/'cache/freeze440-job.json').exists():check_job('freeze440')
    return prior,helper


def main():
    prior,helper = prerequisites()
    own_path=ROOT/'cache/noise529-job.json'
    deadline=time.monotonic()+15
    while not own_path.exists():
        assert time.monotonic()<deadline;time.sleep(.05)
    own=json.loads(own_path.read_text());state,birth=process(os.getpid())
    assert own['pid']==os.getpid() and str(own['birth'])==birth and own['script_sha256']==sha(__file__)
    names=set(REQUIRED)
    for p in (ROOT/'cache').glob('*-job.json'):
        if json.loads(p.read_text()).get('resource')=='gpu_exclusive':names.add(p.name.removesuffix('-job.json'))
    jobs=[check_job(n) for n in sorted(names) if n!='noise529']
    lock=(ROOT/'cache/gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    found=discover();assert found is not None and found[4]
    key=api_key(found[1],found[2]);check_idle(found,key)
    candidate=ROOT/'scripts/topk_topp_sampler400_candidate.py';new='afa1e2cb242649c33c42b58d02709dc3984846d7ea0e89417bd05024148f4322'
    assert sha(candidate)==new
    plan=make_plan(new)
    plan_path=ROOT/'scripts/noise529-short-plan.json';journal=ROOT/'results/noise529_short_sampling.jsonl'
    proof_path=ROOT/'results/noise529-short-sampling-proof.json'
    assert not plan_path.exists() and not journal.exists() and not proof_path.exists()
    for rel,pin in plan['candidate_source_sha256'].items():
        assert sha(TARGET/rel)==(helper if rel=='vllm/v1/sample/ops/topk_topp_sampler.py' else pin)
    backup=ROOT/'scripts/baseline_sampler_noise529.py';assert not backup.exists()
    backup.write_bytes((TARGET/'vllm/v1/sample/ops/topk_topp_sampler.py').read_bytes())
    plan_path.write_text(json.dumps(plan,indent=2))
    child=None;failure=None
    def cancel(signum,frame):raise KeyboardInterrupt(f'signal{signum}')
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    try:
        (TARGET/'vllm/v1/sample/ops/topk_topp_sampler.py').write_bytes(candidate.read_bytes())
        hook_log=ROOT/'logs/noise529-source-hooks.log';assert not hook_log.exists()
        hooks=subprocess.run([str(TARGET/'.venv/bin/python'),'-m','pre_commit','run','--files','vllm/v1/sample/ops/topk_topp_sampler.py'],cwd=TARGET,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PRE_COMMIT_HOME':str(ROOT/'cache/pre-commit'),'UV_CACHE_DIR':str(ROOT/'cache/uv'),'GOPROXY':'https://goproxy.cn,direct'},capture_output=True,text=True,timeout=300)
        hook_log.write_text(hooks.stdout+hooks.stderr)
        assert hooks.returncode==0 and sha(TARGET/'vllm/v1/sample/ops/topk_topp_sampler.py')==new,'Exactcandidate source hooks failed'
        subprocess.run(['git','diff','--check'],cwd=TARGET,check=True)
        fcntl.flock(lock,fcntl.LOCK_UN)
        child=subprocess.Popen([sys.executable,str(RUNNER),str(plan_path)],cwd=ROOT/'scripts',start_new_session=True)
        child.wait(timeout=3600)
    except BaseException as e:
        failure=dict(type=type(e).__name__,message=str(e))
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        if child is not None and child.poll() is None:
            os.kill(child.pid,signal.SIGTERM);child.wait(timeout=900)
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        observed_helper=sha(TARGET/'vllm/v1/sample/ops/topk_topp_sampler.py')
        (TARGET/'vllm/v1/sample/ops/topk_topp_sampler.py').write_bytes(backup.read_bytes())
        assert sha(TARGET/'vllm/v1/sample/ops/topk_topp_sampler.py')==helper
        assert observed_helper==new,'Candidate source changed during screen; BASE restored'
        rows=[json.loads(x) for x in journal.read_text().splitlines()]
        assert rows[-1]['kind']=='run_complete'
        restored=next(x for x in reversed(rows) if x['kind']=='restored')
        assert restored['verification']==dict(passed=True,slots=16,context=240128)
        assert restored['original_command_reused'] and restored['api_key_preserved']
        current=discover();assert current is not None and current[4] and current[1]==found[1]
        assert api_key(current[1],current[2])==key
        for rel,pin in plan['candidate_source_sha256'].items():assert sha(TARGET/rel)==(helper if rel=='vllm/v1/sample/ops/topk_topp_sampler.py' else pin)
        complete=sum(x['kind']=='case_complete' for x in rows)==1 and not any(x['kind']=='case_failed' for x in rows) and rows[-1]['passed']
        sample_path=ROOT/'results/noise529_short_sampling-sampling.jsonl'
        sampling_rows=[json.loads(x) for x in sample_path.read_text().splitlines()]
        assert sampling_rows[-1]['kind']=='sampling_wrapper_terminal'
        cohorts=[x for x in sampling_rows if x['kind']=='cohort']
        for row in cohorts:
            from compare_exact_layer229 import audit_row
            audit_row(row)
        if complete:
            assert len(cohorts)==24
            assert {x['sampling_mode'] for x in cohorts}=={'greedy','random_full','random_filtered'}
        reference_path=ROOT/'results/stoch490-short-sampling-proof.json'
        reference=json.loads(reference_path.read_text());assert reference['screen_completed'] and reference['sampling_cohort_count']==48
        baseline=[json.loads(x) for x in Path(reference['sampling_journal']).read_text().splitlines()];baseline=[x for x in baseline if x['kind']=='cohort' and '_c160_' in x['case']]
        gates=[]
        if complete:
            for mode in ('greedy','random_full','random_filtered'):
                for field in ('aggregate_common_decode_tps','aggregate_e2e_output_tps'):
                    b=[x[field] for x in baseline if x['input_kind']=='natural' and x['sampling_mode']==mode]
                    a=[x[field] for x in cohorts if x['input_kind']=='natural' and x['sampling_mode']==mode]
                    assert len(a)==len(b)==2;ratio=statistics.mean(a)/statistics.mean(b)
                    gates.append(dict(mode=mode,field=field,ratio=ratio,minimum=1.03,passed=ratio>=1.03))
                b=[x['aggregate_common_decode_tps'] for x in baseline if x['concurrency']==16 and x['sampling_mode']==mode]
                a=[x['aggregate_common_decode_tps'] for x in cohorts if x['concurrency']==16 and x['sampling_mode']==mode]
                assert len(a)==len(b)==2;ratio=statistics.mean(a)/statistics.mean(b)
                gates.append(dict(mode=mode,field='short16_decode',ratio=ratio,minimum=.95,passed=ratio>=.95))
        value=dict(status='noise529_short_sampling_completed' if failure is None else 'noise529_short_sampling_failed',
            screen_completed=complete,failure=failure,script_sha256=sha(__file__),job_descriptor_sha256=sha(own_path),
            plan=str(plan_path),plan_sha256=sha(plan_path),journal=str(journal),journal_sha256=sha(journal),
            restored=restored,current_public=dict(pid=current[0],original_command_reused=True,api_key_preserved=True),
            terminal_prior_jobs=jobs,terminal_prior_restorations=prior,helper_sha256=helper,sampling_journal=str(sample_path),sampling_journal_sha256=sha(sample_path),sampling_cohort_count=len(cohorts),
            source_mutation=True,source_retained=False,final_helper_sha256=helper,candidate_sha256=new,
            baseline_sampling_proof=str(reference_path),baseline_sampling_proof_sha256=sha(reference_path),
            short_promotion_passed=complete and all(x['passed'] for x in gates),promotion_gates=gates,
            source_hooks_log=str(hook_log),source_hooks_log_sha256=sha(hook_log),
            measurements=[{k:x[k] for k in ('kind','case','input_kind','sampling_mode','concurrency','aggregate_common_decode_tps','aggregate_e2e_output_tps','input_token_arrays_sha256','cached_prompt_tokens','occurrence_for_exact_input')} for x in cohorts],
            full_context_qualified=False,public_deployment_performed=False,serving_source_mutation=True,
            scope='Private loopback test service alone enablesDEV_MODE1 to mount/reset_prefix_cache; qualified publicprofile remainsDEV_MODE0. Actual uncached512-input/1024-output sampling screen; 3 balanced modes per exact cohort. 393/416 whole-model qualifications remain separate. No stochastic240K performance measured here.')
        proof_path.write_text(json.dumps(value,indent=2));print(json.dumps(dict(status=value['status'],screen_completed=complete,proof=str(proof_path),sha256=sha(proof_path))),flush=True)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    if failure:raise RuntimeError(failure)


if __name__=='__main__':main()
