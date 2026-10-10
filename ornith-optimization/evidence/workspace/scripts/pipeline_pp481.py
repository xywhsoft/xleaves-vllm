"""Bounded PP2/TP1 screen; never labels short-only data as240K qualification."""
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
RUNNER = ROOT/'scripts/onecat_qualified_runner139.py'


def make_plan(helper):
    parent = json.loads((ROOT/'scripts/adaptive393-plan.json').read_text())
    case = copy.deepcopy(next(x for x in parent['cases'] if x['diverse_count']==160))
    case['name'] = 'pp481_tp1_pp2_c160_ctx240128_short_screen'
    command = case['command']
    command[command.index('--tensor-parallel-size')+1] = '1'
    command += ['--pipeline-parallel-size','2','--distributed-executor-backend','mp']
    command[command.index('--kv-cache-memory-bytes')+1]='15032385536'
    assert command[command.index('--max-model-len')+1]=='240128'
    case['screen481_scope'] = 'Startup reported16full-context capacity, MM/text functionality and short throughput only; not a full-context qualification.'
    for key in ('long_pilot','long_residency','image_after_long','require_fresh_images_after_long','cold_concurrent_inputs','cuda_profile'):
        case[key] = False
    case['admission_group'] = 'pp481_screen_only'
    case.pop('adaptive370_prerequisite',None)
    plan = copy.deepcopy(parent)
    plan.update(run='pp481_short_screen',cases=[case],wait_for_gpu_lock=False,long_residency=False,
                tests=[dict(concurrency=n,input_tokens=512,output_tokens=1024,repeats=2) for n in (1,16,160)])
    case['tests'] = copy.deepcopy(plan['tests'])
    plan['candidate_source_sha256']['vllm/v1/sample/ops/topk_topp_sampler.py'] = helper
    plan['qualification_scope'] = case['screen481_scope']
    return plan


def prerequisites():
    assert sha(RUNNER)=='01f36917804c07758408f7ffd0c789aebe47c612ea2c4491e7cec19baecd5fcb'
    assert sha(ROOT/'scripts/adaptive393-plan.json')=='7b2fbf9cdd9a3b2fe823bee0e834ac83c6c0a3b0b150f4948930f6db0fd6bb9c'
    check_job('freeze416');check_job('stoch465')
    stochastic=json.loads((ROOT/'results/stoch465-short-sampling-proof.json').read_text())
    assert stochastic['screen_completed'] and stochastic['failure'] is None
    assert stochastic['restored']['verification']==dict(passed=True,slots=16,context=240128)
    old=json.loads((ROOT/'results/pp445-short-screen-proof.json').read_text())
    assert not old['screen_completed'] and not old['measurements']
    assert old['restored']['verification']==dict(passed=True,slots=16,context=240128)
    prior = restored_runs()
    helper = sha(TARGET/'vllm/v1/sample/ops/topk_topp_sampler.py')
    assert helper in ('ab2ee9e52cc229ced835e840cb90cea3a1d7ed2d927d86507b670b532a56d52f',
                      'afa1e2cb242649c33c42b58d02709dc3984846d7ea0e89417bd05024148f4322')
    assert sha(TARGET/'vllm/v1/sample/ops/topk_topp_triton.py')=='7e1e04a7af6badc8012cfbcae920495672a57d3a30b07d547adc82f8d8177c52'
    if (ROOT/'cache/freeze440-job.json').exists():check_job('freeze440')
    return prior,helper


def main():
    prior,helper = prerequisites()
    own_path=ROOT/'cache/pp481-job.json'
    deadline=time.monotonic()+15
    while not own_path.exists():
        assert time.monotonic()<deadline;time.sleep(.05)
    own=json.loads(own_path.read_text());state,birth=process(os.getpid())
    assert own['pid']==os.getpid() and str(own['birth'])==birth and own['script_sha256']==sha(__file__)
    names=set(REQUIRED)
    for p in (ROOT/'cache').glob('*-job.json'):
        if json.loads(p.read_text()).get('resource')=='gpu_exclusive':names.add(p.name.removesuffix('-job.json'))
    jobs=[check_job(n) for n in sorted(names) if n!='pp481']
    lock=(ROOT/'cache/gpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    found=discover();assert found is not None and found[4]
    key=api_key(found[1],found[2]);check_idle(found,key)
    plan=make_plan(helper)
    plan_path=ROOT/'scripts/pp481-short-plan.json';journal=ROOT/'results/pp481_short_screen.jsonl'
    proof_path=ROOT/'results/pp481-short-screen-proof.json'
    assert not plan_path.exists() and not journal.exists() and not proof_path.exists()
    for rel,pin in plan['candidate_source_sha256'].items():assert sha(TARGET/rel)==pin
    plan_path.write_text(json.dumps(plan,indent=2))
    child=None;failure=None
    def cancel(signum,frame):raise KeyboardInterrupt(f'signal{signum}')
    signal.signal(signal.SIGTERM,cancel);signal.signal(signal.SIGINT,cancel)
    try:
        fcntl.flock(lock,fcntl.LOCK_UN)
        child=subprocess.Popen([sys.executable,str(RUNNER),str(plan_path)],cwd=ROOT/'scripts',start_new_session=True)
        child.wait(timeout=2400)
    except BaseException as e:
        failure=dict(type=type(e).__name__,message=str(e))
    finally:
        signal.signal(signal.SIGTERM,signal.SIG_IGN);signal.signal(signal.SIGINT,signal.SIG_IGN)
        if child is not None and child.poll() is None:
            os.kill(child.pid,signal.SIGTERM);child.wait(timeout=900)
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rows=[json.loads(x) for x in journal.read_text().splitlines()]
        assert rows[-1]['kind']=='run_complete'
        restored=next(x for x in reversed(rows) if x['kind']=='restored')
        assert restored['verification']==dict(passed=True,slots=16,context=240128)
        assert restored['original_command_reused'] and restored['api_key_preserved']
        current=discover();assert current is not None and current[4] and current[1]==found[1]
        assert api_key(current[1],current[2])==key
        for rel,pin in plan['candidate_source_sha256'].items():assert sha(TARGET/rel)==pin
        complete=any(x['kind']=='case_complete' for x in rows) and not any(x['kind']=='case_failed' for x in rows)
        cohorts=[x for x in rows if x['kind'] in ('cohort','diverse_cohort')]
        value=dict(status='pp481_short_screen_completed' if failure is None else 'pp481_short_screen_failed',
            screen_completed=complete,failure=failure,script_sha256=sha(__file__),job_descriptor_sha256=sha(own_path),
            plan=str(plan_path),plan_sha256=sha(plan_path),journal=str(journal),journal_sha256=sha(journal),
            restored=restored,current_public=dict(pid=current[0],original_command_reused=True,api_key_preserved=True),
            terminal_prior_jobs=jobs,terminal_prior_restorations=prior,helper_sha256=helper,
            measurements=[{k:x[k] for k in ('kind','concurrency','aggregate_common_decode_tps','aggregate_e2e_output_tps','input_token_arrays_sha256')} for x in cohorts],
            full_context_qualified=False,public_deployment_performed=False,serving_source_mutation=False,
            scope='PP445 actual graph-capture workspace OOM; reserve2.5GiB via KV16.5GiB to14GiB. Startup must still report16full-context capacity. Short screen only. Requires independent239K/resident/mixed/MM-after-long/cold gates before selection. Actual failedstartup or completedSSE evidence retained; no universal maximum claim.')
        proof_path.write_text(json.dumps(value,indent=2));print(json.dumps(dict(status=value['status'],screen_completed=complete,proof=str(proof_path),sha256=sha(proof_path))),flush=True)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    if failure:raise RuntimeError(failure)


if __name__=='__main__':main()
