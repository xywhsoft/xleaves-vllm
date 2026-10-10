"""CPU orchestration; register each next GPU job only after actual prerequisites end."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from final_jobs286 import check_job,terminal_jobs
from pipeline_layer216 import process
from qualified_profile189 import sha

ROOT=Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
PYTHON=ROOT/'src/onecat-padded-kv140/.venv/bin/python'
STATUS=ROOT/'results/chain447-status.json'


def record(status,**fields):
    value=dict(status=status,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),torch_imported='torch' in sys.modules,
               gpu_work_in_controller=False,serving_source_mutation_in_controller=False,public_deployment_performed=False,**fields)
    STATUS.write_text(json.dumps(value,indent=2));print(json.dumps(value),flush=True)


def wait_actual(name):
    descriptor=ROOT/'cache'/f'{name}-job.json';job=json.loads(descriptor.read_text())
    assert sha(job['script'])==job['script_sha256']
    state,birth=process(job['pid'])
    record('waiting_actual_registered_parent',job=name,pid=job['pid'],birth=job['birth'],actual_state=state,birth_matches=birth==str(job['birth']))
    deadline=time.monotonic()+24*3600
    while state not in ('missing','Z','X') and birth==str(job['birth']):
        assert time.monotonic()<deadline,f'{name} did not finish within24h'
        time.sleep(10);state,birth=process(job['pid'])
    return check_job(name)


def register(name,script,pin,resource):
    assert sha(script)==pin;ast.parse(script.read_text(encoding='utf-8-sig'))
    descriptor=ROOT/'cache'/f'{name}-job.json';log=ROOT/'logs'/f'{name}-pipeline.log'
    assert not descriptor.exists() and not log.exists()
    with log.open('x') as stream:
        child=subprocess.Popen([str(PYTHON),str(script)],cwd=ROOT/'scripts',env=dict(os.environ),stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    state,birth=process(child.pid);assert birth is not None
    job=dict(pid=child.pid,birth=birth,script=str(script),script_sha256=pin,resource=resource,log=str(log),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_mutation_performed=False)
    temporary=descriptor.with_suffix('.json.tmp');temporary.write_text(json.dumps(job,indent=2));temporary.replace(descriptor)
    record('registered_next_job_after_actual_prerequisites',job=name,pid=child.pid,descriptor_sha256=sha(descriptor))
    return job


def main():
    record('waiting_existing393_408_416_no_future_gpu_registration')
    wait_actual('adaptive393');wait_actual('sampler408');wait_actual('freeze416')
    terminal_jobs()
    gpu=json.loads((ROOT/'results/sampler408-gpu-operator-proof.json').read_text())
    assert gpu['status']=='sampler408_gpu_operator_checks_passed' and gpu['failure'] is None,'Actual408 result requires review before installing helper400'
    frozen=json.loads((ROOT/'results/freeze416-qualified-profiles.json').read_text())
    assert frozen['status']=='freeze416_actual_complete_profiles_compared','Actual416 freeze requires review'
    launcher=ROOT/'scripts/launch_noise420.sh'
    assert sha(launcher)=='657780a1f446736632ba5ce8a62f46c1a1c7ae2e4ad5d8f33170f4ef3d74b20d'
    assert not (ROOT/'cache/noise420-job.json').exists()
    result=subprocess.run(['bash','-c',launcher.read_text(encoding='utf-8-sig')],cwd=ROOT/'scripts',capture_output=True,text=True,timeout=180)
    log=ROOT/'logs/chain447-launch420.log';assert not log.exists();log.write_text(result.stdout+result.stderr)
    assert result.returncode==0,'Actual420 admission failed; inspectchain447-launch420.log'
    wait_actual('noise420');terminal_jobs()
    whole=json.loads((ROOT/'results/noise420-whole-model-proof.json').read_text())
    assert whole['status']=='noise420_whole_model_completed' and whole['failure'] is None,'Actual420 failure requires review before later tests'
    register('freeze440',ROOT/'scripts/freeze_noise440.py','1bec30a9ed5f817a62a5da1c72239035dda437f744f47ff9fff7b50d95140e35','cpu_evidence')
    wait_actual('freeze440')
    proof=json.loads((ROOT/'results/freeze440-qualified-profiles.json').read_text())
    assert proof['status'] in ('freeze440_noise_not_retained_use416','freeze440_noise_complete_profiles_compared'),'Actual440 freeze requires review'
    terminal_jobs()
    register('pp445',ROOT/'scripts/pipeline_pp445.py','a08d27c522f5b9c0e9fe4d04cb6902409bd2cacbaab577069ea5eb931172bfa1','gpu_exclusive')
    wait_actual('pp445');terminal_jobs()
    pp=json.loads((ROOT/'results/pp445-short-screen-proof.json').read_text())
    assert pp['restored']['verification']==dict(passed=True,slots=16,context=240128)
    assert pp['current_public']['original_command_reused'] and pp['current_public']['api_key_preserved']
    record('chain447_completed_actual_results_require_selection_and_final_deploy',pp_status=pp['status'],pp_screen_completed=pp['screen_completed'],
           pp_proof_sha256=sha(ROOT/'results/pp445-short-screen-proof.json'),noise_retained=whole['source_retained_for_final_review'])
    assert 'torch' not in sys.modules


if __name__=='__main__':
    try:main()
    except BaseException as error:
        record('chain447_stopped_requires_actual_result_review',error_type=type(error).__name__,error=str(error));raise
