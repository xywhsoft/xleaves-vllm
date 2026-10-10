"""After200 terminates/restores, qualify209 scheduler source on whole Ornith."""
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from qualified_profile189 import audit, sha
from pipeline_numa189 import locality
from manage_onecat71 import api_key, check_idle, discover

ROOT=Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
TARGET=ROOT/'src/onecat-padded-kv140'
REL='vllm/v1/core/sched/scheduler.py'
BASE='441af671c70dd0764a0012630e05227426baf5b92ffece33a3c3b0108a2cc8cf'
NEW='0d0ac21791373bfb55deeaa75622d5eb7df4efea5848da04867a061f1af44632'
PROOF='c3bd152b7d2a3e6c93e80b86580e5b5aecdc6c2cd30a5bfa0abea2c21316b293'
RUN='scheduler_guard210'
PARENT,BIRTH=879195,'231983992'

def process(pid):
    path=Path(f'/proc/{pid}/stat')
    if not path.exists():return 'missing',None
    fields=path.read_text().split(') ',1)[1].split()
    return fields[0],fields[19]

def rows(path):
    if not path.exists():return []
    result=[]
    lines=path.read_text().splitlines()
    for index,line in enumerate(lines):
        try:result.append(json.loads(line))
        except json.JSONDecodeError:assert index==len(lines)-1
    return result

def complete(entries):
    return {x['case'] for x in entries if x['kind']=='case_complete'}-{x['case'] for x in entries if x['kind']=='case_failed'}

def record(status,**values):
    value=dict(status=status,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**values)
    (ROOT/'results/cache210-status.json').write_text(json.dumps(value,indent=2))
    print(json.dumps(value),flush=True)

def prerequisites():
    pins={'workspace187-plan.json':'830e3c10cc1f47865ed533ee6ed858e8c8aef5b8c3f942379ceb36eb79d59470',
          'pipeline_boundary200.py':'567c1f1cbea4e6e7b0b0c0e43e27f80774d46684169cf4c77fa1c26e1412e327',
          'onecat_qualified_runner139.py':'01f36917804c07758408f7ffd0c789aebe47c612ea2c4491e7cec19baecd5fcb',
          'freeze_qualified193.py':'232ba98404824272ec23c29c8262e9fe4260d3d4de428b69bf3707c5a8000ec1',
          'profile_locality201.py':'04be019c1547749a13899dbc6500a80a31a37e85954eae13250f44f553d6e83a'}
    for name,pin in pins.items():assert sha(ROOT/'scripts'/name)==pin
    assert sha(TARGET/REL)==BASE
    assert sha(ROOT/'scripts/scheduler_cache209_candidate.py')==NEW
    proof_path=ROOT/'results/scheduler-source209-cpu-proof.json'
    assert sha(proof_path)==PROOF
    value=json.loads(proof_path.read_text())
    assert value['status']=='scheduler_source209_cpu_gates_passed'
    assert value['candidate_sha256']==NEW and value['serving_base_sha256']==BASE
    assert not value['cuda_initialized'] and not value['serving_source_mutation'] and value['model_requests_sent']==0
    assert sha(ROOT/'scripts/scheduler_source209_cpu.py')==value['script_sha256']
    return value

def main():
    assert not (ROOT/'scripts/cache210-plan.json').exists()
    assert not (ROOT/'results'/f'{RUN}.jsonl').exists()
    proof=prerequisites()
    record('waiting_terminal_restored200',parent_pid=PARENT,parent_birth=BIRTH,no_gpu_work=True,source_mutation_performed=False)
    deadline=time.monotonic()+48*3600
    while process(PARENT)[0] not in ['missing','Z','X']:
        assert process(PARENT)[1]==BIRTH and time.monotonic()<deadline
        time.sleep(10)
    for pid in [825019,847219]:assert process(pid)[0] in ['missing','Z','X']
    proof=prerequisites()
    datasets=[]
    for plan_name,journal_name,locality_name in [
        ('workspace187-plan.json','workspace187_high_and_full_capacity',None),
        ('numa189-plan.json','workspace187_numa189','numa189-worker-locality.json'),
        ('boundary200-plan.json','workspace187_boundary200','boundary200-worker-locality.json')]:
        plan_path=ROOT/'scripts'/plan_name
        journal=ROOT/'results'/f'{journal_name}.jsonl'
        entries=rows(journal)
        assert entries and entries[-1]['kind']=='run_complete'
        assert any(x['kind']=='restored' and x['verification']['passed'] for x in entries)
        plan=json.loads(plan_path.read_text())
        observations=json.loads((ROOT/'results'/locality_name).read_text()) if locality_name and (ROOT/'results'/locality_name).exists() else {}
        datasets.append((plan_path,journal,plan,entries,observations,locality_name))
    frozen=[]
    choices=[]
    lock=(ROOT/'cache/gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    found=discover()
    assert found is not None
    check_idle(found,api_key(found[1],found[2]))
    for plan_path,journal,plan,entries,observations,locality_name in datasets:
        for case in plan['cases']:
            name=case['name']
            if name not in complete(entries):continue
            bound='--numa-bind' in case['command']
            if bound and not observations.get(name,{}).get('passed'):continue
            output=ROOT/'results'/f'qualified-cache210-parent-{name}.json'
            subprocess.run([sys.executable,str(ROOT/'scripts/freeze_qualified193.py'),'--journal',str(journal),'--plan',str(plan_path),'--case',name,'--output',str(output)],cwd=ROOT/'scripts',check=True)
            audit(json.loads(output.read_text()))
            checked=output.with_name(output.stem+'-command201.json')
            command=[sys.executable,str(ROOT/'scripts/profile_locality201.py'),'--profile',str(output),'--output',str(checked)]
            if bound:command+=['--locality',str(ROOT/'results'/locality_name)]
            subprocess.run(command,cwd=ROOT/'scripts',check=True)
            frozen.append(dict(profile=str(checked),sha256=sha(checked)))
            rates=[x['aggregate_e2e_output_tps'] for x in entries if x.get('case')==name and x['kind']=='diverse_cohort']
            assert len(rates)==2
            choices.append((sum(rates)/2,case,plan_path,journal))
    assert choices
    _,high,high_plan,high_journal=max(choices,key=lambda x:x[0])
    baseline=datasets[0][2]
    full=next(c for c in baseline['cases'] if c['minimum_full_contexts']==24)
    cases=[]
    for label,original,parent_plan,parent_journal in [('high',high,high_plan,high_journal),('full24',full,datasets[0][0],datasets[0][1])]:
        case=copy.deepcopy(original)
        case['name']+='_guard210'
        case['admission_group']='guard210_'+label
        case['guard209_prerequisite']=dict(proof=str(ROOT/'results/scheduler-source209-cpu-proof.json'),proof_sha256=PROOF,
            script=str(ROOT/'scripts/scheduler_source209_cpu.py'),script_sha256=proof['script_sha256'],
            candidate=str(ROOT/'scripts/scheduler_cache209_candidate.py'),candidate_sha256=NEW,
            baseline_source_sha256=BASE,parent_case=original['name'],parent_plan=str(parent_plan),parent_journal=str(parent_journal),frozen_parents=frozen,
            scope='Only scheduler waiting-prefix eviction protection with pressure release/retry. Failed24 TPS excluded; correctness/capacity/MM/cache/cold/performance gates remain unchanged.')
        cases.append(case)
    # One additional largest tested full-capacity design, only if prior whole
    # qualification or the same mixed-cache gate identifies it as applicable.
    bp,bj,bplan,brows,bobs,blocality=datasets[2]
    applicable=[]
    for case in bplan['cases']:
        name=case['name']
        if case['minimum_full_contexts']<=24:continue
        if '--numa-bind' in case['command'] and not bobs.get(name,{}).get('passed'):continue
        failed=[x for x in brows if x.get('case')==name and x['kind']=='case_failed']
        if name in complete(brows) or (failed and failed[0]['error']=='Mixed long/short residency/cache/correctness gate failed'):
            applicable.append(case)
    if applicable:
        original=max(applicable,key=lambda x:x['minimum_full_contexts'])
        case=copy.deepcopy(original)
        case['name']+='_guard210'
        case['admission_group']='guard210_boundaryfull'
        case['guard209_prerequisite']=copy.deepcopy(cases[1]['guard209_prerequisite'])
        case['guard209_prerequisite'].update(parent_case=original['name'],parent_plan=str(bp),parent_journal=str(bj))
        cases.append(case)
    plan=copy.deepcopy(baseline)
    plan.update(run=RUN,cases=cases,tests=copy.deepcopy(cases[0]['tests']),wait_for_gpu_lock=False,
        long_inputs=str(ROOT/'scripts/long_independent_prompts28_194.json'))
    backup=ROOT/'scripts/baseline_scheduler_cache210.py'
    assert not backup.exists()
    backup.write_bytes((TARGET/REL).read_bytes())
    installed=False
    keep=False
    runner=None
    try:
        (TARGET/REL).write_bytes((ROOT/'scripts/scheduler_cache209_candidate.py').read_bytes())
        installed=True
        env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PRE_COMMIT_HOME':str(ROOT/'cache/pre-commit'),
             'UV_CACHE_DIR':str(ROOT/'cache/uv'),'GOPROXY':'https://goproxy.cn,direct'}
        attempts=[]
        for attempt in range(2):
            result=subprocess.run([str(TARGET/'.venv/bin/python'),'-m','pre_commit','run','--files',REL],cwd=TARGET,env=env,
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,timeout=300)
            log=ROOT/'logs'/f'cache210-source-hooks-{attempt}.log'
            log.write_text(result.stdout)
            attempts.append(dict(returncode=result.returncode,log=str(log),sha256=sha(log)))
            if result.returncode==0:break
        hooks=dict(passed=attempts[-1]['returncode']==0 and sha(TARGET/REL)==NEW,attempts=attempts,candidate_sha256=sha(TARGET/REL))
        hook_path=ROOT/'results/cache210-source-hooks.json'
        hook_path.write_text(json.dumps(hooks,indent=2))
        assert hooks['passed'],hooks
        subprocess.run(['git','diff','--check'],cwd=TARGET,check=True)
        plan['candidate_source_sha256'][REL]=NEW
        for relative,pin in plan['candidate_source_sha256'].items():assert sha(TARGET/relative)==pin
        for case in cases:
            case['guard209_prerequisite'].update(hooks=str(hook_path),hooks_sha256=sha(hook_path))
        path=ROOT/'scripts/cache210-plan.json'
        path.write_text(json.dumps(plan,indent=2))
        record('running_actual_guard210',plan=str(path),plan_sha256=sha(path),cases=[c['name'] for c in cases],frozen_parents=frozen,source_mutation_performed=True)
        fcntl.flock(lock,fcntl.LOCK_UN)
        runner=subprocess.Popen([sys.executable,str(ROOT/'scripts/onecat_qualified_runner139.py'),str(path)],cwd=ROOT/'scripts')
        observed={}
        journal=ROOT/'results'/f'{RUN}.jsonl'
        while runner.poll() is None:
            entries=rows(journal)
            for case in cases:
                name=case['name']
                if '--numa-bind' not in case['command'] or name in observed:continue
                if not any(x.get('case')==name and x['kind']=='case_ready' for x in entries):continue
                checks=locality(ROOT/'logs'/f'{RUN}_{name}.log')
                if checks is not None:
                    observed[name]=dict(workers=checks,passed=all(c['passed'] for c in checks))
                    (ROOT/'results/cache210-worker-locality.json').write_text(json.dumps(observed,indent=2))
            time.sleep(5)
        entries=rows(journal)
        assert entries[-1]['kind']=='run_complete'
        assert any(x['kind']=='restored' and x['verification']['passed'] for x in entries)
        qualified=sorted(complete(entries))
        qualified=[n for n in qualified if '--numa-bind' not in next(c['command'] for c in cases if c['name']==n) or observed.get(n,{}).get('passed')]
        keep=bool(qualified)
        record('complete' if keep else 'failed_qualification',qualified_cases=qualified,source_retained_for_final_review=keep,
            source_mutation_performed=True,runner_returncode=runner.returncode,public_deployment_performed=False)
    finally:
        assert runner is None or runner.poll() is not None,'Do not mutate source while a model runner remains active'
        if installed and not keep:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            (TARGET/REL).write_bytes(backup.read_bytes())
            assert sha(TARGET/REL)==BASE
            record('failed_candidate_restored_scheduler_base',source_mutation_performed=True,restored_sha256=BASE)
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()

if __name__=='__main__':
    try:main()
    except BaseException as error:
        record('failed',error_type=type(error).__name__,error=str(error))
        raise
