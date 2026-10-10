"""Bind scheduler CPU gates and source hooks to a whole-qualified profile."""
import argparse
import copy
import json
from pathlib import Path
from qualified_profile189 import audit,sha
from profile_locality201 import checked_locality

ROOT=Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
BASE='441af671c70dd0764a0012630e05227426baf5b92ffece33a3c3b0108a2cc8cf'
NEW='0d0ac21791373bfb55deeaa75622d5eb7df4efea5848da04867a061f1af44632'
PROOF209='c3bd152b7d2a3e6c93e80b86580e5b5aecdc6c2cd30a5bfa0abea2c21316b293'
PROOF211='307430ab18bf02fdf8723ee7dd65dc0461eaf6dc5e2ed9461177fe9e3b9b380d'

def check209(value):
    assert value['status']=='scheduler_source209_cpu_gates_passed'
    assert value['candidate_sha256']==NEW and value['serving_base_sha256']==BASE
    assert not value['cuda_initialized'] and not value['serving_source_mutation'] and value['model_requests_sent']==0
    assert len(value['checks'])==3 and {x['full']['budget'] for x in value['checks']}=={32,128,256}
    for x in value['checks']:
        f,h=x['full'],x['hybrid']
        assert f['protected'] and f['all_hot_prefills_scheduled'] and f['cached_tokens_at_admission']==[48]*4
        assert f['all_refs_released'] and f['free_pool_restored']==24
        assert h['protected'] and h['hot_prefills_completed'] and h['admission_cached']==[64]*4
        assert h['groups']==['fullattention','mamba'] and h['all_refs_released'] and h['free_count_restored']==44
    assert all(x['scheduled']==0 and x['running']==0 for x in value['pressure_old']['steps'])
    assert value['pressure_new']['steps'][0]['scheduled']>0 and value['pressure_new']['steps'][0]['cold_computed']==112
    assert any(x.get('pressure_retry') for x in value['pressure_new']['reservations'])
    for key in ['pressure_old','pressure_new','cancellation','exception_cleanup','preemption','partial_touch_exception']:
        assert value[key]['all_refs_released']
    assert value['cancellation']['cancelled_waiting'] and value['cancellation']['hot_prefills_completed']
    assert value['exception_cleanup']['injected_exception']
    assert value['preemption']['preemptions']==1 and value['preemption']['resumed_computed']>=72
    assert value['preemption']['cached_stat_preserved']==64
    assert value['partial_touch_exception']['partial_touch_exception']

def check211(value):
    assert value['status']=='scheduler_layout211_cpu_gates_passed' and value['candidate_sha256']==NEW
    assert not value['cuda_initialized'] and not value['serving_source_mutation'] and value['model_requests_sent']==0
    p=value['protected']
    assert p['groups']==4 and p['mamba_groups']==3 and p['cache_block_tokens']==4016
    assert p['independent_contexts_seeded']==24 and p['input_tokens']==239104
    assert p['initial_cached']==p['admission_cached']==[236944]*16
    assert all(p['hot_prefills_completed']) and p['all_refs_released'] and p['free_count_restored']==p['pool_blocks']-1
    assert value['baseline']['initial_cached']==[236944]*16
    assert sum(x!=236944 for x in value['baseline']['admission_cached'])==2
    assert len(p['guard_ms']['samples'])==21

def cpu_dependencies():
    paths=[]
    for name,pin,check in [('scheduler-source209-cpu-proof.json',PROOF209,check209),
                           ('scheduler-layout211-cpu-proof.json',PROOF211,check211)]:
        path=ROOT/'results'/name
        assert sha(path)==pin
        value=json.loads(path.read_text())
        check(value)
        script=ROOT/'scripts'/('scheduler_source209_cpu.py' if check is check209 else 'scheduler_layout211_cpu.py')
        assert sha(script)==value['script_sha256']
        paths += [path,script]
    layout=ROOT/'scripts/cache211-layout149.json'
    data=json.loads((ROOT/'results/scheduler-layout211-cpu-proof.json').read_text())
    assert sha(layout)==data['layout_sha256']
    candidate=ROOT/'scripts/scheduler_cache209_candidate.py'
    assert sha(candidate)==NEW
    #209 loads pinned AST fixtures from204/205/206, whose main bodies are not
    #executed. Include their exact inputs so the archived gates can be rerun.
    fixture_pins=[('scheduler-cache204-cpu-proof.json','scheduler_cache204_cpu.py','961b4181e543e73e4a6fad733c58241e1d743f9782d542af3f3e2797717bafc2'),
                  ('scheduler-hybrid205-cpu-proof.json','scheduler_hybrid205_cpu.py','46f993fe1520b7469d9e9464513be2a06e2bc2c3c698b57cdb4a4c599d342d67'),
                  ('scheduler-progress206-cpu-proof.json','scheduler_progress206_cpu.py','733dd4ccfbeab9eea106a476b2e4768be6bb4f0098c1875216716ed327b6ce08')]
    for proofname,scriptname,pin in fixture_pins:
        proof=ROOT/'results'/proofname
        assert sha(proof)==pin
        value=json.loads(proof.read_text())
        script=ROOT/'scripts'/scriptname
        assert sha(script)==value['script_sha256']
        paths.extend([proof,script])
    prototype=ROOT/'scripts/candidate_cache_guard206.py'
    value=json.loads((ROOT/'results/scheduler-progress206-cpu-proof.json').read_text())
    assert sha(prototype)==value['candidate_sha256']
    return paths+[layout,candidate,prototype]

def extend(profile):
    validation=audit(profile)
    rows=[json.loads(x) for x in Path(profile['evidence']['path']).read_text().splitlines()]
    spec=next(x['qualification_spec'] for x in rows if x['kind']=='case_start')
    assert 'guard209_prerequisite' in spec,'No wholemodel guard209 qualification in selected evidence'
    requirement=spec['guard209_prerequisite']
    assert requirement['candidate_sha256']==NEW and requirement['baseline_source_sha256']==BASE
    assert requirement['proof_sha256']==PROOF209
    relative='vllm/v1/core/sched/scheduler.py'
    source=Path(profile['cwd'])/relative
    assert sha(source)==NEW
    entries={x['path']:x for x in profile['file_manifest']}
    assert entries[str(source)]['sha256']==NEW
    paths=cpu_dependencies()
    for field,pinfield in [('proof','proof_sha256'),('script','script_sha256'),('candidate','candidate_sha256'),('hooks','hooks_sha256')]:
        path=Path(requirement[field])
        assert sha(path)==requirement[pinfield]
        paths.append(path)
    hooks=json.loads(Path(requirement['hooks']).read_text())
    assert hooks['passed'] and hooks['candidate_sha256']==NEW
    assert hooks['attempts'] and hooks['attempts'][-1]['returncode']==0
    for attempt in hooks['attempts']:
        assert sha(attempt['log'])==attempt['sha256']
        paths.append(Path(attempt['log']))
    parents=requirement['frozen_parents']
    assert parents
    for parent in parents:
        path=Path(parent['profile'])
        assert sha(path)==parent['sha256']
        p=json.loads(path.read_text())
        assert p['status']=='qualified' and p['max_context_per_request']==240128
        assert p['minimum_full_contexts']>=16 and p['image_count']==2
        assert sha(p['evidence']['path'])==p['evidence']['sha256']
        paths.extend([path,Path(p['evidence']['path'])])
    locality=profile['command_locality201']
    assert sha(locality['path'])==locality['sha256']
    snapshot=json.loads(Path(locality['path']).read_text())
    assert snapshot['case']==profile['case']
    result=checked_locality(profile,snapshot['locality'] if locality['enabled'] else None)
    assert result['enabled']==locality['enabled']
    driver=ROOT/'scripts/pipeline_cache210.py'
    assert sha(driver)=='6854731691bc1247e6e1eb228ae23457f5faae17ff7a4a3e7d200b6ad8e8a068'
    paths += [driver,Path(__file__).resolve()]
    output=copy.deepcopy(profile)
    for path in paths:entries[str(path)]=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
    output['file_manifest']=sorted(entries.values(),key=lambda x:x['path'])
    output['guard_validation213']=dict(cpu209_proof_sha256=PROOF209,cpu211_proof_sha256=PROOF211,
        source_sha256=NEW,actual_command_locality=result,whole_model_validation=validation,
        scope='Wholemodel/profile/source/input/multimodal/capacity gates plus immutable schedulerCPU/hook/source provenance. CPU timing is notmodelTPS.')
    audit(output)
    return output

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    assert not args.output.exists()
    output=extend(json.loads(args.profile.read_text()))
    args.output.write_text(json.dumps(output,indent=2))
    print(json.dumps(dict(status='wholequalified_guard_profile213_provenance_bound',output=str(args.output),sha256=sha(args.output),case=output['case'])),flush=True)

if __name__=='__main__':main()
