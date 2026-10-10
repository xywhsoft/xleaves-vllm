"""Final deployment wrapper with source, command-locality and GPU-lock gates."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

import manage_onecat71 as manager
from qualified_profile189 import audit,sha
from profile_locality201 import checked_locality
from pipeline_numa189 import locality,process
from guard_profile213 import extend as guarded_profile

ROOT=Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')

def validate(profile):
    checked=audit(profile)
    binding=profile['command_locality201']
    assert sha(binding['path'])==binding['sha256']
    snapshot=json.loads(Path(binding['path']).read_text())
    assert snapshot['case']==profile['case']
    locality_check=checked_locality(profile,snapshot['locality'] if binding['enabled'] else None)
    assert locality_check['enabled']==binding['enabled']
    evidence=[json.loads(x) for x in Path(profile['evidence']['path']).read_text().splitlines()]
    spec=next(x['qualification_spec'] for x in evidence if x['kind']=='case_start')
    if 'guard209_prerequisite' in spec:
        assert 'guard_validation213' in profile,'Freeze guard provenance with guard_profile213 before deployment'
        extended=guarded_profile(profile)
        manifest={x['path']:x for x in profile['file_manifest']}
        assert all(manifest.get(x['path'])==x for x in extended['file_manifest'])
    else:
        assert sha(Path(profile['cwd'])/'vllm/v1/core/sched/scheduler.py')=='441af671c70dd0764a0012630e05227426baf5b92ffece33a3c3b0108a2cc8cf'
    return dict(validation=checked,command_locality=locality_check)

def terminal_jobs():
    checked=[]
    for name in ['workspace187-job.json','numa189-job.json','boundary200-job.json','cache210-job.json']:
        path=ROOT/'cache'/name
        assert path.exists(),name
        job=json.loads(path.read_text())
        state,birth=process(job['pid'])
        same=birth==str(job['birth'])
        assert not same or state in ['missing','Z','X'],f'Active optimization job {name}: {job["pid"]}'
        checked.append(dict(job=name,pid=job['pid'],same_birth=same,state=state))
    return checked

def invoke(args):
    previous=sys.argv
    sys.argv=[str(ROOT/'scripts/manage_onecat71.py')]+args
    try:return manager.main()
    finally:sys.argv=previous

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',required=True,type=Path)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    profile=json.loads(args.profile.read_text())
    checked=validate(profile)
    if args.validate_only:
        print(json.dumps(dict(status='deployment214_profile_audited_only',case=profile['case'],**checked)),flush=True)
        return
    assert args.output is not None and not args.output.exists()
    jobs=terminal_jobs()
    lock=(ROOT/'cache/gpu.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        terminal_jobs()
        checked=validate(profile)
        before=manager.discover()
        assert before is not None
        manager.check_idle(before,manager.api_key(before[1],before[2]))
        log=ROOT/'logs/onecat-public71.log'
        offset=log.stat().st_size if log.exists() else 0
        invoke([str(args.profile)])
        try:
            current=manager.discover()
            assert current is not None and not current[4]
            public_result=json.loads((ROOT/'results/onecat-public-deployment71.json').read_text())
            assert public_result['status']=='deployed' and public_result['case']==profile['case']
            assert public_result['pid']==current[0] and public_result['smoke']['passed']
            observed=dict(enabled=False)
            if checked['command_locality']['enabled']:
                snapshot=ROOT/'logs'/f'public214-{current[0]}-workers.log'
                assert not snapshot.exists()
                with log.open('rb') as stream:
                    stream.seek(offset)
                    snapshot.write_bytes(stream.read())
                workers=locality(snapshot)
                assert workers is not None and all(w['passed'] for w in workers)
                assert all(os.getpgid(w['pid'])==current[0] for w in workers)
                observed=checked_locality(profile,dict(passed=True,workers=workers))
            record=dict(status='verified_public_deployment214',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                profile=str(args.profile),profile_sha256=sha(args.profile),case=profile['case'],pid=current[0],
                port=8096,api_key_preserved=public_result['api_key_preserved'],smoke=public_result['smoke'],
                actual_public_locality=observed,terminal_optimization_jobs=jobs,**checked)
            args.output.write_text(json.dumps(record,indent=2))
            print(json.dumps(dict(status=record['status'],output=str(args.output),sha256=sha(args.output),case=profile['case'],pid=current[0])),flush=True)
        except BaseException:
            invoke(['--rollback'])
            raise
    finally:
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()

if __name__=='__main__':main()
