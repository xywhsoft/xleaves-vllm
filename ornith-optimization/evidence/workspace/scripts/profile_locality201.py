"""Audit NUMA by measured command flags, independently of case-name suffixes."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from qualified_profile189 import audit,sha

def checked_locality(profile,observation):
    command=profile['command']
    enabled='--numa-bind' in command
    if not enabled:
        assert '--numa-bind-nodes' not in command
        return dict(enabled=False,scope='Measured command has noNUMA binding; no workerlocality claim.')
    assert '--numa-bind-nodes' in command
    index=command.index('--numa-bind-nodes')
    assert command[index+1:index+3]==['0','1']
    assert profile['env'].get('VLLM_WORKER_MULTIPROC_METHOD')=='spawn'
    assert observation is not None and observation.get('passed') is True
    workers=observation['workers']
    assert len(workers)==2 and {worker['rank'] for worker in workers}=={0,1}
    for worker in workers:
        expected=Path(f'/sys/devices/system/node/node{worker["rank"]}/cpulist').read_text().strip()
        assert worker['passed'] is True and worker['cpus']==worker['expected_cpus']==expected
        assert worker['worker_method']=='spawn'
        assert worker['bound_anon_pages']>=4096 and worker['remote_pages_in_bound_anon']==0
        assert f'bind:{worker["rank"]}' in worker['anonymous_policies']
        assert worker['pid']>0 and str(worker['birth']).isdigit()
    return dict(enabled=True,workers=workers,passed=True,
        scope='Actual modelcase worker observations; memorycpuset permissionmask is not NUMA policy. Selected mapping mustbe bind:rank andranklocal.')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--profile',required=True,type=Path)
    parser.add_argument('--locality',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    profile=json.loads(args.profile.read_text())
    validation=audit(profile)
    observation=None
    source=None
    if args.locality:
        raw=args.locality.read_bytes()
        observation=json.loads(raw)[profile['case']]
        source=dict(path=str(args.locality),sha256_at_freeze=hashlib.sha256(raw).hexdigest())
    checked=checked_locality(profile,observation)
    result=dict(case=profile['case'],profile=str(args.profile),profile_sha256=sha(args.profile),
        validation=validation,locality=checked,source_observation=source,model_requests_sent=0)
    if args.output:
        snapshot=args.output.with_name(args.output.stem+'-locality201.json')
        assert not args.output.exists() and not snapshot.exists()
        snapshot.write_text(json.dumps(result,indent=2))
        frozen=copy.deepcopy(profile)
        paths=[snapshot,Path(__file__).resolve()]
        entries={entry['path']:entry for entry in frozen['file_manifest']}
        for path in paths:
            entries[str(path)]=dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
        frozen['file_manifest']=sorted(entries.values(),key=lambda entry:entry['path'])
        frozen['command_locality201']=dict(path=str(snapshot),sha256=sha(snapshot),enabled=checked['enabled'])
        audit(frozen)
        args.output.write_text(json.dumps(frozen,indent=2))
        result.update(output=str(args.output),output_sha256=sha(args.output),locality_snapshot=str(snapshot),locality_snapshot_sha256=sha(snapshot))
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
