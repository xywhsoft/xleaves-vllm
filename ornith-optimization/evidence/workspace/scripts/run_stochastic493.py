"""Reuse139 lifecycle; run balanced greedy/full/filtered sampling cohorts."""
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import urllib.request

ROOT=Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
PARENT=ROOT/'scripts/onecat_qualified_runner139.py'
COHORT=ROOT/'scripts/cohort_stochastic454.py'


def main():
    assert hashlib.sha256(PARENT.read_bytes()).hexdigest()=='01f36917804c07758408f7ffd0c789aebe47c612ea2c4491e7cec19baecd5fcb'
    assert hashlib.sha256(COHORT.read_bytes()).hexdigest()=='51327f0bc7372d5a752a5c48d4a2a3e1a908520a5df6d4fd268005235d8af625'
    plan=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8-sig'))
    path=ROOT/'results'/f"{plan['run']}-sampling.jsonl";assert not path.exists()
    module_spec=importlib.util.spec_from_file_location('qualified_runner139_sampling',PARENT)
    runner=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(runner)
    cohort_spec=importlib.util.spec_from_file_location('cohort454',COHORT)
    sampling_module=importlib.util.module_from_spec(cohort_spec);cohort_spec.loader.exec_module(sampling_module)
    from compare_exact_layer229 import audit_row
    original=runner.cohort;original_emit=runner.emit;calls={};active_case={}
    natural_hashes=set()
    for case in plan['cases']:
        if case.get('extra_diverse_cohorts'):
            diverse=json.loads(Path(case['extra_diverse_cohorts']).read_text())
            for repeat in range(2):
                arrays=[x['prompt'] for x in diverse[f'512-{repeat}']][:case['diverse_count']]
                natural_hashes.add(hashlib.sha256(json.dumps(arrays,separators=(',',':')).encode()).hexdigest())
    def tracked_emit(out,row):
        if row.get('kind')=='case_start':active_case['name']=row['case']
        return original_emit(out,row)
    runner.emit=tracked_emit
    modes=[('greedy',dict(temperature=0,top_k=-1,top_p=1.0)),('random_full',dict(temperature=.7,top_k=-1,top_p=1.0)),('random_filtered',dict(temperature=.7,top_k=50,top_p=.95))]
    with path.open('x') as stream:
        def measured(base,key,prompts,output):
            if output!=1024:return original(base,key,prompts,output)
            fingerprint=hashlib.sha256(json.dumps(prompts,separators=(',',':')).encode()).hexdigest()
            signature=(len(prompts),len(prompts[0]),fingerprint)
            occurrence=calls.get(signature,0);calls[signature]=occurrence+1
            order=modes if occurrence%2==0 else list(reversed(modes))
            rows=[];filtered=None
            for label,params in order:
                short=all(len(p)==512 for p in prompts)
                if short:
                    request=urllib.request.Request(base+'/reset_prefix_cache',data=b'',headers={'Authorization':'Bearer '+key},method='POST')
                    with urllib.request.urlopen(request,timeout=60) as response:assert response.status==200
                row=sampling_module.cohort(base,key,prompts,output,params)
                audit_row(row)
                cached=[]
                for item in row['requests']:
                    details=item['usage'].get('prompt_tokens_details')
                    assert details is None or (isinstance(details,dict) and 'cached_tokens' in details)
                    cached.append((details or {}).get('cached_tokens',0))
                assert not short or cached==[0]*len(prompts),'Short inputs must be uncached in every sampling mode'
                if not short:assert cached==[236944]*len(prompts),'Long measurements must preserve the same cached prefix in every mode'
                row.update(case=active_case['name'],input_kind='natural' if fingerprint in natural_hashes else 'counting',sampling_mode=label,occurrence_for_exact_input=occurrence,mode_order=[x[0] for x in order],
                           cached_prompt_tokens=cached,measurement_scope='Actual1024-output SSE; sampling changes only temperature/topk/topp. Common decode window and E2E output TPS kept separate.')
                original_emit(path,row);rows.append(row)
                if label=='random_filtered':filtered=row
            assert filtered is not None
            assert len({tuple(x['cached_prompt_tokens']) for x in rows})==1
            assert len({x['input_token_arrays_sha256'] for x in rows})==1
            return filtered
        runner.cohort=measured
        try:
            runner.main()
        finally:
            runner.cohort=original
            runner.emit=original_emit
            original_emit(path,dict(kind='sampling_wrapper_terminal',run=plan['run'],module_sha256=hashlib.sha256(COHORT.read_bytes()).hexdigest(),
                wrapper_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_mutation=False))


if __name__=='__main__':main()
