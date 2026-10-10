"""CPU sequencing only: actual stochastic465 terminal before registering PP481."""
import ast,json,os,subprocess,sys,time
from pathlib import Path
from final_jobs286 import check_job,terminal_jobs
from pipeline_layer216 import process
from qualified_profile189 import sha
R=Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
STATUS=R/'results/chain482-status.json'
def record(status,**fields):
 v=dict(status=status,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),gpu_work=False,source_mutation=False,**fields)
 STATUS.write_text(json.dumps(v,indent=2));print(json.dumps(v),flush=True)
def wait(name):
 j=json.loads((R/'cache'/f'{name}-job.json').read_text());assert sha(j['script'])==j['script_sha256']
 deadline=time.monotonic()+7200
 while True:
  state,birth=process(j['pid'])
  if state in ('missing','Z','X') or birth!=str(j['birth']):return check_job(name)
  assert time.monotonic()<deadline;time.sleep(10)
def main():
 record('chain482_waiting_actual_stoch465_no_PP_future_GPU_registration')
 wait('stoch465');terminal_jobs()
 v=json.loads((R/'results/stoch465-short-sampling-proof.json').read_text())
 assert v['screen_completed'] and v['failure'] is None and v['sampling_cohort_count']==48
 assert v['restored']['verification']==dict(passed=True,slots=16,context=240128)
 from pipeline_pp481 import prerequisites
 prerequisites()
 script=R/'scripts/pipeline_pp481.py';pin='b962f42f037e1203e14e80395e3439cb783f198379c1d4cef4896f4a5d3b6b9f'
 assert sha(script)==pin;ast.parse(script.read_text())
 d=R/'cache/pp481-job.json';log=R/'logs/pp481-pipeline.log';assert not d.exists() and not log.exists()
 with log.open('x') as f:
  c=subprocess.Popen([str(R/'src/onecat-padded-kv140/.venv/bin/python'),str(script)],cwd=R/'scripts',env=dict(os.environ),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 state,birth=process(c.pid);assert birth is not None
 j=dict(pid=c.pid,birth=birth,script=str(script),script_sha256=pin,resource='gpu_exclusive',log=str(log),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_mutation_performed=False)
 t=d.with_suffix('.json.tmp');t.write_text(json.dumps(j,indent=2));t.replace(d)
 record('chain482_registered_PP481_after_actual_stoch465_pass',pid=c.pid,descriptor_sha256=sha(d))
 wait('pp481');terminal_jobs()
 p=R/'results/pp481-short-screen-proof.json';v=json.loads(p.read_text())
 assert v['restored']['verification']==dict(passed=True,slots=16,context=240128)
 assert v['current_public']['original_command_reused'] and v['current_public']['api_key_preserved']
 record('chain482_completed_actual_PP481_result_requires_selection',proof=str(p),sha256=sha(p),screen_completed=v['screen_completed'],failure=v['failure'])
 assert 'torch' not in sys.modules
if __name__=='__main__':
 try:main()
 except BaseException as e:
  record('chain482_stopped_requires_actual_result_review',error_type=type(e).__name__,error=str(e));raise
