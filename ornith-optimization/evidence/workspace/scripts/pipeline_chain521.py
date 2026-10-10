"""CPU waiter: register paired sampler511 only after actual PP491 closure."""
import ast,json,os,subprocess,sys,time
from pathlib import Path
from final_jobs286 import check_job,terminal_jobs
from pipeline_layer216 import process
from qualified_profile189 import sha
R=Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
STATUS=R/'results/chain521-status.json'
def record(status,**fields):
 v=dict(status=status,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),gpu_work_in_controller=False,serving_source_mutation=False,**fields)
 STATUS.write_text(json.dumps(v,indent=2));print(json.dumps(v),flush=True)
def wait(name):
 j=json.loads((R/'cache'/f'{name}-job.json').read_text());assert sha(j['script'])==j['script_sha256']
 deadline=time.monotonic()+7200
 while True:
  state,birth=process(j['pid'])
  if state in ('missing','Z','X') or birth!=str(j['birth']):return check_job(name)
  assert time.monotonic()<deadline;time.sleep(10)
def main():
 record('chain521_waiting_actual_PP491_no_sampler_future_GPU_registration')
 wait('pp491');check_job('stoch490');terminal_jobs()
 for name in ('stoch490-short-sampling-proof.json','pp491-short-screen-proof.json'):
  v=json.loads((R/'results'/name).read_text())
  assert v['restored']['verification']==dict(passed=True,slots=16,context=240128)
  assert v['current_public']['original_command_reused'] and v['current_public']['api_key_preserved']
 from pipeline_sampler511 import inputs
 inputs()
 script=R/'scripts/pipeline_sampler511.py';pin='05a74d901f66f52ae02b46f44bd6c758bc097393514af3d7e13dc4deb6ae1659'
 assert sha(script)==pin;ast.parse(script.read_text())
 d=R/'cache/sampler511-job.json';log=R/'logs/sampler511-pipeline.log';assert not d.exists() and not log.exists()
 with log.open('x') as f:
  c=subprocess.Popen([str(R/'src/onecat-padded-kv140/.venv/bin/python'),str(script)],cwd=R/'scripts',env=dict(os.environ),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 state,birth=process(c.pid);assert birth is not None
 j=dict(pid=c.pid,birth=birth,script=str(script),script_sha256=pin,resource='gpu_exclusive',log=str(log),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),serving_source_mutation=False,admission='ActualPP491 terminal/restored, beforebalanced511; helper400 inactive. No wholemodel qualificationclaim.')
 t=d.with_suffix('.json.tmp');t.write_text(json.dumps(j,indent=2));t.replace(d)
 record('chain521_actual_paired_sampler511_registered',pid=c.pid,descriptor_sha256=sha(d))
 wait('sampler511');terminal_jobs()
 p=R/'results/sampler511-gpu-operator-proof.json';v=json.loads(p.read_text())
 assert v['restored']['verification']==dict(passed=True,slots=16,context=240128)
 assert v['restored']['original_command_reused'] and v['restored']['api_key_preserved']
 record('chain521_completed_actual_balanced_result_requires_wholemodel_decision',proof=str(p),sha256=sha(p),operator_status=v['status'],failure=v['failure'],helper400_installed=False)
 assert 'torch' not in sys.modules
if __name__=='__main__':
 try:main()
 except BaseException as e:
  record('chain521_stopped_requires_actual_result_review',error_type=type(e).__name__,error=str(e));raise
