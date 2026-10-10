"""Serial final delivery after actual530 review; deployment wrapper holds GPUlock."""
import json,os,subprocess,sys,time
from pathlib import Path
from deploy_verified534 import ROOT,final_closed_runs534,terminal_jobs,validate,manager
from pipeline_layer216 import process
from qualified_profile189 import sha
PINS={'deploy_verified534.py':'0f671eab9ecbf50ae2142c28d4e812e2fd6f86d4f1da70e026075342c9d3f190','report_final535.py':'93d88371ae892f58d985dedcbe7c5d642abf00c2bf7ca84ebd839e7e6944dd4a','freeze_convergence536.py':'58a4efb139c03205c3af00170a0826ff2f6af0a81e1ed82b88e4e747ed8df283','package_final537.py':'dd096df1fefbfefb398f670d31a83783d4b784db8bbd35b8e42d3bf4036a7b27'}
def main():
 d=ROOT/'cache/delivery540-job.json';deadline=time.monotonic()+15
 while not d.exists():assert time.monotonic()<deadline;time.sleep(.05)
 own=json.loads(d.read_text());state,birth=process(os.getpid());assert own['pid']==os.getpid() and str(own['birth'])==birth and own['script_sha256']==sha(__file__)
 for f,pin in PINS.items():assert sha(ROOT/'scripts'/f)==pin
 terminal_jobs();runs=final_closed_runs534()
 profiles=[ROOT/'results'/f'qualified-adaptive416-{i}-final.json' for i in (0,1)]
 for p in profiles:validate(json.loads(p.read_text()))
 selected=profiles[0]
 deployment=ROOT/'results/public-deployment534.json'
 status=ROOT/'results/delivery540-status.json'
 def record(stage,**more):
  v=dict(status=stage,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),script_sha256=sha(__file__),job_descriptor_sha256=sha(d),selected_profile=str(selected),selected_profile_sha256=sha(selected),**more)
  temp=status.with_suffix('.tmp');temp.write_text(json.dumps(v,indent=2));temp.replace(status);print(json.dumps(v),flush=True)
 def run(script,args=(),timeout=3600):
  assert sha(ROOT/'scripts'/script)==PINS[script]
  subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],cwd=ROOT/'scripts',check=True,timeout=timeout)
 record('delivery540_admitted_all_parent_jobs_terminal_public_deployment_starting')
 try:
  run('deploy_verified534.py',['--profile',selected,'--output',deployment])
  record('delivery540_public_deployment_passed_reports_in_progress',deployment_sha256=sha(deployment))
  run('freeze_convergence536.py',timeout=120)
  run('report_final535.py',timeout=120)
  record('delivery540_reports_closed_packaging_in_progress',deployment_sha256=sha(deployment))
  extras=[ROOT/'scripts/pipeline_delivery540.py',ROOT/'cache/delivery540-job.json',ROOT/'reports/ornith-final535.json',ROOT/'reports/ornith-final535.html',ROOT/'reports/ornith-final535-operations.txt',ROOT/'results/final536-software-convergence-and-thermal.json',ROOT/'results/frozen-thermal536.jsonl']
  args=['--profile',selected,'--deployment',deployment,'--comparison',ROOT/'results/exact-layer416-qualified-comparison.json','--directory',ROOT/'reports/onecat-final537-source-evidence','--archive',ROOT/'reports/onecat-final537-source-evidence.tar.gz','--proof',ROOT/'results/final537-package.json']
  for p in extras:args+=['--extra',p]
  run('package_final537.py',args,timeout=1800)
  current=manager.discover();proof=json.loads(deployment.read_text());assert current and not current[4] and current[0]==proof['pid']
  package=json.loads((ROOT/'results/final537-package.json').read_text());assert sha(package['archive'])==package['sha256'] and package['public_api_credential_absence_verified']
  record('delivery540_completed_deployed_report_and_archive_verified',public_pid=current[0],deployment_sha256=sha(deployment),package_proof_sha256=sha(ROOT/'results/final537-package.json'),archive=package['archive'],archive_sha256=package['sha256'],archive_bytes=package['bytes'])
 except BaseException as e:
  record('delivery540_stopped_actual_failure_requires_review',error_type=type(e).__name__,error=str(e),deployment_exists=deployment.exists())
  raise
if __name__=='__main__':main()
