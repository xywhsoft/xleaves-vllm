"""Immutable complete-row telemetry prefix and narrowly stated software convergence."""
import hashlib,json,statistics,time
from datetime import datetime,timezone
from pathlib import Path
from deploy_verified534 import final_closed_runs534 as restored_runs
from final_jobs286 import terminal_jobs
from pipeline_layer216 import process
from qualified_profile189 import sha
R=Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
def main():
 terminal_jobs();restored_runs()
 deployment=R/'results/public-deployment534.json';d=json.loads(deployment.read_text());assert d['status']=='verified_public_deployment534' and d['smoke']['passed']
 descriptor=R/'cache/thermal312-job.json';j=json.loads(descriptor.read_text());assert sha(j['script'])==j['script_sha256']
 source=Path(j['output']);raw=source.read_bytes();end=raw.rfind(b'\n')+1;prefix=raw[:end]
 rows=[json.loads(x) for x in prefix.splitlines()];samples=[x for x in rows if x['kind']=='telemetry312_sample']
 assert samples and all(x['script_sha256']==j['script_sha256'] and not x['hardware_changes'] and not x['cuda_context_created'] for x in samples)
 window_start=datetime.fromisoformat(json.loads((R/'cache/adaptive393-job.json').read_text())['started_utc'].replace('Z','+00:00')).timestamp()
 selected=[x for x in samples if x['unix_time']>=window_start];assert len(selected)>10
 frozen=R/'results/frozen-thermal536.jsonl';out=R/'results/final536-software-convergence-and-thermal.json';assert not frozen.exists() and not out.exists()
 frozen.write_bytes(prefix);assert sha(frozen)==hashlib.sha256(prefix).hexdigest()
 gpus=[]
 for rank in ('0','1'):
  xs=[g for x in selected if 'error' not in x for g in x['gpus'] if g['index']==rank]
  def numeric(field):
   values=[]
   for x in xs:
    try:values.append(float(x[field]))
    except ValueError:pass
   return dict(minimum=min(values),median=statistics.median(values),maximum=max(values)) if values else None
  gpus.append(dict(rank=rank,samples=len(xs),temperature_c=numeric('temperature.gpu'),sm_clock_mhz=numeric('clocks.sm'),thermal_active_samples=sum(x['clocks_event_reasons.sw_thermal_slowdown'].lower()=='active' for x in xs),power_cap_active_samples=sum(x['clocks_event_reasons.sw_power_cap'].lower()=='active' for x in xs)))
 c_path=R/'results/qualified-high198-convergence.json';c=json.loads(c_path.read_text());assert sha(c_path)=='fdc523048ce7716ca0eb00e2e30dc0abacc2e47b27b4f65ba3f7afd27a0b9e50'
 pp_path=R/'results/pp491-short-screen-proof.json';pp=json.loads(pp_path.read_text())
 sample_path=R/'results/stoch490-short-sampling-proof.json';stochastic=json.loads(sample_path.read_text());assert stochastic['screen_completed'] and stochastic['sampling_cohort_count']==48
 operator=R/'results/sampler511-gpu-operator-proof.json';op=json.loads(operator.read_text());assert op['status']=='sampler511_gpu_operator_checks_passed' and op['failure'] is None
 screen_path=R/'results/noise530-short-sampling-proof.json';screen=json.loads(screen_path.read_text());assert not screen['short_promotion_passed'] and not screen['source_retained']
 parent=R/'results/adaptive393-whole-model-proof.json';p=json.loads(parent.read_text());assert len(p['qualified_cases'])==2 and p['failure'] is None
 gaps=[b['unix_time']-a['unix_time'] for a,b in zip(selected,selected[1:])]
 state,birth=process(j['pid'])
 v=dict(status='final536_actual_closed_profiles_and_sampled_thermal_convergence',hardware_changes=False,model_requests_sent=0,source_mutation=False,telemetry=dict(source=str(source),snapshot=str(frozen),snapshot_sha256=sha(frozen),snapshot_bytes=len(prefix),complete_rows=len(rows),partial_suffix_excluded=len(raw)-end,observer_pid=j['pid'],observer_state=state,observer_birth_matches=birth==str(j['birth']),observer_finished=any(x['kind']=='telemetry312_complete' for x in rows),descriptor_sha256=sha(descriptor),window_start_utc=datetime.fromtimestamp(window_start,timezone.utc).isoformat(),first_sample_utc=selected[0]['utc'],last_sample_utc=selected[-1]['utc'],sampled_rows=len(selected),errors=sum('error' in x for x in selected),maximum_observed_sample_gap_s=max(gaps),gpus=gpus,scope='Read-only30second sampled observations acrosscapture/prefill/decode/idle. Prefix isimmutable; observer maycontinue. OldJOBS fields notused ascurrentcase attribution. No causalTPScorrection, noHBMcounter.'),
 gqa_convergence=c['direct_current_gqa_comparisons'],gqa_scope='Same retained GQA nativebinary, C16/C24 and239104tokens:19 measured alternatives allslower. Thisisdesign-limited operator convergence, not universaltheoretical optimality.',memory_boundary=dict(qualified_high_active=160,qualified_high_full_contexts=16,qualified_capacity_full_contexts=26,original176_failed_allocation_scope='Historical393startup OOM; not a TPS measurement.',helper400_paired_operator_passed=True,helper400_reversible_shortscreen=screen,helper400_currently_installed=False),pp_screen=dict(screen_completed=pp['screen_completed'],measurements=pp['measurements'],whole240K_qualified=False),stochastic=dict(closed_cohorts=48,three_modes=True,all_short_cache_zero=True,scope='Actual512input/1024output only; no stochastic239K measurements.'),theoretical_maximum_proven=False,measured_hbm_bandwidth_available=False,selected_case=d['case'],public_deployment_sha256=sha(deployment),evidence=[dict(path=str(x),sha256=sha(x)) for x in (c_path,pp_path,sample_path,operator,parent,screen_path)],limitations=['Cooling unavailable; software only. Observed thermalflags/clocks mayvary across runs.','No fullclean enginebuild orheldoutperplexity qualification.','Operator screen doesnot guarantee no future framework/kernel/speculative decoding improvement. Only measured valid designs can beclaimed converged.'])
 out.write_text(json.dumps(v,indent=2));print(json.dumps(dict(status=v['status'],proof=str(out),sha256=sha(out),snapshot=str(frozen),snapshot_sha256=sha(frozen),gpus=gpus)))
if __name__=='__main__':main()
