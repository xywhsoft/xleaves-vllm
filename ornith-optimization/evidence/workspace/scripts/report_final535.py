"""CPU-only final report from closed qualified streams and actual public deployment."""
import html,json,statistics,time
from pathlib import Path
from compare_exact_layer229 import audit_row
from deploy_verified534 import final_closed_runs534 as restored_runs,validate,manager
from qualified_profile189 import sha
R=Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
D=('aggregate_common_decode_tps','average_common_decode_request_tps','aggregate_e2e_output_tps','average_e2e_request_tps','median_ttft_s','wall_s')
K=('cohort','diverse_cohort','independent_resident_cohort','mixed_long_short_cohort','cold_concurrent_long_cohort')
def group(rows,keys):
 out=[]
 for sig in sorted({tuple(x[k] for k in keys) for x in rows}):
  xs=[x for x in rows if tuple(x[k] for k in keys)==sig]
  g=dict(zip(keys,sig));g['repeats']=len(xs);g['metrics']={}
  for f in D:
   vs=[x[f] for x in xs];g['metrics'][f]=dict(mean=statistics.mean(vs) if all(v is not None for v in vs) else None,samples=vs)
  out.append(g)
 return out

def main():
 paths=[R/'reports/ornith-final535.json',R/'reports/ornith-final535.html',R/'reports/ornith-final535-operations.txt']
 assert all(not p.exists() for p in paths)
 restored=restored_runs()
 deployment_path=R/'results/public-deployment534.json';deployment=json.loads(deployment_path.read_text())
 assert deployment['status']=='verified_public_deployment534' and deployment['smoke']['passed'] and deployment['api_key_preserved']
 current=manager.discover();assert current and not current[4] and current[0]==deployment['pid']
 profiles=[];evidence=[]
 for i in (0,1):
  p=R/'results'/f'qualified-adaptive416-{i}-final.json';v=json.loads(p.read_text());validation=validate(v)
  raw=Path(v['evidence']['path']);assert sha(raw)==v['evidence']['sha256']
  rows=[json.loads(x) for x in raw.read_text().splitlines()];cohorts=[x for x in rows if x['kind'] in K]
  for row in cohorts:audit_row(row)
  profiles.append(dict(index=i,case=v['case'],profile=str(p),profile_sha256=sha(p),maximum_active_sequences=v['maximum_active_sequences'],verified_full_contexts=v['minimum_full_contexts'],uncached_full_context_concurrency=16,context_per_request=v['max_context_per_request'],validation=validation,measurements=group(cohorts,('kind','concurrency'))))
  evidence.append(dict(path=str(raw),sha256=sha(raw)))
 comparison_path=R/'results/exact-layer416-qualified-comparison.json';comparison=json.loads(comparison_path.read_text())
 assert comparison['status']=='exact_layer229_completed_paired_comparison'
 for paired in comparison['comparisons']:
  assert all(x['exact_input_token_arrays_matched'] and x['repeats']==2 for x in paired['groups'])
 sampling_path=R/'results/stoch490-short-sampling-proof.json';sampling=json.loads(sampling_path.read_text())
 assert sampling['screen_completed'] and sampling['failure'] is None and sampling['sampling_cohort_count']==48
 sr=[json.loads(x) for x in Path(sampling['sampling_journal']).read_text().splitlines()];sr=[x for x in sr if x['kind']=='cohort']
 for row in sr:audit_row(row)
 assert len(sr)==48 and all(x['cached_prompt_tokens']==[0]*x['concurrency'] for x in sr)
 pp_path=R/'results/pp491-short-screen-proof.json';pp=json.loads(pp_path.read_text())
 converged_path=R/'results/qualified-high198-convergence.json';convergence=json.loads(converged_path.read_text())
 assert sha(converged_path)=='fdc523048ce7716ca0eb00e2e30dc0abacc2e47b27b4f65ba3f7afd27a0b9e50'
 assert convergence['direct_current_gqa_comparisons']['count']==19 and not convergence['direct_current_gqa_comparisons']['any_faster_alternative']
 thermal_path=R/'results/final536-software-convergence-and-thermal.json';thermal=json.loads(thermal_path.read_text());assert not thermal['hardware_changes'] and not thermal['theoretical_maximum_proven']
 for p in (thermal_path,deployment_path,comparison_path,sampling_path,pp_path,converged_path,R/'results/sampler511-gpu-operator-proof.json',R/'results/noise530-short-sampling-proof.json',R/'results/noise529-root-cause538.json'):evidence.append(dict(path=str(p),sha256=sha(p)))
 result=dict(status='final535_closed_streams_exact_baseline_and_deployed_public_report',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),hardware='2xV100 PCIe32GB, noNVLink; CPU R730 dualE5-2696v4',hardware_changes=False,profiles=profiles,paired_baseline=comparison['comparisons'],sampling=group(sr,('case','input_kind','concurrency','sampling_mode')),sampling_scope='512 uncached input/1024 output, same input across3modes; actual modeorder recorded, notcounterbalanced withinoneprofile; private loopback DEV_MODE1 forcache reset. No stochastic239K benchmark.',helper176_short_screen=json.loads((R/'results/noise530-short-sampling-proof.json').read_text()),paired_sampler511=json.loads((R/'results/sampler511-gpu-operator-proof.json').read_text()),pp_short_screen=pp['measurements'],pp_full_context_qualified=False,selected_case=deployment['case'],public_port=8096,public_pid=deployment['pid'],public_smoke=deployment['smoke'],api_key_preserved=True,thermal_observations=thermal['telemetry'],source_convergence=convergence['direct_current_gqa_comparisons'],convergence_scope='19 fixed-GQA operator alternatives are slower. PP/source/boundary conclusions cover measured programs and modes only. No universal theoretical maximum proven.',limitations=['160active meansshort/mixed;16 independent full240128 contexts qualified.26profile verifies26resident fullcontexts, butuncached simultaneous239K tested16only.','Greedy239K rows reuse236944cached tokens. Cold16E2E includesfull inputprocessing andqueueing; decode overlap isundefined.','Warm repeats2, coldrepeat1. Temperatures/clocks uncontrolled. Changes ofa fewpercent arenotstatistical proof.','Stochastic modes shareexactinput percohort, butmodeorder isnotcounterbalanced withinoneprofile: eachdistinctfingerprintfirstusesgreedy/full/filtered; secondprofilemayreverse repeatedfingerprints. Samplingdifferences alsoincludegeneratedcontent/MoErouting.','Numerical/functional/needle/MM gatespassed; noheldoutperplexity evaluation. Differentquantization backends mayproduce differenttokens.','Core engine/attention binaries inherited; nofullcleanbuild qualification. Source/native/runtime hashesincluded.','NoHBMhardware counter measurement; roofline assumptions do not establish attainableTPS orprove theoretical saturation.'],evidence=evidence,terminal_restorations=restored)
 paths[0].write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 def n(v):return '—' if v is None else f'{v:.2f}'
 def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in rows)+'</table></div>'
 labels=dict(cohort='512计数文本',diverse_cohort='512自然文本',independent_resident_cohort='239K缓存命中',mixed_long_short_cohort='16长＋其余短',cold_concurrent_long_cohort='16路239K无缓存')
 rows=[]
 for p in profiles:
  for g in p['measurements']:
   rows.append([f"{p['maximum_active_sequences']}活跃/{p['verified_full_contexts']}完整长",labels[g['kind']],g['concurrency'],g['repeats']]+[n(g['metrics'][f]['mean']) for f in D])
 body=table(['配置','输入场景','并发','轮数','聚合decodeTPS','平均每路decodeTPS','聚合全过程输出TPS','平均每路全过程TPS','TTFT秒','整组秒'],rows)
 pairs=[]
 for p in comparison['comparisons']:
  for g in p['groups']:
   d=g['measurements']['aggregate_common_decode_tps'];e=g['measurements']['aggregate_e2e_output_tps']
   pairs.append(['160活跃' if '_c160_' in p['case'] else '26完整长',labels[g['kind']],g['concurrency'],n(d['layer']['mean']),n(d['optimized']['mean']),f"{d['ratio']:.2f}×",n(e['layer']['mean']),n(e['optimized']['mean'])])
 body+='<h2>相同输入的原始层切对比</h2>'+table(['配置','场景','并发','层切decodeTPS','优化decodeTPS','比例','层切全过程TPS','优化全过程TPS'],pairs)
 sampled=[]
 for g in result['sampling']:
  sampled.append(['160活跃' if '_c160_' in g['case'] else '26完整长',g['input_kind'],g['concurrency'],g['sampling_mode'],g['repeats']]+[n(g['metrics'][f]['mean']) for f in D[:4]])
 body+='<h2>随机采样实测</h2><p>每轮相同512-token输入、1024-token输出、输入缓存命中为0。greedy=temperature0；random_full=temperature0.7、Top-K关闭、Top-P1；random_filtered=temperature0.7、Top-K50、Top-P0.95。输出内容和专家路由随采样变化，因此差值不是纯采样器耗时。</p>'+table(['配置','文本','并发','采样方式','轮数','聚合decodeTPS','平均每路decodeTPS','聚合全过程TPS','平均每路全过程TPS'],sampled)
 candidate=result['helper176_short_screen'];candidate_rows=[]
 for mode in ('greedy','random_full','random_filtered'):
  xs=[x for x in candidate['measurements'] if x['sampling_mode']==mode and x['input_kind']=='natural']
  baseline=[x for x in sr if x['sampling_mode']==mode and x['input_kind']=='natural' and '_c160_' in x['case']]
  if xs:
   d=statistics.mean(x['aggregate_common_decode_tps'] for x in xs);e=statistics.mean(x['aggregate_e2e_output_tps'] for x in xs)
   bd=statistics.mean(x['aggregate_common_decode_tps'] for x in baseline);be=statistics.mean(x['aggregate_e2e_output_tps'] for x in baseline)
   candidate_rows.append([mode,n(bd),n(d),f'{(d/bd-1)*100:.2f}%',n(be),n(e),f'{(e/be-1)*100:.2f}%',n(d/176),n(e/176)])
 body+='<h2>176路显存复用候选</h2><p>双卡配对算子检查通过；降低KV预算到16.5GiB后，24组短输入测试全部成功。自然文本decode和全过程吞吐在三种采样下没有同时达到预设至少3%的晋级标准。候选源码已撤回，未进行完整240K验证；正式配置保留已完整验证的160路。以下为每种采样的两轮自然文本均值。</p>'+table(['采样','160路decode','176路decode','变化','160路全过程','176路全过程','变化','176平均每路decode','176平均每路全过程'],candidate_rows)
 body+='<h2>范围与限制</h2><ul>'+''.join('<li>'+html.escape(x)+'</li>' for x in result['limitations'])+'</ul><p>已验证的软件优化在已测设计中出现收益收敛；没有证明达到通用理论上限。</p>'
 paths[1].write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>双V100 Ornith软件优化最终实测</title><style>body{font:16px/1.6 system-ui;max-width:1600px;margin:32px auto;padding:0 16px}table{border-collapse:collapse;width:100%;margin:18px 0}th,td{border:1px solid #ddd;padding:7px;text-align:right}td:first-child,td:nth-child(2){text-align:left}.scroll{overflow:auto}</style><h1>双V100 Ornith软件优化：最终实测</h1><p>正式8096服务已部署通过完整验证的配置，保留原APIKey，双768×768图片和算术公开接口检查通过。每请求总上下文240128。所有吞吐由实际SSE的token计数和计时重新核算。</p><p>decode为所有请求共同生成窗口的聚合速度；平均每路=聚合/并发。全过程输出TPS=输出token总量/整组墙钟耗时，包含输入处理、等待和生成。未提供原始层切冷16路加速比例，因为没有匹配基线。</p>'+body+'<p><a href="ornith-final535.json">逐轮数据和证据</a> · <a href="ornith-final535-operations.txt">运行、切换与回滚</a></p></html>',encoding='utf-8')
 paths[2].write_text('''Qualified runtime and operation notes\nWorkspace: /home/xs/codex-work/ornith-opt-20261006/onecat-optimization\nUse src/onecat-padded-kv140/.venv/bin/python; CUDA12.8/Python3.12/Torch2.10cu128.\nPublic endpoint: http://222.186.10.53:8096/v1 ; model: ornith-1.5-35b\nAPI credential remains runtime-only and is not included in this report/package.\nQualified profiles: results/qualified-adaptive416-0-final.json (160active,16full240128) and -1-final.json (26active,26full240128).\nStop/switch only when service is idle; manager enforces current request/GPU-process guards.\nRollback to saved prior server: src/onecat-padded-kv140/.venv/bin/python scripts/manage_onecat71.py --rollback\nSwitching directly with manager overwrites its one-step rollback snapshot; keep prior rollback file if needed.\nExact launch command and environment are in each qualified JSON profile, credentials supplied at runtime.\nNoautomatic systemd/reboot persistence was configured by this optimization task.\nSampler helper400 passed paired511 GPU numerical/pressure/latency gates. Reversible176 shortscreen530 did not pass all-mode promotion, and restoredBASE. See actual529 proof for outcome.\nPP445 failed graph-workspace OOM; PP491 shrinksKV from16.5GiB to14GiB forworkspace while retainingstartup16minimum. PP491shortscreen alone is not240Kqualification.\nQuality includesbitwise operator contracts, text/natural/needle/MM/capacity gates, notheldoutperplexity.\nThe source archive includespatches, changed/untracked sources, native libraries, runtimefreeze, rawstreams and hashes. It doesnotinclude modelweights.\nOriginal engine/attention binaries reused; fullclean rebuild notqualified. Rerun operator/model gates afterdependency/source changes.\n''',encoding='utf-8')
 print(json.dumps(dict(status=result['status'],files=[dict(path=str(p),sha256=sha(p)) for p in paths]),ensure_ascii=False))
if __name__=='__main__':main()
