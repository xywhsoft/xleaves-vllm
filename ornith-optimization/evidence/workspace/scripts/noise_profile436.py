"""Bind unchanged370 plus actual400 numeric and whole420 stage evidence."""
import argparse
import copy
import json
from pathlib import Path

from qualified_profile189 import ROOT, audit, sha

NEW = '7e1e04a7af6badc8012cfbcae920495672a57d3a30b07d547adc82f8d8177c52'
CPU = 'a00b62d937b538844b4d14850eb82e5564f5827c87fcf2cd9aed1ebaaac2e95d'
GPU = '81cf8f705a628c0bbc7af256597947629cd56932424128523d43c9fb7910c569'


def extend(profile):
    validation = audit(profile)
    entries = [json.loads(x) for x in Path(profile['evidence']['path']).read_text().splitlines()]
    spec = next(x['qualification_spec'] for x in entries if x['kind'] == 'case_start')
    requirement = spec['adaptive370_prerequisite']
    source = Path(profile['cwd']) / requirement['serving_relative_path']
    assert sha(source) == requirement['candidate_sha256'] == NEW
    assert requirement['cpu_sha256'] == CPU and requirement['gpu_sha256'] == GPU
    paths = [Path(__file__).resolve()]
    for name, pin in (('cpu', 'cpu_sha256'), ('gpu', 'gpu_sha256'), ('candidate', 'candidate_sha256'),
                      ('checker', 'checker_sha256'), ('hooks', 'hooks_sha256'), ('driver', 'driver_sha256')):
        path = Path(requirement[name])
        assert sha(path) == requirement[pin]
        paths.append(path)
    cpu = json.loads(Path(requirement['cpu']).read_text())
    assert cpu['candidate_sha256'] == NEW and len(cpu['checks']) == 360
    assert not cpu['cuda_initialized'] and not cpu['serving_source_mutation']
    cpu_script = ROOT / 'scripts/test_sampler371_cpu.py'
    assert sha(cpu_script) == cpu['script_sha256']
    paths.append(cpu_script)
    gpu = json.loads(Path(requirement['gpu']).read_text())
    assert gpu['status'] == 'sampler374_gpu_operator_checks_passed' and gpu['failure'] is None
    assert len(gpu['completed']) == 8 and all(x['ratio'] <= 1.05 for x in gpu['fast_latency_ratios'])
    for entry in gpu['completed']:
        assert sha(entry['output']) == entry['sha256'] and sha(entry['log']) == entry['log_sha256']
        paths.extend([Path(entry['output']), Path(entry['log'])])
    gpu_driver = ROOT / 'scripts/pipeline_sampler374.py'
    assert sha(gpu_driver) == gpu['script_sha256']
    paths.append(gpu_driver)
    hooks = json.loads(Path(requirement['hooks']).read_text())
    assert hooks['passed'] and hooks['candidate_sha256'] == NEW and hooks['returncode'] == 0
    assert sha(hooks['log']) == hooks['log_sha256']
    paths.append(Path(hooks['log']))
    whole_path = ROOT / 'results/adaptive393-whole-model-proof.json'
    whole = json.loads(whole_path.read_text())
    assert whole['status'] == 'adaptive393_whole_model_completed' and whole['failure'] is None
    assert whole['qualified_cases'] and whole['source_retained_for_final_review']
    assert 'noise400_prerequisite' in spec, 'Use adaptive413 for unchanged-helper393 profiles'
    assert whole['final_source_sha256'] == NEW and whole['script_sha256'] == requirement['driver_sha256']
    assert sha(whole['journal']) == whole['journal_sha256']
    assert whole['restored']['verification']['passed'] and whole['restored']['verification']['context'] == 240128
    assert whole['current_public']['original_command_reused'] and whole['current_public']['api_key_preserved']
    job = ROOT / 'cache/adaptive393-job.json'
    assert sha(job) == whole['job_descriptor_sha256'] == requirement['job_descriptor_sha256']
    backup = ROOT / 'scripts/baseline_sampler_adaptive393.py'
    assert sha(backup) == requirement['baseline_sha256']
    paths.extend([whole_path, job, backup])
    for parent in requirement['frozen_parents']:
        path = Path(parent['path'])
        assert sha(path) == parent['sha256']
        paths.append(path)
    output = copy.deepcopy(profile)
    manifest = {x['path']: x for x in output['file_manifest']}
    assert manifest[str(source)]['sha256'] == NEW
    for path in paths:
        manifest[str(path)] = dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))
    output['file_manifest'] = sorted(manifest.values(), key=lambda x: x['path'])
    output['adaptive_validation436'] = dict(cpu_proof_sha256=CPU, gpu_proof_sha256=GPU, source_sha256=NEW,
        whole_model_proof_sha256=sha(whole_path), whole_model_validation=validation,
        scope='Exact adaptive source CPU/GPU bits, RNG/capture, actual OOM/cached replay and normal latency, source hooks, whole-model capacity, independent240K, images, mixed cache and cold16 gates. Operator timing is not model TPS.')
    output = bind_noise(output, spec, paths)
    audit(output)
    return output


def bind_noise(profile, spec, inherited_paths):
    requirement = spec['noise400_prerequisite']
    helper = Path(profile['cwd']) / requirement['serving_relative_path']
    candidate = 'afa1e2cb242649c33c42b58d02709dc3984846d7ea0e89417bd05024148f4322'
    baseline = 'ab2ee9e52cc229ced835e840cb90cea3a1d7ed2d927d86507b670b532a56d52f'
    assert sha(helper) == requirement['candidate_sha256'] == candidate
    assert requirement['baseline_sha256'] == baseline
    paths = list(inherited_paths)
    for name, pin in (('cpu','cpu_sha256'), ('gpu','gpu_sha256'), ('candidate','candidate_sha256'),
                      ('checker','checker_sha256'), ('hooks','hooks_sha256'), ('driver','driver_sha256'), ('frozen416','frozen416_sha256')):
        path = Path(requirement[name]); assert sha(path) == requirement[pin]; paths.append(path)
    cpu = json.loads(Path(requirement['cpu']).read_text())
    assert sha(requirement['cpu']) == 'bccd8ac11e89bda9f7bfc03b3164f4a017a2d09bc584d23d726d655f87fae76b'
    assert len(cpu['checks']) == 138 and len(cpu['metadata_eligibility_fixtures']) == 32
    assert not cpu['cuda_initialized']
    cpu_script = ROOT/'scripts/test_sampler401_cpu.py'
    assert sha(cpu_script) == cpu['script_sha256']; paths.append(cpu_script)
    gpu = json.loads(Path(requirement['gpu']).read_text())
    assert gpu['status'] == 'sampler408_gpu_operator_checks_passed' and gpu['failure'] is None
    assert len(gpu['completed']) == 6 and len(gpu['fast_latency_ratios']) == 6
    assert all(x['ratio'] <= 1.05 for x in gpu['fast_latency_ratios'])
    assert not gpu['serving_source_mutation']
    gpu_driver = ROOT/'scripts/pipeline_sampler408.py'; gpu_job = ROOT/'cache/sampler408-job.json'
    assert sha(gpu_driver) == gpu['script_sha256'] == '3c90ea18fd9bf7266ef7220545f7f5431300e538091ff280098eb4c9276edfad'
    assert sha(gpu_job) == gpu['job_descriptor_sha256']; paths.extend([gpu_driver,gpu_job])
    for entry in gpu['completed']:
        assert entry['returncode'] == 0
        assert sha(entry['output']) == entry['sha256'] and sha(entry['log']) == entry['log_sha256']
        checked = json.loads(Path(entry['output']).read_text())
        assert checked['script_sha256'] == requirement['checker_sha256']
        assert checked['candidate_sha256'] == candidate and checked['baseline_helper_sha256'] == baseline
        if entry['mode'] == 'numeric':
            assert len(checked['checks']) == 120
        else:
            samples = checked['measurements'][0]['samples']
            assert len(samples) == (1 if entry['variant'] == 'baseline' else 3)
            if entry['variant'] == 'baseline': assert all(x['oom'] for x in samples)
            else: assert all(not x['oom'] and x['ids_equal'] and x['rng_equal'] for x in samples)
        paths.extend([Path(entry['output']),Path(entry['log'])])
    assert gpu['restored']['verification'] == dict(passed=True,slots=16,context=240128)
    assert gpu['restored']['original_command_reused'] and gpu['restored']['api_key_preserved']
    hooks = json.loads(Path(requirement['hooks']).read_text())
    assert hooks['passed'] and hooks['candidate_sha256'] == candidate and hooks['returncode'] == 0
    assert sha(hooks['log']) == hooks['log_sha256']; paths.append(Path(hooks['log']))
    whole_path = ROOT/'results/noise420-whole-model-proof.json'; whole = json.loads(whole_path.read_text())
    assert whole['status'] == 'noise420_whole_model_completed' and whole['failure'] is None
    assert whole['source_retained_for_final_review'] and whole['promotion']['passed']
    assert profile['case'] in whole['qualified_cases']
    assert whole['final_source_sha256'] == candidate and whole['final_topk_source_sha256'] == NEW
    assert whole['script_sha256'] == requirement['driver_sha256'] == 'd48f4ec8d3fc09dca8bf8f37d5f9442762f8dcab57f977f1733ebb0c9eeb5cf4'
    job = ROOT/'cache/noise420-job.json'
    assert sha(job) == whole['job_descriptor_sha256'] == requirement['job_descriptor_sha256']
    assert whole['current_public']['original_command_reused'] and whole['current_public']['api_key_preserved']
    paths.extend([whole_path,job])
    stage = next(x for x in whole['stages'] if x['case'] == profile['case'])
    assert stage['qualified'] and sha(stage['journal']) == stage['journal_sha256'] and sha(stage['plan']) == stage['plan_sha256']
    plan = json.loads(Path(stage['plan']).read_text()); selected = next(x for x in plan['cases'] if x['name'] == profile['case'])
    assert selected['noise400_prerequisite'] == requirement
    assert plan['candidate_source_sha256'][requirement['serving_relative_path']] == candidate
    rows = [json.loads(x) for x in Path(stage['journal']).read_text().splitlines()]
    assert rows[-1]['kind'] == 'run_complete'
    assert any(x['kind'] == 'case_complete' and x.get('case') == profile['case'] for x in rows)
    assert not any(x['kind'] == 'case_failed' and x.get('case') == profile['case'] for x in rows)
    assert stage['restored']['verification']['passed'] and stage['restored']['verification']['context'] == 240128
    assert stage['restored']['original_command_reused'] and stage['restored']['api_key_preserved']
    paths.extend([Path(stage['journal']),Path(stage['plan'])])
    backup = ROOT/'scripts/baseline_sampler_noise420.py'; assert sha(backup) == baseline; paths.append(backup)
    manifest = {x['path']:x for x in profile['file_manifest']}
    assert manifest[str(helper)]['sha256'] == candidate
    for path in paths: manifest[str(path)] = dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path))
    profile['file_manifest'] = sorted(manifest.values(),key=lambda x:x['path'])
    profile['noise_validation436'] = dict(cpu_proof_sha256=sha(requirement['cpu']),gpu_proof_sha256=sha(requirement['gpu']),
        helper_source_sha256=candidate,whole_model_proof_sha256=sha(whole_path),stage_journal_sha256=stage['journal_sha256'],
        scope='Exact helper400 CPU/GPU probability bits, RNG, fullSampler, layouts and captured replay; normal latency<=1.05; actual pressure oldOOM/new3exact; source hooks; actual whole model full-context, MM, mixed and cold gates. Two repeats are selection evidence, not statistical significance.')
    return profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    output = extend(json.loads(args.profile.read_text()))
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(dict(status='noise436_whole_qualified_provenance_bound', case=output['case'],
        output=str(args.output), sha256=sha(args.output))))


if __name__ == '__main__':
    main()
