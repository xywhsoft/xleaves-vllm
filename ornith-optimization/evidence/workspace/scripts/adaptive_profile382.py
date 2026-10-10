"""Bind exact adaptive sampler operator/source hooks to whole-model evidence."""
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
    whole_path = ROOT / 'results/adaptive381-whole-model-proof.json'
    whole = json.loads(whole_path.read_text())
    assert whole['status'] == 'adaptive381_whole_model_completed' and whole['failure'] is None
    assert profile['case'] in whole['qualified_cases'] and whole['source_retained_for_final_review']
    assert whole['final_source_sha256'] == NEW and whole['script_sha256'] == requirement['driver_sha256']
    assert sha(whole['journal']) == whole['journal_sha256']
    assert whole['restored']['verification']['passed'] and whole['restored']['verification']['context'] == 240128
    assert whole['current_public']['original_command_reused'] and whole['current_public']['api_key_preserved']
    job = ROOT / 'cache/adaptive381-job.json'
    assert sha(job) == whole['job_descriptor_sha256'] == requirement['job_descriptor_sha256']
    backup = ROOT / 'scripts/baseline_sampler_adaptive381.py'
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
    output['adaptive_validation382'] = dict(cpu_proof_sha256=CPU, gpu_proof_sha256=GPU, source_sha256=NEW,
        whole_model_proof_sha256=sha(whole_path), whole_model_validation=validation,
        scope='Exact adaptive source CPU/GPU bits, RNG/capture, actual OOM/cached replay and normal latency, source hooks, whole-model capacity, independent240K, images, mixed cache and cold16 gates. Operator timing is not model TPS.')
    audit(output)
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    output = extend(json.loads(args.profile.read_text()))
    args.output.write_text(json.dumps(output, indent=2))
    print(json.dumps(dict(status='adaptive382_whole_qualified_provenance_bound', case=output['case'],
        output=str(args.output), sha256=sha(args.output))))


if __name__ == '__main__':
    main()
