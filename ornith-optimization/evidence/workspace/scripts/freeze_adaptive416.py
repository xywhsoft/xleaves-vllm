"""CPU-only evidence freeze after actual393 terminal; no serving changes."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from pipeline_layer216 import process
from qualified_profile189 import sha, audit
from test_sampler406_gpu import parent_restoration

ROOT = Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
TARGET = ROOT / 'src/onecat-padded-kv140'
HELPER = 'ab2ee9e52cc229ced835e840cb90cea3a1d7ed2d927d86507b670b532a56d52f'
TOPK = '7e1e04a7af6badc8012cfbcae920495672a57d3a30b07d547adc82f8d8177c52'


def record(status, **values):
    value = dict(status=status, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        torch_imported='torch' in sys.modules, serving_source_mutation=False, **values)
    (ROOT / 'results/freeze416-status.json').write_text(json.dumps(value, indent=2))
    print(json.dumps(value), flush=True)


def main():
    proof_path = ROOT / 'results/freeze416-qualified-profiles.json'
    assert not proof_path.exists()
    job = json.loads((ROOT / 'cache/adaptive393-job.json').read_text())
    record('waiting_actual393_terminal', pid=job['pid'], birth=job['birth'])
    deadline = time.monotonic() + 24 * 3600
    while True:
        state, birth = process(job['pid'])
        if state in ('missing', 'Z', 'X') or birth != str(job['birth']):
            break
        assert time.monotonic() < deadline
        time.sleep(10)
    restored = parent_restoration()
    whole_path = ROOT / 'results/adaptive393-whole-model-proof.json'
    whole = json.loads(whole_path.read_text())
    assert whole['status'] == 'adaptive393_whole_model_completed' and whole['failure'] is None
    assert whole['qualified_cases'] and whole['source_retained_for_final_review']
    assert sha(TARGET / 'vllm/v1/sample/ops/topk_topp_sampler.py') == HELPER
    assert sha(TARGET / 'vllm/v1/sample/ops/topk_topp_triton.py') == TOPK
    assert sha(ROOT / 'scripts/adaptive_profile413.py') == '842dc54f67f14d6fb499262fdcf93683fb831a96ad3b5c5631cdaab82afad9b2'
    outputs = []
    for index, case in enumerate(whole['qualified_cases']):
        base = ROOT / 'results' / f'qualified-adaptive416-{index}'
        raw = Path(str(base) + '-raw.json')
        command = Path(str(base) + '-command.json')
        guard = Path(str(base) + '-guard.json')
        final = Path(str(base) + '-final.json')
        commands = [
            ['freeze_qualified193.py', '--journal', whole['journal'], '--plan', str(ROOT / 'scripts/adaptive393-plan.json'),
                '--case', case, '--output', str(raw)],
            ['profile_locality201.py', '--profile', str(raw), '--output', str(command)],
            ['guard_profile213.py', '--profile', str(command), '--output', str(guard)],
            ['adaptive_profile413.py', '--profile', str(guard), '--output', str(final)],
        ]
        for args in commands:
            subprocess.run([sys.executable, str(ROOT / 'scripts' / args[0]), *args[1:]],
                cwd=ROOT / 'scripts', env={**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'}, check=True)
        value = json.loads(final.read_text())
        checked = audit(value)
        outputs.append(dict(case=case, profile=str(final), sha256=sha(final), validation=checked,
            maximum_active_sequences=value['maximum_active_sequences'], minimum_full_contexts=value['minimum_full_contexts']))
        record('freezing_actual_complete_cases', completed=outputs)
    comparison = ROOT / 'results/exact-layer416-qualified-comparison.json'
    args = [sys.executable, str(ROOT / 'scripts/compare_exact_layer229.py'),
        '--baseline', str(ROOT / 'results/llama_exact_layer216.jsonl'),
        '--input-proof', str(ROOT / 'results/exact-layer215-input-proof.json'), '--output', str(comparison)]
    for item in outputs:
        args += ['--profile', item['profile']]
    subprocess.run(args, cwd=ROOT / 'scripts', env={**os.environ, 'CUDA_VISIBLE_DEVICES': '-1'}, check=True)
    value = dict(status='freeze416_actual_complete_profiles_compared', profiles=outputs,
        comparison=str(comparison), comparison_sha256=sha(comparison), script_sha256=sha(__file__),
        parent_restoration=restored, whole_model_proof_sha256=sha(whole_path), serving_source_mutation=False,
        model_requests_sent=0, public_deployment_performed=False)
    proof_path.write_text(json.dumps(value, indent=2))
    record(value['status'], proof=str(proof_path), proof_sha256=sha(proof_path), completed=outputs)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        record('freeze416_failed', error_type=type(error).__name__, error=str(error))
        raise
