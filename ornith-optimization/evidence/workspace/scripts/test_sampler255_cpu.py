"""CPU semantic checks over the actual original/candidate fallback blocks; no GPU work."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import time

import torch

ROOT = Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
TARGET = ROOT / 'src/onecat-padded-kv140'
BASE = '35c2aa9437444104df8f23f3c1b6abe2efaa6e6909257e6a7cdebd4aeb8b3dd3'
HELPER = 'ab2ee9e52cc229ced835e840cb90cea3a1d7ed2d927d86507b670b532a56d52f'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def functions(path):
    return {n.name: n for n in ast.parse(Path(path).read_text()).body if isinstance(n, ast.FunctionDef)}


def dense(path, name, helper):
    fn = functions(path)['apply_top_k_top_p_triton']
    block = next(n for n in reversed(fn.body) if isinstance(n, ast.If)
        and isinstance(n.test, ast.Call) and isinstance(n.test.func, ast.Attribute)
        and n.test.func.attr == 'any' and isinstance(n.test.func.value, ast.Name)
        and n.test.func.value.id == 'reference_rows')
    body = [copy.deepcopy(n) for n in block.body if not isinstance(n, ast.ImportFrom)]
    node = ast.parse(f'def {name}(logits, original_logits, reference_rows, k, p, mask_value):\n    pass').body[0]
    node.body = body + [ast.Return(value=ast.Name(id='logits', ctx=ast.Load()))]
    scope = dict(torch=torch, apply_top_k_top_p_pytorch=helper)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), 'exec'), scope)
    # The caller normally enters this block only for a nonempty reference set.
    return scope[name]


def main():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    base = TARGET / 'vllm/v1/sample/ops/topk_topp_triton.py'
    sampler = TARGET / 'vllm/v1/sample/ops/topk_topp_sampler.py'
    candidate = ROOT / 'scripts/topk_topp_reference254_candidate.py'
    assert sha(base) == BASE and sha(sampler) == HELPER
    candidate_source = candidate.read_text()
    compile(candidate_source, str(candidate), 'exec')
    helper_ast = functions(sampler)['apply_top_k_top_p_pytorch']
    scope = dict(torch=torch)
    exec(compile(ast.Module(body=[helper_ast], type_ignores=[]), str(sampler), 'exec'), scope)
    helper = scope['apply_top_k_top_p_pytorch']
    baseline = dense(base, 'original', helper)
    calls = []
    def tracked(*args, **kwargs):
        calls.append(dict(rows=args[0].shape[0], columns=args[0].shape[1], rowwise=kwargs.get('rowwise_sort', False)))
        return helper(*args, **kwargs)
    updated = dense(candidate, 'updated', tracked)
    generator = torch.Generator().manual_seed(254)
    checks = []
    for count in (1, 3, 7, 16, 160, 176):
        for mode in ('broad', 'ties', 'top_p_only', 'top_k_only', 'ieee'):
            vocab = 2048
            source = torch.randn(count, vocab, generator=generator)
            if mode == 'ties':
                source = source.round()
                source[:, :31] = -float('inf')
            if mode == 'ieee':
                source[:, 0] = -0.0
                source[:, 1] = float('inf')
                source[:, 2] = -float('inf')
                source.view(torch.int32)[:, 3] = 2143289345
                source.view(torch.int32)[:, 4] = -4194303
            k = None if mode == 'top_p_only' else torch.full((count,), vocab - 1, dtype=torch.int32)
            p = None if mode == 'top_k_only' else torch.full((count,), 0.9)
            for selected in (1, min(count, 3), count):
                indices = torch.arange(count)[::max(1, count // selected)][:selected]
                mask = torch.zeros(count, dtype=torch.bool)
                mask[indices] = True
                for mask_value in (-float('inf'), -123.0):
                    initial = torch.full_like(source, 17.0)
                    original_saved = source.clone()
                    k_saved = None if k is None else k.clone()
                    p_saved = None if p is None else p.clone()
                    before = baseline(initial.clone(), source, mask, k, p, mask_value)
                    calls.clear()
                    after = updated(initial.clone(), source, mask, k, p, mask_value)
                    torch.testing.assert_close(after, before, rtol=0, atol=0, equal_nan=True)
                    assert torch.equal(after.contiguous().view(torch.int32), before.contiguous().view(torch.int32))
                    assert torch.equal(source.view(torch.int32), original_saved.view(torch.int32))
                    assert k is None or torch.equal(k, k_saved)
                    assert p is None or torch.equal(p, p_saved)
                    assert torch.equal(after[~mask], initial[~mask])
                    assert all(call['rows'] <= 2 and call['rowwise'] for call in calls)
                    assert sum(call['rows'] for call in calls) == selected + (selected % 2 if selected > 1 else 0)
                    checks.append(dict(batch=count, mode=mode, reference_rows=selected, finite_mask=mask_value != -float('inf'),
                        bitwise_equal=True, maximum_dense_rows=max(call['rows'] for call in calls), source_and_parameters_unchanged=True))
    plan = ROOT / 'scripts/boundary200-plan.json'
    assert sha(plan) == '3ca29534138bc4c8a6dbf47482c7063db45d25df5f52cf78ff937e236c336c07'
    failure_logs = []
    for name in ('mm_aliasvalue80_acc119_pad135_moe163_str177_ws187_bd200_c176_kv18387828736_b512_full16_none',
                 'mm_aliasvalue80_acc119_pad135_moe163_str177_ws187_bd200_c160_kv17716740096_b1024_full16_none'):
        path = ROOT / 'logs' / f'workspace187_boundary200_{name}.log'
        text = path.read_text(errors='replace')
        assert 'OutOfMemoryError' in text and 'warming up sampler with' in text
        assert 'reference_logits = apply_top_k_top_p_pytorch(' in text and 'logits_sort, logits_idx = logits.sort' in text
        failure_logs.append(dict(case=name, path=str(path), sha256=sha(path), dense_sampler_fallback_oom_confirmed=True))
    assert not torch.cuda.is_initialized()
    output = ROOT / 'results/sampler255-cpu-bitwise-prerequisites.json'
    assert not output.exists()
    result = dict(status='sampler255_cpu_exact_dense_fallback_bitwise_semantics_passed',
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), source_before_sha256=BASE,
        helper_sha256=HELPER, candidate_sha256=sha(candidate), script_sha256=sha(Path(__file__).resolve()),
        boundary_plan_sha256=sha(plan), actual_failure_logs=failure_logs, checks=checks,
        torch_version=torch.__version__, torch_threads=1, cuda_initialized=False,
        source_mutation_performed=False, model_requests_sent=0, deployment_performed=False,
        scope='CPU checks of actual source blocks: 2048-vocab fixtures, selected/unselected rows, odd/reference-singleton sets, ties, optional k/p, finite masks, signed zero and explicit positive/negative NaN payloads. Strict int32 bit-pattern equality asserted for every output and original input. No GPU numerical/memory/latency results, no model TPS or qualification. Candidate is inactive; generic broad-k fallback is bounded like the existing compact fallback. Actual GPU/full-vocab/operator and whole-model checks remain required.')
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(status=result['status'], checks=len(checks), candidate_sha256=sha(candidate), proof_sha256=sha(output), cuda_initialized=False)), flush=True)


if __name__ == '__main__':
    main()
