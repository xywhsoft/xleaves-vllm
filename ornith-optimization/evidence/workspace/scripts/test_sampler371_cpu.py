"""Actual-source CPU bit tests for fast, forced-OOM and cached bounded paths."""
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
CANDIDATE = '7e1e04a7af6badc8012cfbcae920495672a57d3a30b07d547adc82f8d8177c52'
HELPER = 'ab2ee9e52cc229ced835e840cb90cea3a1d7ed2d927d86507b670b532a56d52f'
GPU260 = '66c09944b80fe47869019296e501faf93b09505bcdcec8749c553a0d1fcd2dec'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def functions(path):
    return {node.name: node for node in ast.parse(Path(path).read_text()).body if isinstance(node, ast.FunctionDef)}


def dense(path, name, helper, cache):
    fn = functions(path)['apply_top_k_top_p_triton']
    block = next(node for node in reversed(fn.body) if isinstance(node, ast.If)
        and isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Attribute)
        and node.test.func.attr == 'any' and isinstance(node.test.func.value, ast.Name)
        and node.test.func.value.id == 'reference_rows')
    body = [copy.deepcopy(node) for node in block.body if not isinstance(node, ast.ImportFrom)]
    wrapper = ast.parse(f'def {name}(logits, original_logits, reference_rows, k, p, mask_value):\n    pass').body[0]
    wrapper.body = body + [ast.Return(value=ast.Name(id='logits', ctx=ast.Load()))]
    scope = dict(torch=torch, apply_top_k_top_p_pytorch=helper, _DENSE_REFERENCE_CHUNK_SHAPES=cache)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), str(path), 'exec'), scope)
    return scope[name]


def equal(left, right):
    assert torch.equal(left.contiguous().view(torch.int32), right.contiguous().view(torch.int32))


def main():
    assert not torch.cuda.is_initialized()
    torch.set_num_threads(1)
    base = TARGET / 'vllm/v1/sample/ops/topk_topp_triton.py'
    sampler = TARGET / 'vllm/v1/sample/ops/topk_topp_sampler.py'
    candidate = ROOT / 'scripts/topk_topp_adaptive370_candidate.py'
    assert sha(base) == BASE and sha(sampler) == HELPER and sha(candidate) == CANDIDATE
    predecessor = ROOT / 'results/sampler260-gpu-operator-proof.json'
    assert sha(predecessor) == GPU260
    prior = json.loads(predecessor.read_text())
    assert prior['status'] == 'sampler260_gpu_operator_checks_passed' and len(prior['completed']) == 6
    assert prior['restored']['verification']['passed'] and not prior['source_mutation_performed']
    for entry in prior['completed']:
        assert sha(entry['output']) == entry['sha256'] and entry['returncode'] == 0
    compile(candidate.read_text(), str(candidate), 'exec')
    helper_node = functions(sampler)['apply_top_k_top_p_pytorch']
    scope = dict(torch=torch)
    exec(compile(ast.Module(body=[helper_node], type_ignores=[]), str(sampler), 'exec'), scope)
    helper = scope['apply_top_k_top_p_pytorch']
    baseline = dense(base, 'original', helper, set())
    cache, calls, failures = set(), [], []
    force_pressure = False
    force_other_error = False

    def tracked(*args, **kwargs):
        rows = args[0].shape[0]
        rowwise = kwargs.get('rowwise_sort', False)
        if force_other_error:
            raise RuntimeError('controlled non-memory failure')
        if force_pressure and rows > 2 and not rowwise:
            failures.append(rows)
            raise torch.cuda.OutOfMemoryError('controlled CPU fixture allocation failure')
        calls.append(dict(rows=rows, rowwise=rowwise))
        return helper(*args, **kwargs)

    updated = dense(candidate, 'updated', tracked, cache)
    generator = torch.Generator().manual_seed(371)
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
                    source_copy = source.clone()
                    k_copy = None if k is None else k.clone()
                    p_copy = None if p is None else p.clone()
                    expected = baseline(initial.clone(), source, mask, k, p, mask_value)
                    for force_pressure in (False, True):
                        cache.clear()
                        calls.clear()
                        failures.clear()
                        after = updated(initial.clone(), source, mask, k, p, mask_value)
                        equal(after, expected)
                        equal(source, source_copy)
                        equal(after[~mask], initial[~mask])
                        assert k is None or torch.equal(k, k_copy)
                        assert p is None or torch.equal(p, p_copy)
                        chunked = force_pressure and selected > 2
                        assert failures == ([selected] if chunked else [])
                        assert len(cache) == (1 if chunked else 0)
                        if chunked:
                            assert all(call['rows'] <= 2 and call['rowwise'] for call in calls)
                            assert sum(call['rows'] for call in calls) == selected + selected % 2
                        else:
                            assert calls == [dict(rows=selected, rowwise=False)]
                        calls.clear()
                        failure_count = len(failures)
                        replay = updated(initial.clone(), source, mask, k, p, mask_value)
                        equal(replay, expected)
                        assert len(failures) == failure_count
                        assert all(call['rows'] <= 2 and call['rowwise'] for call in calls) if chunked else calls == [dict(rows=selected, rowwise=False)]
                        checks.append(dict(batch=count, mode=mode, reference_rows=selected, finite_mask=mask_value != -float('inf'),
                            simulated_pressure=force_pressure, chunked=chunked, first_and_cached_replay_bitwise_equal=True,
                            original_and_parameters_unchanged=True, no_second_failed_dense_attempt=True))
    force_pressure = False
    force_other_error = True
    cache.clear()
    try:
        updated(initial.clone(), source, mask, k, p, mask_value)
    except RuntimeError as error:
        assert str(error) == 'controlled non-memory failure'
    else:
        raise AssertionError('Non-memory failure swallowed')
    assert not cache and not torch.cuda.is_initialized()
    output = ROOT / 'results/sampler371-adaptive-cpu-bitwise-proof.json'
    assert not output.exists()
    value = dict(status='sampler371_adaptive_cpu_fast_oom_cached_semantics_passed',
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), script_sha256=sha(__file__),
        source_before_sha256=BASE, candidate_sha256=CANDIDATE, helper_sha256=HELPER,
        predecessor_gpu260_sha256=GPU260, checks=checks, non_memory_failure_propagated=True,
        cuda_initialized=False, serving_source_mutation=False, model_requests_sent=0,
        scope='Actual source AST blocks and strict int32 output/input bits on CPU2048 vocabulary. Controlled allocation failure exercises first/cached fallback. No actual adaptive GPU OOM/latency or whole-model qualification yet.')
    assert len(checks) == 360
    output.write_text(json.dumps(value, indent=2))
    print(json.dumps(dict(status=value['status'], checks=len(checks), proof_sha256=sha(output), candidate_sha256=CANDIDATE)), flush=True)


if __name__ == '__main__':
    main()
