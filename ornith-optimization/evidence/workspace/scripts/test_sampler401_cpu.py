"""Actual-source CPU RNG/bit contracts and metadata-only CUDA eligibility."""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
TARGET = ROOT / 'src/onecat-padded-kv140'
BASE = 'ab2ee9e52cc229ced835e840cb90cea3a1d7ed2d927d86507b670b532a56d52f'
NEW = 'afa1e2cb242649c33c42b58d02709dc3984846d7ea0e89417bd05024148f4322'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def functions(path):
    tree = ast.parse(Path(path).read_text())
    result = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'TopKTopPSampler')
    result.update({n.name: n for n in cls.body if isinstance(n, ast.FunctionDef)})
    return result


def compile_functions(nodes):
    scope = dict(torch=torch, apply_top_k_top_p=lambda logits, k, p: logits)
    module = ast.Module(body=[nodes['random_sample'], nodes['forward_native']], type_ignores=[])
    exec(compile(module, 'actual_sampler_source', 'exec'), scope)
    return scope


def equal(left, right):
    assert left.dtype == right.dtype and left.shape == right.shape
    if left.dtype == torch.float32:
        left, right = left.contiguous().view(torch.int32), right.contiguous().view(torch.int32)
    assert torch.equal(left, right)


def generators(count, mode):
    indices = range(count) if mode == 'all' else range(0, count, 3) if mode == 'mixed' else []
    return {i: torch.Generator().manual_seed(4107 + i * 31) for i in indices}


def main():
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    baseline = TARGET / 'vllm/v1/sample/ops/topk_topp_sampler.py'
    candidate = ROOT / 'scripts/topk_topp_sampler400_candidate.py'
    assert sha(baseline) == BASE and sha(candidate) == NEW
    old_nodes, new_nodes = functions(baseline), functions(candidate)
    assert old_nodes.keys() == new_nodes.keys()
    unchanged = []
    for name in old_nodes:
        if name in ('forward_native', 'random_sample'):
            continue
        assert ast.dump(old_nodes[name]) == ast.dump(new_nodes[name]), name
        unchanged.append(dict(name=name, ast_sha256=hashlib.sha256(ast.dump(old_nodes[name]).encode()).hexdigest()))
    old, new = compile_functions(old_nodes), compile_functions(new_nodes)
    checks = []
    input_generator = torch.Generator().manual_seed(401)
    for count in (1, 3, 7, 16, 160, 176):
        for vocab in (17, 1024, 2048):
            probs = torch.randn(count, vocab, generator=input_generator).softmax(dim=-1)
            probs[:, :2] = 0
            for mode in ('none', 'mixed', 'all'):
                for noise_enabled in (False, True):
                    old_g, new_g = generators(count, mode), generators(count, mode)
                    torch.manual_seed(12401)
                    start = torch.get_rng_state().clone()
                    a = probs.clone()
                    expected = old['random_sample'](a, old_g)
                    end = torch.get_rng_state().clone()
                    torch.set_rng_state(start)
                    b = probs.clone()
                    noise = torch.full_like(probs, float('nan')) if noise_enabled else None
                    actual = new['random_sample'](b, new_g, noise=noise)
                    equal(expected, actual)
                    equal(a, b)
                    assert torch.equal(end, torch.get_rng_state())
                    assert all(torch.equal(old_g[i].get_state(), new_g[i].get_state()) for i in old_g)
                    if noise_enabled:
                        assert not torch.isnan(noise).any() and torch.all(noise >= 0)
                    checks.append(dict(kind='actual_random_function', batch=count, vocab=vocab,
                        generators=mode, supplied_noise=noise_enabled, ids_equal=True,
                        probability_bits_equal=True, default_and_request_rng_equal=True))
    for count in (1, 3, 16):
        for mode in (None, 'raw_logits', 'raw_logprobs', 'processed_logits', 'processed_logprobs'):
            for dtype in (torch.float16, torch.float32):
                logits = torch.randn(count, 1024, generator=input_generator).to(dtype)
                torch.manual_seed(12402)
                start = torch.get_rng_state().clone()
                expected_logits = logits.clone()
                expected = old['forward_native'](SimpleNamespace(logprobs_mode=mode), expected_logits, {}, None, None)
                end = torch.get_rng_state().clone()
                torch.set_rng_state(start)
                actual_logits = logits.clone()
                actual = new['forward_native'](SimpleNamespace(logprobs_mode=mode), actual_logits, {}, None, None)
                equal(expected[0], actual[0])
                assert (expected[1] is None) == (actual[1] is None)
                if expected[1] is not None:
                    equal(expected[1], actual[1])
                equal(expected_logits, actual_logits)
                assert torch.equal(end, torch.get_rng_state())
                checks.append(dict(kind='actual_cpu_forward_without_filter', batch=count,
                    logprobs_mode=mode, dtype=str(dtype), ids_outputs_inputs_rng_equal=True,
                    scope='CPU fallback path; filter source AST unchanged. CUDA reuse not exercised here.'))
    noise_node = next(n for n in new_nodes['forward_native'].body
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'noise' for t in n.targets))
    expression = compile(ast.Expression(noise_node.value), 'actual_cuda_eligibility_ast', 'eval')
    eligibility = []
    for cuda in (False, True):
        for returned in (False, True):
            for dtype in (torch.float16, torch.float32):
                for contiguous in (False, True):
                    for aligned in (False, True):
                        tensor = SimpleNamespace(is_cuda=cuda, dtype=dtype,
                            is_contiguous=lambda: contiguous, data_ptr=lambda: 256 if aligned else 260)
                        selected = eval(expression, dict(logits=tensor,
                            logits_to_return=tensor if returned else None, probs=SimpleNamespace(dtype=torch.float32)))
                        allowed = cuda and not returned and dtype == torch.float32 and contiguous and aligned
                        assert (selected is tensor) == allowed
                        eligibility.append(dict(cuda=cuda, returned_logits=returned, dtype=str(dtype),
                            contiguous=contiguous, aligned=aligned, selected=allowed))
    assert len(checks) == 138 and len(eligibility) == 32
    assert not torch.cuda.is_initialized() and sha(baseline) == BASE
    output = ROOT / 'results/sampler401-noise-cpu-contract-proof.json'
    assert not output.exists()
    value = dict(status='sampler401_actual_cpu_contracts_passed', script_sha256=sha(__file__),
        baseline_sha256=BASE, candidate_sha256=NEW, unchanged_function_asts=unchanged,
        checks=checks, metadata_eligibility_fixtures=eligibility, cuda_initialized=False,
        serving_source_mutation=False, actual_cuda_noise_reuse_tested=False,
        scope='138 actual-source CPU random/forward bit and RNG checks plus32 source-AST metadata eligibility fixtures. No CUDA latency, CUDA RNG vectorization, real OOM, whole-model or capacity qualification.')
    output.write_text(json.dumps(value, indent=2))
    print(json.dumps(dict(status=value['status'], output=str(output), sha256=sha(output), checks=len(checks), eligibility=len(eligibility))))


if __name__ == '__main__':
    main()
