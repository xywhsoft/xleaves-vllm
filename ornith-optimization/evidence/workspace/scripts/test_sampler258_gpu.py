"""Inactive-source GPU sampler checks; refuse concurrent optimization or serving."""
import argparse
import fcntl
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
TARGET = ROOT / 'src/onecat-padded-kv140'
BASE = '35c2aa9437444104df8f23f3c1b6abe2efaa6e6909257e6a7cdebd4aeb8b3dd3'
CANDIDATE = 'dbd0c827ebb7ca431e9cb26ceff1f8401abe11a396220023da98a61366e70922'
HELPER = 'ab2ee9e52cc229ced835e840cb90cea3a1d7ed2d927d86507b670b532a56d52f'
CPU = '57de58f126ca8cd64c6916d100942a33e3b9302288e4b424225f26429de803a8'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    assert sha(TARGET / 'vllm/v1/sample/ops/topk_topp_triton.py') == BASE
    assert sha(TARGET / 'vllm/v1/sample/ops/topk_topp_sampler.py') == HELPER
    candidate = ROOT / 'scripts/topk_topp_reference254_candidate.py'
    assert sha(candidate) == CANDIDATE
    proof = ROOT / 'results/sampler255-cpu-bitwise-prerequisites.json'
    assert sha(proof) == CPU
    value = json.loads(proof.read_text())
    assert value['candidate_sha256'] == CANDIDATE and len(value['checks']) == 180
    assert not value['cuda_initialized'] and not value['source_mutation_performed']
    assert sha(ROOT / 'scripts/test_sampler255_cpu.py') == value['script_sha256']
    from deploy_verified228 import terminal_jobs
    return terminal_jobs()


def save(path, result):
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(dict(status=result['status'], output=str(path), sha256=sha(path))), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, choices=(0, 1), default=0)
    parser.add_argument('--mode', choices=('numeric', 'pressure'), default='numeric')
    parser.add_argument('--variant', choices=('baseline', 'candidate'))
    parser.add_argument('--lock-fd', type=int)
    parser.add_argument('--guard-only', action='store_true')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    if args.guard_only:
        try:
            jobs = prepare()
        except AssertionError as error:
            save(args.output, dict(status='sampler258_cpu_guard_rejected_active_or_invalid_inputs', reason=str(error),
                utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), script_sha256=sha(__file__),
                torch_imported='torch' in sys.modules, gpu_tests_run=False, source_mutation_performed=False))
            return
        save(args.output, dict(status='sampler258_cpu_prerequisites_terminal_only', jobs=jobs,
            script_sha256=sha(__file__), torch_imported='torch' in sys.modules, gpu_tests_run=False))
        return
    jobs = prepare()
    from deploy_verified228 import restored_runs
    restored = restored_runs()
    lock_path = ROOT / 'cache/gpu.lock'
    lock = lock_path.open('a') if args.lock_fd is None else os.fdopen(os.dup(args.lock_fd), 'a')
    actual, expected = os.fstat(lock.fileno()), lock_path.stat()
    assert (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino)
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    assert not apps.strip(), 'GPU must be unused; caller must stop and restore the idle public service in finally'
    os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
    sys.path.insert(0, str(TARGET))
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_device(args.device)
    device = torch.device('cuda', args.device)
    assert torch.cuda.get_device_capability(device) == (7, 0)
    import gguf
    reader = gguf.GGUFReader('/data/models/Ornith-1.5-35B-CRACK/Ornith-1.5-35B-A3B-CRACK-Q4_K_M.gguf', 'r')
    vocab = len(reader.fields['tokenizer.ggml.tokens'].data)
    assert vocab == 248320
    assert all(int(t.shape[1]) == vocab for t in reader.tensors if t.name in ('output.weight', 'token_embd.weight'))
    del reader
    from vllm.v1.sample.ops import topk_topp_triton as original
    from vllm.v1.sample.ops import topk_topp_sampler as public
    assert Path(original.__file__).resolve() == TARGET / 'vllm/v1/sample/ops/topk_topp_triton.py'
    assert Path(public.__file__).resolve() == TARGET / 'vllm/v1/sample/ops/topk_topp_sampler.py'
    spec = importlib.util.spec_from_file_location('inactive_sampler254', ROOT / 'scripts/topk_topp_reference254_candidate.py')
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    modules = dict(baseline=original, candidate=candidate)
    from test_sampler255_cpu import dense
    pairs = {name: dense(module.__file__, 'reference_' + name, public.apply_top_k_top_p_pytorch) for name, module in modules.items()}
    generator = torch.Generator(device=device).manual_seed(258)
    checks, measurements = [], []

    def equal(a, b):
        assert torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)), 'Different output bits'

    def clear():
        original.reset_buffer_cache()
        candidate.reset_buffer_cache()
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    def make(count, mode):
        values = torch.randn(count, vocab, device=device, generator=generator)
        if mode == 'ties':
            values = values.round()
            values[:, :128] = -float('inf')
        if mode == 'ieee':
            values[:, 0] = -0.0
            values[:, 1] = float('inf')
            values[:, 2] = -float('inf')
            values.view(torch.int32)[:, 3] = 2143289345
            values.view(torch.int32)[:, 4] = -4194303
        k = None if mode == 'top_p_only' else torch.full((count,), vocab - 1, dtype=torch.int32, device=device)
        p = None if mode == 'top_k_only' else torch.full((count,), 0.9, device=device)
        return values, k, p

    with torch.inference_mode():
        if args.mode == 'pressure':
            assert args.variant is not None
            module = modules[args.variant]
            values, k, p = make(176, 'broad')
            clear()
            free, total = torch.cuda.mem_get_info(device)
            leave = 550 * 1024**2
            assert free > leave + 1024**3
            ballast_bytes = (free - leave) // (2 * 1024**2) * (2 * 1024**2)
            ballast = torch.empty(ballast_bytes, dtype=torch.uint8, device=device)
            torch.cuda.synchronize(device)
            free_before, _ = torch.cuda.mem_get_info(device)
            torch.cuda.reset_peak_memory_stats(device)
            allocated_before = torch.cuda.memory_allocated(device)
            oom, error = False, None
            try:
                output = module.apply_top_k_top_p_triton(values, k, p)
                torch.cuda.synchronize(device)
                assert torch.isfinite(output).any(dim=-1).all()
            except torch.cuda.OutOfMemoryError as failure:
                oom, error = True, str(failure)
            measurements.append(dict(variant=args.variant, batch=176, vocab=vocab, ballast_bytes=ballast_bytes,
                free_before=free_before, total_memory=total, oom=oom, error=error,
                peak_extra_allocated=torch.cuda.max_memory_allocated(device) - allocated_before,
                scope='Synthetic free-memory pressure on actual full filter; no model weights or whole-service capacity proof.'))
            del ballast
            clear()
        else:
            assert args.variant is None
            for count in (1, 3, 16, 160, 176):
                for mode in ('broad', 'ties', 'top_p_only', 'top_k_only', 'ieee'):
                    values, k, p = make(count, mode)
                    for selected in sorted({1, min(3, count), count}):
                        mask = torch.zeros(count, dtype=torch.bool, device=device)
                        mask[:selected] = True
                        initial = torch.full_like(values, 17.0)
                        saved = values.clone()
                        for mask_value in (-float('inf'), -123.0):
                            before = pairs['baseline'](initial.clone(), values, mask, k, p, mask_value)
                            after = pairs['candidate'](initial.clone(), values, mask, k, p, mask_value)
                            equal(before, after)
                            equal(values, saved)
                            equal(after[~mask], initial[~mask])
                            checks.append(dict(kind='dense_reference', batch=count, mode=mode, reference_rows=selected,
                                finite_mask=mask_value != -float('inf'), output_bitwise_equal=True, original_preserved=True))
                    # Non-finite values exercise the changed reference block;
                    # they are outside the pivot kernel's valid finite-input cases.
                    if mode != 'ieee':
                        before = original.apply_top_k_top_p_triton(values.clone(), k, p)
                        after = candidate.apply_top_k_top_p_triton(values.clone(), k, p)
                        equal(before, after)
                        checks.append(dict(kind='actual_full_filter', batch=count, mode=mode, output_bitwise_equal=True))
                    del values, saved, before, after, initial, mask, k, p
                    clear()
            for count in (3, 16, 160, 176):
                values, k, p = make(count, 'broad')
                state = torch.cuda.get_rng_state(device)
                sampler = public.TopKTopPSampler(logprobs_mode='processed_logits')
                results = []
                try:
                    for name, module in modules.items():
                        public.apply_top_k_top_p_triton = module.apply_top_k_top_p_triton
                        torch.cuda.set_rng_state(state, device)
                        gens = {i: torch.Generator(device=device).manual_seed(700 + i) for i in sorted({0, count - 1})}
                        ids, logits = sampler.forward_native(values.clone(), gens, k, p)
                        results.append((ids, logits, torch.cuda.get_rng_state(device), {i:g.get_state() for i,g in gens.items()}))
                    assert torch.equal(results[0][0], results[1][0])
                    equal(results[0][1], results[1][1])
                    assert torch.equal(results[0][2], results[1][2])
                    assert all(torch.equal(results[0][3][i], results[1][3][i]) for i in results[0][3])
                    checks.append(dict(kind='actual_sampler', batch=count, sampled_ids_equal=True,
                        filtered_logits_bitwise_equal=True, default_and_request_rng_states_equal=True))
                finally:
                    public.apply_top_k_top_p_triton = original.apply_top_k_top_p_triton
                del results, sampler, values, k, p
                clear()
            for count in (16, 160, 176):
                values, k, p = make(count, 'broad')
                for name, module in modules.items():
                    samples, peaks = [], []
                    module.apply_top_k_top_p_triton(values.clone(), k, p)
                    torch.cuda.synchronize(device)
                    for repeat in range(8):
                        argument = values.clone()
                        torch.cuda.synchronize(device)
                        initial_bytes = torch.cuda.memory_allocated(device)
                        torch.cuda.reset_peak_memory_stats(device)
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        wall_start = time.perf_counter()
                        start.record()
                        output = module.apply_top_k_top_p_triton(argument, k, p)
                        end.record()
                        end.synchronize()
                        samples.append(dict(gpu_ms=start.elapsed_time(end), wall_ms=(time.perf_counter()-wall_start)*1000))
                        peaks.append(torch.cuda.max_memory_allocated(device) - initial_bytes)
                        del output, argument
                    measurements.append(dict(kind='full_filter_latency_and_workspace', variant=name, batch=count,
                        vocab=vocab, samples=samples, gpu_ms_median=statistics.median(x['gpu_ms'] for x in samples),
                        wall_ms_median=statistics.median(x['wall_ms'] for x in samples), peak_extra_allocated_samples=peaks))
                    clear()
                del values, k, p
            values, k, p = make(16, 'broad')
            expected = public.apply_top_k_top_p_pytorch(values.clone(), k, p)
            for name, module in modules.items():
                argument = values.clone()
                public.apply_top_k_top_p_pytorch(argument.clone(), k, p)
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = module.apply_top_k_top_p_triton(argument, k, p)
                argument.copy_(values)
                graph.replay()
                torch.cuda.synchronize(device)
                equal(output, expected)
                checks.append(dict(kind='captured_existing_reference_route', variant=name, batch=16, output_bitwise_equal=True))
                del graph, output, argument
            del values, k, p, expected
            clear()
    assert sha(TARGET / 'vllm/v1/sample/ops/topk_topp_triton.py') == BASE
    assert sha(TARGET / 'vllm/v1/sample/ops/topk_topp_sampler.py') == HELPER
    save(args.output, dict(status='sampler258_gpu_' + args.mode + '_completed',
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), device=args.device,
        gpu_name=torch.cuda.get_device_name(device), torch_version=torch.__version__, vocab=vocab,
        script_sha256=sha(__file__), original_sha256=BASE, candidate_sha256=CANDIDATE,
        helper_sha256=HELPER, cpu_proof_sha256=CPU, terminal_jobs=jobs, restored_runs=restored,
        checks=checks, measurements=measurements, source_mutation_performed=False,
        model_requests_sent=0, deployment_performed=False,
        scope='Actual sampler operator checks on one idle V100, original/candidate source and full GGUF vocabulary. No whole-model TPS, concurrency capacity or theoretical ceiling proof. Pressure mode reports observations and does not assert an expected winner.'))


if __name__ == '__main__':
    main()
