"""Inactive noise400 GPU contracts; actual public classes, RNG and memory."""
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
HELPER = 'ab2ee9e52cc229ced835e840cb90cea3a1d7ed2d927d86507b670b532a56d52f'
NEW = 'afa1e2cb242649c33c42b58d02709dc3984846d7ea0e89417bd05024148f4322'
CPU = 'bccd8ac11e89bda9f7bfc03b3164f4a017a2d09bc584d23d726d655f87fae76b'
CALLER = '71058953620a12947b294a76379446497266bb7e5f4e02d294d28da4e15b5f91'
TOPKS = ('35c2aa9437444104df8f23f3c1b6abe2efaa6e6909257e6a7cdebd4aeb8b3dd3',
         '7e1e04a7af6badc8012cfbcae920495672a57d3a30b07d547adc82f8d8177c52')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parent_restoration():
    from pipeline_layer216 import process
    job = json.loads((ROOT / 'cache/adaptive393-job.json').read_text())
    state, birth = process(job['pid'])
    assert state in ('missing', 'Z', 'X') or birth != str(job['birth']), 'Actual393 owner still live'
    assert sha(job['script']) == job['script_sha256'] == 'a363ab7afa2ede5426d4b9ac354956741aa2eac4a8f3635b4166367e05f2523a'
    path = ROOT / 'results/adaptive393-whole-model-proof.json'
    value = json.loads(path.read_text())
    assert value['status'] in ('adaptive393_whole_model_completed', 'adaptive393_whole_model_failed')
    assert value['script_sha256'] == job['script_sha256']
    assert value['job_descriptor_sha256'] == sha(ROOT / 'cache/adaptive393-job.json')
    assert value['restored']['verification'] == dict(passed=True, slots=16, context=240128)
    assert value['restored']['original_command_reused'] and value['restored']['api_key_preserved']
    assert value['current_public']['original_command_reused'] and value['current_public']['api_key_preserved']
    assert sha(value['journal']) == value['journal_sha256']
    assert sha(TARGET / 'vllm/v1/sample/ops/topk_topp_triton.py') == value['final_source_sha256'] in TOPKS
    return dict(path=str(path), sha256=sha(path), final_source_sha256=value['final_source_sha256'],
        restored=value['restored'], qualified_cases=value['qualified_cases'])


def prerequisites(owner_pid=None):
    from final_jobs286 import REQUIRED, check_job
    from pipeline_layer216 import process
    assert sha(TARGET / 'vllm/v1/sample/ops/topk_topp_sampler.py') == HELPER
    assert sha(TARGET / 'vllm/v1/sample/sampler.py') == CALLER
    assert sha(ROOT / 'scripts/topk_topp_sampler400_candidate.py') == NEW
    proof = ROOT / 'results/sampler401-noise-cpu-contract-proof.json'
    assert sha(proof) == CPU
    value = json.loads(proof.read_text())
    assert value['status'] == 'sampler401_actual_cpu_contracts_passed' and len(value['checks']) == 138
    assert len(value['metadata_eligibility_fixtures']) == 32 and not value['cuda_initialized']
    assert sha(ROOT / 'scripts/test_sampler401_cpu.py') == value['script_sha256']
    names = set(REQUIRED)
    for path in (ROOT / 'cache').glob('*-job.json'):
        if json.loads(path.read_text()).get('resource') == 'gpu_exclusive':
            names.add(path.name.removesuffix('-job.json'))
    authorized = False
    jobs = []
    for name in sorted(names):
        path = ROOT / 'cache' / f'{name}-job.json'
        job = json.loads(path.read_text())
        if owner_pid is not None and job['pid'] == owner_pid:
            assert name == 'sampler511' and job['resource'] == 'gpu_exclusive'
            assert job['script'] == str(ROOT / 'scripts/pipeline_sampler511.py')
            assert sha(job['script']) == job['script_sha256']
            state, birth = process(owner_pid)
            assert birth == str(job['birth']) and state not in ('missing', 'Z', 'X')
            assert job['script'].encode() in Path(f'/proc/{owner_pid}/cmdline').read_bytes().split(b'\0')
            authorized = True
        else:
            jobs.append(check_job(name))
    assert owner_pid is None or authorized
    from deploy_verified383 import restored_runs
    previous = restored_runs()
    parent = parent_restoration()
    return dict(jobs=jobs, prior_restorations=previous, parent_restoration=parent)


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
            checked = prerequisites()
            status, reason = 'sampler510_cpu_prerequisites_terminal_only', None
        except (AssertionError, FileNotFoundError) as error:
            checked, status, reason = None, 'sampler510_cpu_guard_rejected', str(error)
        value = dict(status=status, reason=reason, prerequisites=checked, script_sha256=sha(__file__),
            torch_imported='torch' in sys.modules, serving_source_mutation=False, gpu_tests_run=False)
        args.output.write_text(json.dumps(value, indent=2))
        print(json.dumps(dict(status=status, output=str(args.output), sha256=sha(args.output))))
        return
    checked = prerequisites(os.getppid() if args.lock_fd is not None else None)
    assert args.lock_fd is not None, 'Registered restoring driver must own exclusive GPU lock'
    lock = os.fdopen(os.dup(args.lock_fd), 'a')
    actual, expected = os.fstat(lock.fileno()), (ROOT / 'cache/gpu.lock').stat()
    assert (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino)
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    assert not apps.strip(), 'Idle GPUs only; driver stops/restores public service'
    os.environ['VLLM_USE_FLASHINFER_SAMPLER'] = '0'
    sys.path.insert(0, str(TARGET))
    import torch
    torch.set_num_threads(1)
    device = torch.device('cuda', args.device)
    torch.cuda.set_device(device)
    assert torch.cuda.get_device_capability(device) == (7, 0)
    from vllm.v1.sample.ops import topk_topp_sampler as baseline
    from vllm.v1.sample.sampler import Sampler
    from vllm.v1.sample.metadata import SamplingMetadata
    from types import SimpleNamespace
    spec = importlib.util.spec_from_file_location('inactive_noise400', ROOT / 'scripts/topk_topp_sampler400_candidate.py')
    candidate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(candidate)
    import gguf
    reader = gguf.GGUFReader('/data/models/Ornith-1.5-35B-CRACK/Ornith-1.5-35B-A3B-CRACK-Q4_K_M.gguf', 'r')
    vocab = len(reader.fields['tokenizer.ggml.tokens'].data)
    assert vocab == 248320
    del reader
    input_generator = torch.Generator(device=device).manual_seed(406)
    checks, measurements = [], []
    modules = dict(baseline=baseline, candidate=candidate)

    def equal(a, b):
        assert a.dtype == b.dtype and a.shape == b.shape
        if a.dtype in (torch.float32, torch.float16):
            integer = torch.int32 if a.dtype == torch.float32 else torch.int16
            a, b = a.contiguous().view(integer), b.contiguous().view(integer)
        assert torch.equal(a, b), 'Different output bits'

    def clear():
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()

    def gens(count, kind):
        indices = range(count) if kind == 'all' else sorted({0, count - 1}) if kind == 'mixed' else []
        return {i: torch.Generator(device=device).manual_seed(810 + i) for i in indices}

    def argument(values, layout):
        count, width = values.shape
        if layout == 'offset32':
            backing = torch.empty((count + 1, width), dtype=torch.float32, device=device)
            result = backing[1:]
        elif layout == 'unaligned32':
            backing = torch.empty(count * width + 1, dtype=torch.float32, device=device)
            result = backing[1:].view(count, width)
        elif layout == 'strided32':
            backing = torch.empty((count, width + 1), dtype=torch.float32, device=device)
            result = backing[:, :width]
        else:
            return values.to(torch.float16).clone() if layout == 'half' else values.clone()
        result.copy_(values)
        return result

    def states(generators):
        return torch.cuda.get_rng_state(device), {i: g.get_state() for i, g in generators.items()}

    def compare_states(a, b):
        assert torch.equal(a[0], b[0]) and a[1].keys() == b[1].keys()
        assert all(torch.equal(a[1][i], b[1][i]) for i in a[1])

    if args.mode == 'pressure':
        assert args.variant is not None
        count = 176
        values = torch.randn(count, vocab, device=device, generator=input_generator)
        probe = values.clone()
        torch.cuda.manual_seed(82406)
        expected_ids, _ = baseline.TopKTopPSampler(logprobs_mode='raw_logprobs').forward_native(probe, {}, None, None)
        expected_ids = expected_ids.cpu()
        expected_state = torch.cuda.get_rng_state(device)
        del probe
        clear()
        actual_argument = values.clone()
        clear()
        free, total = torch.cuda.mem_get_info(device)
        ballast_bytes = max(0, ((free - 250 * 1024**2) // (2 * 1024**2)) * (2 * 1024**2))
        ballast = torch.empty(ballast_bytes, dtype=torch.uint8, device=device)
        free_before, _ = torch.cuda.mem_get_info(device)
        assert 230 * 1024**2 <= free_before <= 280 * 1024**2, free_before
        sampler = modules[args.variant].TopKTopPSampler(logprobs_mode='raw_logprobs')
        samples = []
        for repeat in range(3 if args.variant == 'candidate' else 1):
            actual_argument.copy_(values)
            torch.cuda.manual_seed(82406)
            torch.cuda.synchronize(device)
            initial = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            oom, error, output_equal, rng_equal = False, None, False, False
            output = None
            try:
                output, returned = sampler.forward_native(actual_argument, {}, None, None)
                torch.cuda.synchronize(device)
                assert returned is None
                output_equal = torch.equal(output.cpu(), expected_ids)
                rng_equal = torch.equal(torch.cuda.get_rng_state(device), expected_state)
                assert output_equal and rng_equal
            except torch.cuda.OutOfMemoryError as failure:
                oom, error = True, str(failure)
            samples.append(dict(repeat=repeat, oom=oom, error=error, ids_equal=output_equal,
                rng_equal=rng_equal, wall_ms=(time.perf_counter()-start)*1000,
                peak_extra_allocated=torch.cuda.max_memory_allocated(device)-initial))
            del output
            if oom:
                break
        measurements.append(dict(kind='actual_low_memory_native_sampler', variant=args.variant,
            batch=count, vocab=vocab, free_before=free_before, ballast_bytes=ballast_bytes,
            total_memory=total, samples=samples))
        del ballast, actual_argument, values, expected_ids, sampler
        clear()
    else:
        assert args.variant is None
        # Verify the mutated probability bits as well as the selected token and
        # default/per-request CUDA RNG state when reusing aligned storage.
        for count in (1, 16, 176):
            probabilities = torch.softmax(torch.randn(count, vocab, device=device,
                generator=input_generator), dim=-1)
            results = []
            for kind in ('none', 'mixed', 'all'):
                for name, module in modules.items():
                    probs = probabilities.clone()
                    noise = torch.empty_like(probs) if name == 'candidate' else None
                    torch.cuda.manual_seed(81406)
                    generators = gens(count, kind)
                    ids = module.random_sample(probs, generators, **({'noise': noise} if noise is not None else {}))
                    results.append((ids, probs, states(generators)))
                equal(results[0][0], results[1][0])
                equal(results[0][1], results[1][1])
                compare_states(results[0][2], results[1][2])
                checks.append(dict(kind='actual_cuda_random_probability_bits', batch=count,
                    generators=kind, ids_probability_bits_rng_equal=True))
                results.clear()
            del probabilities, results, probs, noise, ids, generators
            clear()
        for count in (1, 3, 16, 160, 176):
            values = torch.randn(count, vocab, device=device, generator=input_generator)
            values[:, :17] = -float('inf')
            for layout in ('contiguous32', 'offset32', 'unaligned32', 'strided32', 'half'):
                for kind in ('none', 'mixed', 'all'):
                    outputs, rng_states = [], []
                    noise_choices = []
                    original_random = candidate.random_sample
                    def tracked(probs, generators, *, noise=None):
                        noise_choices.append(noise is not None)
                        return original_random(probs, generators, noise=noise)
                    candidate.random_sample = tracked
                    try:
                        expected_reuse = None
                        for name, module in modules.items():
                            arg = argument(values, layout)
                            if name == 'candidate':
                                expected_reuse = arg.dtype == torch.float32 and arg.is_contiguous() and arg.data_ptr() % 16 == 0
                            torch.cuda.manual_seed(83406)
                            generators = gens(count, kind)
                            ids, returned = module.TopKTopPSampler(logprobs_mode='raw_logprobs').forward_native(arg, generators, None, None)
                            assert returned is None
                            outputs.append(ids)
                            rng_states.append(states(generators))
                        equal(outputs[0], outputs[1])
                        compare_states(rng_states[0], rng_states[1])
                        assert noise_choices == [expected_reuse]
                        checks.append(dict(kind='actual_native_layout_rng', batch=count, layout=layout,
                            generators=kind, reused_noise=expected_reuse, ids_rng_equal=True))
                    finally:
                        candidate.random_sample = original_random
                    del outputs, rng_states, arg, ids, generators
            # Returned intermediate tensors must keep the original allocation route.
            for mode in ('processed_logits', 'processed_logprobs'):
                if count not in (1, 16, 176):
                    continue
                results = []
                for name, module in modules.items():
                    torch.cuda.manual_seed(84406)
                    generators = gens(count, 'mixed')
                    output = module.TopKTopPSampler(logprobs_mode=mode).forward_native(values.clone(), generators, None, None)
                    results.append((output, states(generators)))
                equal(results[0][0][0], results[1][0][0])
                equal(results[0][0][1], results[1][0][1])
                compare_states(results[0][1], results[1][1])
                checks.append(dict(kind='returned_native_intermediates', batch=count, logprobs_mode=mode, ids_outputs_rng_equal=True))
                del results, output, generators
            del values
            clear()
        # Actual complete Sampler.forward, including raw preservation and gathered outputs.
        for mode in ('raw_logits', 'raw_logprobs', 'processed_logits', 'processed_logprobs'):
            for pattern, logprobs in [('random', None), ('random', 5), ('random', -1),
                    ('mixed', 5), ('greedy', 5), ('specific', None)]:
                count = 3
                values = torch.randn(count, vocab, device=device, generator=input_generator)
                results = []
                for name, module in modules.items():
                    torch.cuda.manual_seed(85406)
                    generators = gens(count, 'mixed')
                    temperature = torch.tensor([0., 1., .7] if pattern == 'mixed' else
                        [0., 0., 0.] if pattern == 'greedy' else [1., 1., 1.], device=device)
                    metadata = SamplingMetadata(temperature=temperature, all_greedy=pattern == 'greedy',
                        all_random=pattern not in ('greedy', 'mixed'), top_p=None, top_k=None, generators=generators, max_num_logprobs=logprobs,
                        no_penalties=True, prompt_token_ids=None, frequency_penalties=torch.zeros(count, device=device),
                        presence_penalties=torch.zeros(count, device=device), repetition_penalties=torch.ones(count, device=device),
                        output_token_ids=[[] for _ in range(count)], allowed_token_ids_mask=None, bad_words_token_ids={},
                        logitsprocs=SimpleNamespace(non_argmax_invariant=[], argmax_invariant=[]),
                        logprob_token_ids={0: [17, 1000], 2: [2000]} if pattern == 'specific' else None)
                    sampler = Sampler(logprobs_mode=mode)
                    sampler.topk_topp_sampler = module.TopKTopPSampler(logprobs_mode=mode)
                    output = sampler(values.clone(), metadata)
                    results.append((output, states(generators)))
                equal(results[0][0].sampled_token_ids, results[1][0].sampled_token_ids)
                a, b = results[0][0].logprobs_tensors, results[1][0].logprobs_tensors
                assert (a is None) == (b is None)
                if a is not None:
                    for field in ('logprob_token_ids', 'logprobs', 'selected_token_ranks'):
                        equal(getattr(a, field), getattr(b, field))
                compare_states(results[0][1], results[1][1])
                checks.append(dict(kind='actual_full_sampler_raw_processed_outputs', batch=count,
                    logprobs_mode=mode, requested_logprobs=logprobs, pattern=pattern, ids_outputs_rng_equal=True))
                del results, output, sampler, metadata, values, generators
                clear()
        # Full model vocabulary: unchanged broad filter followed by new sampling storage.
        for count in (3, 16, 176):
            values = torch.randn(count, vocab, device=device, generator=input_generator).round()
            k = torch.full((count,), vocab-1, dtype=torch.int32, device=device)
            p = torch.full((count,), .9, dtype=torch.float32, device=device)
            results = []
            for name, module in modules.items():
                torch.cuda.manual_seed(86406)
                generators = gens(count, 'mixed')
                ids, _ = module.TopKTopPSampler(logprobs_mode='raw_logprobs').forward_native(values.clone(), generators, k, p)
                results.append((ids, states(generators)))
            equal(results[0][0], results[1][0])
            compare_states(results[0][1], results[1][1])
            checks.append(dict(kind='actual_broad_filter_and_noise', batch=count, ids_rng_equal=True))
            del results, ids, values, k, p, generators
            clear()
        # Warm both implementations, then alternate paired ordering to limit
        # temporal clock/allocator drift. Numerical/pressure gates are unchanged.
        for count in (16, 160, 176):
            values = torch.randn(count, vocab, device=device, generator=input_generator)
            samplers = {name:module.TopKTopPSampler(logprobs_mode='raw_logprobs') for name,module in modules.items()}
            paired_samples={name:[] for name in modules};paired_peaks={name:[] for name in modules}
            for repeat in range(8):
                order=list(samplers) if repeat%2==0 else list(reversed(samplers))
                for name in order:
                    arg=values.clone();samplers[name].forward_native(arg,{},None,None);del arg
            torch.cuda.synchronize(device)
            for repeat in range(16):
                order=list(samplers) if repeat%2==0 else list(reversed(samplers))
                for name in order:
                    arg=values.clone();torch.cuda.synchronize(device)
                    initial=torch.cuda.memory_allocated(device);torch.cuda.reset_peak_memory_stats(device)
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    wall=time.perf_counter();start.record()
                    ids,_=samplers[name].forward_native(arg,{},None,None)
                    end.record();end.synchronize()
                    paired_samples[name].append(dict(gpu_ms=start.elapsed_time(end),wall_ms=(time.perf_counter()-wall)*1000,paired_repeat=repeat,order=order))
                    paired_peaks[name].append(torch.cuda.max_memory_allocated(device)-initial)
                    del arg,ids
            for name in samplers:
                samples=paired_samples[name]
                measurements.append(dict(kind='native_sampler_latency_and_workspace',variant=name,batch=count,
                    gpu_ms_median=statistics.median(x['gpu_ms'] for x in samples),samples=samples,
                    peak_extra_allocated_samples=paired_peaks[name],timing_scope='8pairedwarmups;16alternatingpairedmeasurements,nohardwareclockchanges'))
            del samplers,values
            clear()
        # Capture/replay both real native functions and compare IDs + default RNG.
        count = 16
        values = torch.randn(count, vocab, device=device, generator=input_generator)
        captured = []
        for name, module in modules.items():
            arg = values.clone()
            sampler = module.TopKTopPSampler(logprobs_mode='raw_logprobs')
            for _ in range(3):
                sampler.forward_native(arg, {}, None, None)
                arg.copy_(values)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                ids, _ = sampler.forward_native(arg, {}, None, None)
            captured.append((name, arg, graph, ids, sampler))
        for repeat in range(3):
            results = []
            for name, arg, graph, ids, sampler in captured:
                arg.copy_(values)
                torch.cuda.manual_seed(87406 + repeat)
                graph.replay()
                torch.cuda.synchronize(device)
                results.append((ids.clone(), torch.cuda.get_rng_state(device)))
            equal(results[0][0], results[1][0])
            assert torch.equal(results[0][1], results[1][1])
            checks.append(dict(kind='actual_captured_native_noise', batch=count, repeat=repeat, ids_default_rng_equal=True))
            del results
        del captured, values, arg, graph, ids, sampler
        clear()
        assert len(checks) == 120
    assert sha(TARGET / 'vllm/v1/sample/ops/topk_topp_sampler.py') == HELPER
    assert sha(TARGET / 'vllm/v1/sample/sampler.py') == CALLER
    assert sha(TARGET / 'vllm/v1/sample/ops/topk_topp_triton.py') == checked['parent_restoration']['final_source_sha256']
    value = dict(status=f'sampler510_gpu_{args.mode}_completed', device=args.device,
        gpu_name=torch.cuda.get_device_name(device), torch_version=torch.__version__, vocab=vocab,
        script_sha256=sha(__file__), baseline_helper_sha256=HELPER, candidate_sha256=NEW,
        cpu_proof_sha256=CPU, caller_sha256=CALLER, prerequisites=checked, checks=checks,
        measurements=measurements, variant=args.variant, serving_source_mutation=False,
        whole_model_qualified=False, scope='Actual two-source sampler contracts on idle SM70. No model TPS/concurrency qualification.')
    args.output.write_text(json.dumps(value, indent=2))
    print(json.dumps(dict(status=value['status'], output=str(args.output), sha256=sha(args.output))))


if __name__ == '__main__':
    main()
