"""Constrained admission, hybrid preemption and exception gates, no GPU."""
import ast
import functools
import hashlib
import json
from pathlib import Path
import tempfile
import time
import torch
import vllm.platforms as fixture_platforms
from vllm.platforms.cpu import CpuPlatform
fixture_platforms._current_platform = CpuPlatform()
from tests.v1.core import utils
from tests.v1.core.utils import create_requests, create_scheduler
from vllm.v1.request import RequestStatus
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec, MambaSpec
from candidate_cache_guard206 import prototype

ROOT = Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
OUTPUT = ROOT / 'results/scheduler-progress206-cpu-proof.json'
assert not OUTPUT.exists() and not torch.cuda.is_initialized()


def load_definitions(filename, names):
    source = ROOT / 'scripts' / filename
    proofname = 'scheduler-cache204-cpu-proof.json' if '204' in filename else 'scheduler-hybrid205-cpu-proof.json'
    proof = json.loads((ROOT / 'results' / proofname).read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == proof['script_sha256']
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    scope = dict(globals())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
    return scope


full = load_definitions('scheduler_cache204_cpu.py', ['prototype', 'fixture'])
old_prototype = full['prototype']
full['prototype'] = prototype
hybrid = load_definitions('scheduler_hybrid205_cpu.py', ['scheduler', 'requests', 'seed', 'cleanup', 'fixture'])


def pressure(model, old):
    s = create_scheduler(model=model, max_num_seqs=8, max_num_batched_tokens=128,
        max_model_len=128, enable_prefix_caching=True, skip_tokenizer_init=True, num_blocks=25)
    seed = create_requests(num_requests=6, num_tokens=64, block_size=16, max_tokens=8,
                           req_ids=[f'seed{i}' for i in range(6)])
    for request in seed:
        s.add_request(request)
        s.schedule()
        assert request.num_computed_tokens >= 64
        s.finish_requests(request.request_id, RequestStatus.FINISHED_ABORTED)
    hot = create_requests(num_requests=6, num_tokens=64, block_size=16, max_tokens=8,
                          req_ids=[f'hot{i}' for i in range(6)])
    cold = create_requests(num_requests=9, num_tokens=112, block_size=16, max_tokens=8,
                           req_ids=[f'cold{i}' for i in range(9)])[-1]
    initial = [s.kv_cache_manager.get_computed_blocks(r)[1] for r in hot]
    assert initial == [48] * 6, initial
    history = (old_prototype if old else prototype)(s)
    for request in [cold] + hot:
        s.add_request(request)
    steps = []
    for _ in range(2):
        output = s.schedule()
        steps.append(dict(scheduled=sum(output.num_scheduled_tokens.values()),
                          running=len(s.running), cold_computed=cold.num_computed_tokens))
    if old:
        assert all(step['scheduled'] == 0 and step['running'] == 0 for step in steps), steps
    else:
        assert steps[0]['scheduled'] > 0 and cold.num_computed_tokens >= 112, steps
        assert any(item.get('pressure_retry') for item in history)
    hybrid['cleanup'](s, [cold] + hot)
    return dict(old_prototype=old, initial_cached=initial, steps=steps, reservations=history,
                all_refs_released=True, free_count_restored=24)


def preemption(model):
    s = hybrid['scheduler'](model, 128)
    hot, cold, initial = hybrid['seed'](s)
    history = prototype(s)
    for request in hot:
        s.add_request(request)
    request = hot[0]
    for _ in range(20):
        s.schedule()
        if request.num_computed_tokens >= 72:
            break
    assert request.num_computed_tokens >= 72 and request.status == RequestStatus.RUNNING, (request.num_computed_tokens, request.status)
    initial_stat = request.prefill_stats.num_cached_tokens
    s.running.remove(request)
    s._preempt_request(request, time.monotonic())
    s.prev_step_scheduled_req_ids.discard(request.request_id)
    for _ in range(20):
        s.schedule()
        if request.num_computed_tokens >= 72:
            break
    assert request.num_preemptions == 1 and request.num_computed_tokens >= 72
    assert initial_stat == 64 and request.prefill_stats.num_cached_tokens == initial_stat
    hybrid['cleanup'](s, hot)
    return dict(preemptions=1, resumed_computed=request.num_computed_tokens,
                cached_stat_preserved=initial_stat, reservations=history,
                all_refs_released=True, free_count_restored=44)


def metrics_exception(model):
    s = hybrid['scheduler'](model, 128)
    hot, cold, initial = hybrid['seed'](s)
    history = prototype(s)
    pool = s.kv_cache_manager.block_pool
    original_metrics = pool.metrics_collector
    class Collector:
        accesses = 0
        def on_block_accessed(self, *args, **kwargs):
            self.accesses += 1
            if self.accesses == 2:
                raise RuntimeError('injected_touch_metrics206')
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    collector = Collector()
    for request in cold + hot:
        s.add_request(request)
    pool.metrics_collector = collector
    try:
        s.schedule()
    except RuntimeError as error:
        assert str(error) == 'injected_touch_metrics206'
    else:
        raise AssertionError('callback failure was not injected')
    finally:
        pool.metrics_collector = original_metrics
    assert collector.accesses == 2
    hybrid['cleanup'](s, cold + hot)
    return dict(partial_touch_exception=True, all_refs_released=True, free_count_restored=44)


config = dict(architectures=['OPTForCausalLM'], model_type='opt', hidden_size=64, ffn_dim=256,
    num_attention_heads=4, num_hidden_layers=2, vocab_size=128, max_position_embeddings=128, eos_token_id=2)
checks = []
with tempfile.TemporaryDirectory(prefix='codex-progress206-') as folder:
    (Path(folder) / 'config.json').write_text(json.dumps(config))
    pressure_old = pressure(folder, True)
    pressure_new = pressure(folder, False)
    for budget in [32, 128, 256]:
        f = full['fixture'](folder, True, budget)
        assert f['all_hot_prefills_scheduled'] and f['cached_tokens_at_admission'] == [48] * 4
        h = hybrid['fixture'](folder, budget, True)
        checks.append(dict(full=f, hybrid=h))
    cancellation = hybrid['fixture'](folder, 128, True, cancel=True)
    exception = hybrid['fixture'](folder, 128, True, exception=True)
    preempted = preemption(folder)
    touch_error = metrics_exception(folder)
assert not torch.cuda.is_initialized()
candidate = ROOT / 'scripts/candidate_cache_guard206.py'
result = dict(status='scheduler_progress206_cpu_gates_passed', utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    pressure_old=pressure_old, pressure_new=pressure_new, checks=checks, cancellation=cancellation,
    exception_cleanup=exception, preemption=preempted, partial_touch_exception=touch_error,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
    cuda_initialized=False, model_requests_sent=0, serving_source_mutation=False,
    scope='Actual small fullattention/hybrid Scheduler CPU metadata fixtures. Pressure fallback permits cache loss when capacity is insufficient. No Ornith GPU/numeric/MM/TPS qualification.')
OUTPUT.write_text(json.dumps(result, indent=2))
print(json.dumps(dict(status=result['status'], output=str(OUTPUT), sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest())), flush=True)
