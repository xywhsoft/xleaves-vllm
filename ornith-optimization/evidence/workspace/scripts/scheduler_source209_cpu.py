"""Run actual isolated Scheduler source candidate, no manager monkeypatch guard."""
import ast
import functools
import hashlib
import importlib.util
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

ROOT = Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
OUTPUT = ROOT / 'results/scheduler-source209-cpu-proof.json'
assert not OUTPUT.exists() and not torch.cuda.is_initialized()
candidate = ROOT / 'scripts/scheduler_cache209_candidate.py'
assert hashlib.sha256(candidate.read_bytes()).hexdigest() == '0d0ac21791373bfb55deeaa75622d5eb7df4efea5848da04867a061f1af44632'
spec = importlib.util.spec_from_file_location('vllm.v1.core.sched.scheduler_cache207', candidate)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
OriginalGuard = module._WaitingPrefixCacheGuard
OriginalScheduler = utils.Scheduler


class TraceGuard(OriginalGuard):
    def __init__(self, scheduler):
        super().__init__(scheduler)
        if not hasattr(scheduler, '_trace207'):
            scheduler._trace207 = []
        self.history = scheduler._trace207
        self.in_allocate = False
    def protect(self):
        super().protect()
        self.history.append(dict(reserved_blocks=len(self.held), running=len(self.scheduler.running)))
    def release(self):
        if self.in_allocate and self.held:
            self.history.append(dict(pressure_retry=True, released_blocks=len(self.held)))
        super().release()
    def allocate_slots(self, request, *args, **kwargs):
        self.in_allocate = True
        try:
            return super().allocate_slots(request, *args, **kwargs)
        finally:
            self.in_allocate = False


module._WaitingPrefixCacheGuard = TraceGuard
utils.Scheduler = module.Scheduler


def prototype(scheduler, inject_exception=False):
    scheduler._trace207 = []
    if inject_exception:
        original = scheduler.kv_cache_manager.allocate_slots
        allocations = 0
        def allocate(request, *args, **kwargs):
            nonlocal allocations
            if request.status == RequestStatus.WAITING:
                allocations += 1
                if allocations == 2:
                    raise RuntimeError('injected_cpu_allocation_exception204')
            return original(request, *args, **kwargs)
        scheduler.kv_cache_manager.allocate_slots = allocate
    return scheduler._trace207


def load_definitions(filename, names):
    source = ROOT / 'scripts' / filename
    proofs = {'scheduler_cache204_cpu.py': 'scheduler-cache204-cpu-proof.json',
              'scheduler_hybrid205_cpu.py': 'scheduler-hybrid205-cpu-proof.json',
              'scheduler_progress206_cpu.py': 'scheduler-progress206-cpu-proof.json'}
    proof = json.loads((ROOT / 'results' / proofs[filename]).read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == proof['script_sha256']
    tree = ast.parse(source.read_text())
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(nodes) == len(names)
    scope = dict(globals())
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
    return scope


full = load_definitions('scheduler_cache204_cpu.py', ['fixture'])
old_prototype = load_definitions('scheduler_cache204_cpu.py', ['prototype'])['prototype']
hybrid = load_definitions('scheduler_hybrid205_cpu.py', ['scheduler', 'requests', 'seed', 'cleanup', 'fixture'])
extended = load_definitions('scheduler_progress206_cpu.py', ['pressure', 'preemption', 'metrics_exception'])
config = dict(architectures=['OPTForCausalLM'], model_type='opt', hidden_size=64, ffn_dim=256,
    num_attention_heads=4, num_hidden_layers=2, vocab_size=128, max_position_embeddings=128, eos_token_id=2)
checks = []
with tempfile.TemporaryDirectory(prefix='codex-source207-') as folder:
    (Path(folder) / 'config.json').write_text(json.dumps(config))
    utils.Scheduler = OriginalScheduler
    pressure_old = extended['pressure'](folder, True)
    utils.Scheduler = module.Scheduler
    pressure_new = extended['pressure'](folder, False)
    for budget in [32, 128, 256]:
        f = full['fixture'](folder, True, budget)
        assert f['all_hot_prefills_scheduled'] and f['cached_tokens_at_admission'] == [48] * 4
        h = hybrid['fixture'](folder, budget, True)
        checks.append(dict(full=f, hybrid=h))
    cancellation = hybrid['fixture'](folder, 128, True, cancel=True)
    exception = hybrid['fixture'](folder, 128, True, exception=True)
    preempted = extended['preemption'](folder)
    touch_error = extended['metrics_exception'](folder)
utils.Scheduler = OriginalScheduler
assert not torch.cuda.is_initialized()
serving = ROOT / 'src/onecat-padded-kv140/vllm/v1/core/sched/scheduler.py'
assert hashlib.sha256(serving.read_bytes()).hexdigest() == '441af671c70dd0764a0012630e05227426baf5b92ffece33a3c3b0108a2cc8cf'
result = dict(status='scheduler_source209_cpu_gates_passed', utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    checks=checks, pressure_old=pressure_old, pressure_new=pressure_new, cancellation=cancellation,
    exception_cleanup=exception, preemption=preempted, partial_touch_exception=touch_error,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
    serving_base_sha256=hashlib.sha256(serving.read_bytes()).hexdigest(),
    cuda_initialized=False, model_requests_sent=0, serving_source_mutation=False,
    scope='Actual isolated Scheduler source with direct waiting-stage guard, joint FullAttention/Mamba CPU metadata. No manager monkeypatch for guard; trace subclass and allocation-error injection are fixture-only. No Ornith GPU/MM/numerics/TPS qualification.')
OUTPUT.write_text(json.dumps(result, indent=2))
print(json.dumps(dict(status=result['status'], output=str(OUTPUT), sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest())), flush=True)
