"""Instance-only admission prototype on real Scheduler; no serving source mutation."""
import functools
import hashlib
import json
from pathlib import Path
import tempfile
import time

import torch
import vllm.platforms as fixture_platforms
from vllm.platforms.cpu import CpuPlatform

# Process-local official CPU platform prevents CUDA's NVML device mapping from
# interpreting CUDA_VISIBLE_DEVICES=-1 while loading scheduler test utilities.
# This changes only the isolated fixture platform, not any serving process.
fixture_platforms._current_platform=CpuPlatform()
from tests.v1.core.utils import create_requests,create_scheduler
from vllm.v1.request import RequestStatus

ROOT=Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
OUTPUT=ROOT/'results/scheduler-cache204-cpu-proof.json'
assert not OUTPUT.exists() and not torch.cuda.is_initialized()

def prototype(scheduler,inject_exception=False):
    manager=scheduler.kv_cache_manager
    pool=manager.block_pool
    original_schedule=scheduler.schedule
    original_allocate=manager.allocate_slots
    held=[]
    protected=False
    allocations=0
    history=[]

    def allocate(request,*args,**kwargs):
        nonlocal protected,allocations
        # Running-token allocation already precedes this stage. Do not reserve
        # waiting caches before decode allocation or change FIFO admission.
        if request.status==RequestStatus.WAITING and not protected:
            protected=True
            seen=set()
            capacity=max(0,scheduler.max_num_running_reqs-len(scheduler.running))
            candidates=list(scheduler.waiting)+list(scheduler.skipped_waiting)
            old_boundary=manager.coordinator.shared_prefix_boundary
            try:
                for pending in candidates[:capacity]:
                    if pending.status!=RequestStatus.WAITING or pending.num_computed_tokens or pending.skip_reading_prefix_cache:
                        continue
                    blocks,_=manager.coordinator.find_longest_cache_hit(pending.block_hashes,pending.num_tokens-1)
                    for group in blocks:
                        for block in group:
                            if block.is_null or block.block_id in seen:continue
                            seen.add(block.block_id)
                            held.append(block)
                pool.touch(held)
            finally:
                manager.coordinator.shared_prefix_boundary=old_boundary
            history.append(dict(reserved_blocks=len(held),running=len(scheduler.running),candidate_count=min(capacity,len(candidates))))
        if request.status==RequestStatus.WAITING:
            allocations+=1
            if inject_exception and allocations==2:raise RuntimeError('injected_cpu_allocation_exception204')
        return original_allocate(request,*args,**kwargs)

    @functools.wraps(original_schedule)
    def schedule():
        nonlocal protected,allocations
        assert not held
        protected=False
        allocations=0
        manager.allocate_slots=allocate
        try:return original_schedule()
        finally:
            manager.allocate_slots=original_allocate
            if held:pool.free_blocks(held)
            held.clear()
    scheduler.schedule=schedule
    return history

def fixture(model,protected,budget,inject=False):
    scheduler=create_scheduler(model=model,max_num_seqs=8,max_num_batched_tokens=budget,
        max_model_len=128,enable_prefix_caching=True,skip_tokenizer_init=True,num_blocks=25)
    old=create_requests(num_requests=6,num_tokens=64,block_size=16,max_tokens=8,req_ids=[f'old{i}' for i in range(6)])
    for request in old:
        scheduler.add_request(request)
        for _ in range(10):
            scheduler.schedule()
            if request.num_computed_tokens>=64:break
        assert request.num_computed_tokens>=64
        scheduler.finish_requests(request.request_id,RequestStatus.FINISHED_ABORTED)
    hot=create_requests(num_requests=4,num_tokens=64,block_size=16,max_tokens=8,req_ids=[f'hot{i}' for i in range(4)])
    cold=create_requests(num_requests=8,num_tokens=32,block_size=16,max_tokens=8,req_ids=[f'short{i}' for i in range(8)])[4:]
    initial=[scheduler.kv_cache_manager.get_computed_blocks(request)[1] for request in hot]
    assert initial==[48]*4
    history=prototype(scheduler,inject) if protected else []
    # FIFO cold arrivals precede cached arrivals; every request is visible in
    # the same admission cohort. This is not an HTTP/GPU chronology replay.
    for request in cold+hot:scheduler.add_request(request)
    exception=False
    try:
        for _ in range(20):
            scheduler.schedule()
            if all(request.prefill_stats is not None for request in hot) and all(request.num_computed_tokens>=64 for request in hot):break
    except RuntimeError as error:
        assert inject and str(error)=='injected_cpu_allocation_exception204'
        exception=True
    cached=[request.prefill_stats.num_cached_tokens if request.prefill_stats is not None else None for request in hot]
    scheduled_all=all(request.num_computed_tokens>=64 for request in hot)
    scheduler.finish_requests([request.request_id for request in cold+hot],RequestStatus.FINISHED_ABORTED)
    pool=scheduler.kv_cache_manager.block_pool
    assert all(block.ref_cnt==0 for block in pool.blocks if not block.is_null)
    assert pool.get_num_free_blocks()==24
    return dict(protected=protected,budget=budget,initial_cached_tokens=initial,cached_tokens_at_admission=cached,
        all_hot_prefills_scheduled=scheduled_all,exception_injected=exception,reservations=history,
        all_refs_released=True,free_pool_restored=24)

config=dict(architectures=['OPTForCausalLM'],model_type='opt',hidden_size=64,ffn_dim=256,
    num_attention_heads=4,num_hidden_layers=2,vocab_size=128,max_position_embeddings=128,eos_token_id=2)
checks=[]
with tempfile.TemporaryDirectory(prefix='codex-cache204-') as folder:
    (Path(folder)/'config.json').write_text(json.dumps(config))
    for budget in [32,128,256]:
        before=fixture(folder,False,budget)
        after=fixture(folder,True,budget)
        assert after['all_hot_prefills_scheduled'] and after['cached_tokens_at_admission']==[48]*4
        checks.append(dict(baseline=before,prototype=after))
    exception=fixture(folder,True,128,True)
    assert exception['exception_injected']
assert not torch.cuda.is_initialized()
sources=[]
for module in ['vllm.v1.core.sched.scheduler','vllm.v1.core.kv_cache_manager','vllm.v1.core.block_pool']:
    source=Path(__import__(module,fromlist=['x']).__file__)
    assert source.is_relative_to(ROOT/'src/onecat-padded-kv140')
    sources.append(dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
result=dict(status='scheduler_cache204_instance_prototype_cpu_passed',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    checks=checks,exception_cleanup=exception,sources=sources,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    cuda_initialized=False,model_requests_sent=0,source_mutation_performed=False,
    scope='RealScheduler/CacheManager/BlockPool withofficialCPUplatform andsmallOPT fullattention fixtures; instance-only temporaryref protection atwaitingallocation. Not Ornithhybrid/MM/numeric/GPU/performance qualification, not exact failed24 chronology.',
    required_next_gates=['HybridMamba coordinator cases','Preemption/cancellation and constrained-admission cases','Source review/hooks inisolated candidate afterallliveGPUjobs','Full model qualification and performance comparison'])
OUTPUT.write_text(json.dumps(result,indent=2))
print(json.dumps(dict(status=result['status'],output=str(OUTPUT),sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),checks=len(checks),cuda_initialized=False)),flush=True)
