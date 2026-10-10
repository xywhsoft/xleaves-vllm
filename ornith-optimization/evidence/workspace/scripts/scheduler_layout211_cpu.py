"""Observed Ornith cache metadata, actual Scheduler, no tensor/model execution."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import tempfile
import time
import torch
import vllm.platforms as fixture_platforms
from vllm.platforms.cpu import CpuPlatform
fixture_platforms._current_platform=CpuPlatform()
from tests.v1.core import utils
from vllm.v1.request import RequestStatus
from vllm.v1.kv_cache_interface import KVCacheConfig,KVCacheGroupSpec,MambaSpec,TQFullAttentionSpec

os.nice(10)
ROOT=Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
OUTPUT=ROOT/'results/scheduler-layout211-cpu-proof.json'
assert not OUTPUT.exists() and not torch.cuda.is_initialized()
layout_path=ROOT/'scripts/cache211-layout149.json'
assert hashlib.sha256(layout_path.read_bytes()).hexdigest()=='faf33d28aae4b23f2a64138199db2d995396f8be71edc912030bfdb3d4343c3a'
layout=json.loads(layout_path.read_text())
candidate=ROOT/'scripts/scheduler_cache209_candidate.py'
assert hashlib.sha256(candidate.read_bytes()).hexdigest()=='0d0ac21791373bfb55deeaa75622d5eb7df4efea5848da04867a061f1af44632'
spec=importlib.util.spec_from_file_location('vllm.v1.core.sched.scheduler_layout211',candidate)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
OriginalScheduler=utils.Scheduler
groups=[]
for group in layout['before_initialization']['fields']['kv_cache_groups']:
    info=group['fields']
    raw=info['kv_cache_spec']['fields']
    if info['kv_cache_spec']['type']=='MambaSpec':
        value=MambaSpec(block_size=raw['block_size'],dcp_sharded=raw['dcp_sharded'],
            shapes=tuple(tuple(shape) for shape in raw['shapes']),
            dtypes=tuple(getattr(torch,dtype.removeprefix('torch.')) for dtype in raw['dtypes']),
            page_size_padded=raw['page_size_padded'],mamba_cache_mode=raw['mamba_cache_mode'],
            num_speculative_blocks=raw['num_speculative_blocks'],tp_replicated=raw['tp_replicated'])
    else:
        assert info['kv_cache_spec']['type']=='TQFullAttentionSpec'
        arguments={**raw,'dtype':getattr(torch,raw['dtype'].removeprefix('torch.'))}
        quant_mode=arguments.pop('kv_quant_mode')
        value=TQFullAttentionSpec(**arguments)
        assert value.kv_quant_mode.value==quant_mode
    assert value.page_size_bytes==1076288
    groups.append(KVCacheGroupSpec(info['layer_names'],value))
assert len(groups)==4 and [type(g.kv_cache_spec).__name__ for g in groups]==['MambaSpec']*3+['TQFullAttentionSpec']
blocks=(16*1024**3)//(1076288*10)


def scheduler(model,protected,budget=4096):
    def construct(**kwargs):
        config=kwargs['vllm_config']
        config.cache_config.mamba_cache_mode='align'
        config.cache_config.mamba_block_size=4016
        kwargs['kv_cache_config']=KVCacheConfig(num_blocks=blocks,kv_cache_tensors=[],kv_cache_groups=groups)
        return (module.Scheduler if protected else OriginalScheduler)(**kwargs)
    utils.Scheduler=construct
    try:return utils.create_scheduler(model=model,max_num_seqs=32,max_num_batched_tokens=budget,
        max_model_len=240128,enable_prefix_caching=True,skip_tokenizer_init=True,num_blocks=blocks,block_size=4016)
    finally:utils.Scheduler=OriginalScheduler


def requests(count,length,label):
    return utils.create_requests(num_requests=count,num_tokens=length,block_size=4016,max_tokens=1024,
        req_ids=[f'{label}{i}' for i in range(count)])


def step(s):
    output=s.schedule()
    # Isolated metadata fixture acknowledgement of completed scheduled work.
    # No actual GPU forward/output execution occurs in this test.
    for request in s.requests.values():request.num_in_flight_tokens=0
    return output


def cleanup(s,items):
    s.finish_requests([r.request_id for r in items],RequestStatus.FINISHED_ABORTED)
    pool=s.kv_cache_manager.block_pool
    assert all(b.ref_cnt==0 for b in pool.blocks if not b.is_null)
    assert pool.get_num_free_blocks()==blocks-1


def fixture(model,protected):
    s=scheduler(model,protected)
    old=requests(24,239104,'old')
    for request in old:
        s.add_request(request)
        for _ in range(100):
            step(s)
            if request.num_computed_tokens>=239104:break
        assert request.num_computed_tokens>=239104,(request.request_id,request.num_computed_tokens)
        s.finish_requests(request.request_id,RequestStatus.FINISHED_ABORTED)
    hot=requests(16,239104,'hot')
    cold=requests(32,512,'short')[16:]
    initial=[s.kv_cache_manager.get_computed_blocks(r)[1] for r in hot]
    assert initial==[236944]*16,initial
    for request in cold+hot:s.add_request(request)
    schedule_ms=[]
    for _ in range(100):
        started=time.perf_counter()
        out=step(s)
        schedule_ms.append((time.perf_counter()-started)*1000)
        if all(r.num_computed_tokens>=239104 for r in hot):break
        if not out.num_scheduled_tokens:break
    cached=[r.prefill_stats.num_cached_tokens if r.prefill_stats else None for r in hot]
    completed=[r.num_computed_tokens>=239104 for r in hot]
    cleanup(s,cold+hot)
    if protected:assert cached==[236944]*16 and all(completed),(cached,completed)
    # Benchmark after the admission gate. Guard touches change LRU order,
    # so timing probes must not perturb the baseline's eviction test.
    probes=requests(16,239104,'probe')
    snapshot=s.kv_cache_manager.coordinator.shared_prefix_boundary
    times=[]
    reserved=[]
    for _ in range(21):
        started=time.perf_counter()
        with module._WaitingPrefixCacheGuard(s) as guard:
            for request in probes:s.waiting.add_request(request)
            guard.protect()
            reserved.append(len(guard.held))
            for request in probes:s.waiting.remove_request(request)
        times.append((time.perf_counter()-started)*1000)
        assert s.kv_cache_manager.coordinator.shared_prefix_boundary==snapshot
    pool=s.kv_cache_manager.block_pool
    assert all(b.ref_cnt==0 for b in pool.blocks if not b.is_null)
    assert pool.get_num_free_blocks()==blocks-1
    return dict(protected=protected,budget=4096,pool_blocks=blocks,groups=4,mamba_groups=3,cache_block_tokens=4016,
        independent_contexts_seeded=24,input_tokens=239104,initial_cached=initial,admission_cached=cached,hot_prefills_completed=completed,
        guard_ms=dict(median=statistics.median(times[1:]),minimum=min(times[1:]),maximum=max(times[1:]),samples=times),
        guard_reserved_blocks=reserved,schedule_ms=schedule_ms,all_refs_released=True,free_count_restored=blocks-1)


config=dict(architectures=['OPTForCausalLM'],model_type='opt',hidden_size=64,ffn_dim=256,
    num_attention_heads=4,num_hidden_layers=2,vocab_size=128,max_position_embeddings=240128,eos_token_id=2)
with tempfile.TemporaryDirectory(prefix='codex-layout211-') as folder:
    (Path(folder)/'config.json').write_text(json.dumps(config))
    baseline=fixture(folder,False)
    protected=fixture(folder,True)
assert not torch.cuda.is_initialized()
result=dict(status='scheduler_layout211_cpu_gates_passed',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    baseline=baseline,protected=protected,layout_source=str(layout_path),layout_sha256=hashlib.sha256(layout_path.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
    cuda_initialized=False,model_requests_sent=0,serving_source_mutation=False,
    scope='Observed historical149 Ornith group/state/page metadata reconstructed on actual Scheduler with officialCPUplatform and localOPT config. Full239104 inputs but no GPU tensors or model outputs; explicit metadata completion acknowledgement. Guard timing is CPU fixture overhead, not TPS or current GPU thermal attribution.')
OUTPUT.write_text(json.dumps(result,indent=2))
print(json.dumps(dict(status=result['status'],output=str(OUTPUT),sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),guard_median_ms=protected['guard_ms']['median'],cached=protected['admission_cached'])),flush=True)
