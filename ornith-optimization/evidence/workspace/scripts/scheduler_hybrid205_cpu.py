"""Actual hybrid cache coordinator CPU gates for the unchanged204 prototype."""
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
fixture_platforms._current_platform=CpuPlatform()
from tests.v1.core import utils
from vllm.v1.request import RequestStatus
from vllm.v1.kv_cache_interface import KVCacheConfig,KVCacheGroupSpec,MambaSpec

ROOT=Path('/home/xs/codex-work/ornith-opt-20261006/onecat-optimization')
OUTPUT=ROOT/'results/scheduler-hybrid205-cpu-proof.json'
assert not OUTPUT.exists() and not torch.cuda.is_initialized()
prior=ROOT/'results/scheduler-cache204-cpu-proof.json'
assert hashlib.sha256(prior.read_bytes()).hexdigest()=='961b4181e543e73e4a6fad733c58241e1d743f9782d542af3f3e2797717bafc2'
prototype_path=ROOT/'scripts/scheduler_cache204_cpu.py'
prior_value=json.loads(prior.read_text())
assert hashlib.sha256(prototype_path.read_bytes()).hexdigest()==prior_value['script_sha256']
# Load only the exact qualified prototype definition, avoiding204's main body.
tree=ast.parse(prototype_path.read_text())
definition=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='prototype')
scope={'functools':functools,'RequestStatus':RequestStatus}
exec(compile(ast.Module(body=[definition],type_ignores=[]),str(prototype_path),'exec'),scope)
prototype=scope['prototype']

def scheduler(model,budget,blocks=45):
    original=utils.Scheduler
    def construct(**kwargs):
        config=kwargs['vllm_config']
        config.cache_config.mamba_cache_mode='align'
        config.cache_config.mamba_block_size=16
        cache=kwargs['kv_cache_config']
        full=cache.kv_cache_groups[0]
        state=MambaSpec(block_size=16,shapes=((1,),),dtypes=(torch.float32,),
                        page_size_padded=full.kv_cache_spec.page_size_bytes,mamba_cache_mode='align')
        kwargs['kv_cache_config']=KVCacheConfig(num_blocks=cache.num_blocks,kv_cache_tensors=[],
            kv_cache_groups=[full,KVCacheGroupSpec(['state'],state)])
        return original(**kwargs)
    utils.Scheduler=construct
    try:return utils.create_scheduler(model=model,max_num_seqs=8,max_num_batched_tokens=budget,
        max_model_len=128,enable_prefix_caching=True,skip_tokenizer_init=True,num_blocks=blocks)
    finally:utils.Scheduler=original

def requests(count,length,label):
    return utils.create_requests(num_requests=count,num_tokens=length,block_size=16,max_tokens=8,
        req_ids=[f'{label}{index}' for index in range(count)])

def seed(s):
    old=requests(6,72,'old')
    for request in old:
        s.add_request(request)
        for _ in range(20):
            s.schedule()
            if request.num_computed_tokens>=72:break
        assert request.num_computed_tokens>=72
        s.finish_requests(request.request_id,RequestStatus.FINISHED_ABORTED)
    hot=requests(4,72,'hot')
    cold=requests(8,24,'short')[4:]
    initial=[s.kv_cache_manager.get_computed_blocks(request)[1] for request in hot]
    assert initial==[64]*4,initial
    return hot,cold,initial

def cleanup(s,requests):
    s.finish_requests([request.request_id for request in requests],RequestStatus.FINISHED_ABORTED)
    pool=s.kv_cache_manager.block_pool
    assert all(block.ref_cnt==0 for block in pool.blocks if not block.is_null)
    assert pool.get_num_free_blocks()==pool.num_gpu_blocks-1

def fixture(model,budget,protected,exception=False,cancel=False):
    s=scheduler(model,budget)
    hot,cold,initial=seed(s)
    history=prototype(s,exception) if protected else []
    if cancel:
        s.add_request(hot[0])
        s.finish_requests(hot[0].request_id,RequestStatus.FINISHED_ABORTED)
        hot=hot[1:]
    for request in cold+hot:s.add_request(request)
    injected=False
    try:
        for _ in range(30):
            s.schedule()
            if all(request.num_computed_tokens>=72 for request in hot):break
    except RuntimeError as error:
        assert exception and str(error)=='injected_cpu_allocation_exception204'
        injected=True
    cached=[request.prefill_stats.num_cached_tokens if request.prefill_stats is not None else None for request in hot]
    complete=all(request.num_computed_tokens>=72 for request in hot)
    cleanup(s,cold+hot)
    if protected and not exception:
        assert complete and cached==[64]*len(hot),(budget,cached,complete)
    if exception:assert injected
    return dict(budget=budget,protected=protected,coordinator=type(s.kv_cache_manager.coordinator).__name__,
        cache_mode='align',groups=['fullattention','mamba'],initial_cached=initial,admission_cached=cached,
        hot_prefills_completed=complete,cancelled_waiting=cancel,injected_exception=injected,
        reservations=history,all_refs_released=True,free_count_restored=44)

config=dict(architectures=['OPTForCausalLM'],model_type='opt',hidden_size=64,ffn_dim=256,
    num_attention_heads=4,num_hidden_layers=2,vocab_size=128,max_position_embeddings=128,eos_token_id=2)
checks=[]
with tempfile.TemporaryDirectory(prefix='codex-hybrid205-') as folder:
    (Path(folder)/'config.json').write_text(json.dumps(config))
    for budget in [32,128,256]:
        before=fixture(folder,budget,False)
        after=fixture(folder,budget,True)
        checks.append(dict(baseline=before,prototype=after))
    cancellation=fixture(folder,128,True,cancel=True)
    exception=fixture(folder,128,True,exception=True)
assert not torch.cuda.is_initialized()
result=dict(status='scheduler_hybrid205_cpu_gates_passed',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
    checks=checks,cancellation=cancellation,exception_cleanup=exception,
    prototype_source=str(prototype_path),prototype_sha256=prior_value['script_sha256'],prior_proof_sha256=hashlib.sha256(prior.read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),cuda_initialized=False,model_requests_sent=0,source_mutation_performed=False,
    scope='RealScheduler andjointFullAttention/Mamba align metadata coordinator onofficialCPUplatform. Smallstate shapes; no GPU state tensor execution, Ornithimages/numerics orTPS. Exact204 prototype loaded frompinnedAST.',
    required_next_gates=['Preemption/constrained-admission/progress','Isolated sourcecandidate review/hooks afterlive187/189/200 terminate','WholeOrnith model performance/capacity/quality qualification'])
OUTPUT.write_text(json.dumps(result,indent=2))
print(json.dumps(dict(status=result['status'],output=str(OUTPUT),sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),checks=len(checks),cuda_initialized=False)),flush=True)
