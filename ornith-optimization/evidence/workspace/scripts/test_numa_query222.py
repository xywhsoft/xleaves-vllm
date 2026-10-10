"""Real CPU-only positive mixed-file and negative remote-anonymous fixtures."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from query_numa_anon221 import ROOT, PAGE
from qualified_profile189 import sha

output = ROOT / 'results/numa222-query-cpu-proof.json'
fixture = ROOT / 'cache/numa222-file-cache.bin'
assert not output.exists() and not fixture.exists()
producer = "from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b'A'*(16*1024**2))"
subprocess.run(['numactl', '--cpunodebind=0', '--membind=0', sys.executable, '-c', producer, str(fixture)], check=True)
child_code = r'''
import ctypes,json,mmap,os,sys
from pathlib import Path
from query_numa_anon221 import LIB,PAGE,inspect,private_anonymous_addresses,process,query_nodes
stream=open(sys.argv[1],'r+b')
filemap=mmap.mmap(stream.fileno(),0,flags=mmap.MAP_PRIVATE,prot=mmap.PROT_READ|mmap.PROT_WRITE)
for offset in range(0,len(filemap),PAGE):unused=filemap[offset]
filemap[0]=66
address=ctypes.addressof(ctypes.c_char.from_buffer(filemap))
file_nodes=query_nodes(os.getpid(),[address+i*PAGE for i in range(len(filemap)//PAGE)])
assert file_nodes[0]==1 and any(node==0 for node in file_nodes[1:])
private=private_anonymous_addresses(os.getpid(),address,address+len(filemap))
assert private==[address] and query_nodes(os.getpid(),private)==[1]
pool=mmap.mmap(-1,16*1024**2,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS)
for offset in range(0,len(pool),PAGE):pool[offset]=77
negative=sys.argv[2]=='negative'
if negative:
    LIB.numa_allocate_nodemask.restype=ctypes.c_void_p
    LIB.numa_bitmask_setbit.argtypes=[ctypes.c_void_p,ctypes.c_uint]
    LIB.numa_bitmask_setbit.restype=ctypes.c_void_p
    LIB.numa_set_membind.argtypes=[ctypes.c_void_p]
    LIB.numa_set_membind.restype=None
    zero=LIB.numa_bitmask_setbit(LIB.numa_allocate_nodemask(),0)
    one=LIB.numa_bitmask_setbit(LIB.numa_allocate_nodemask(),1)
    LIB.numa_set_membind(zero)
    remote=mmap.mmap(-1,16*1024**2,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS)
    for offset in range(0,len(remote),PAGE):remote[offset]=88
    remote_address=ctypes.addressof(ctypes.c_char.from_buffer(remote))
    assert set(query_nodes(os.getpid(),[remote_address+i*PAGE for i in range(len(remote)//PAGE)]))=={0}
    LIB.numa_set_membind(one)
result=inspect(os.getpid(),1,process(os.getpid())[1])
if negative:assert not result['passed'] and result['remote_pages_in_bound_anon']>=4096
else:assert result['passed'] and result['remote_pages_in_bound_anon']==0
print(json.dumps(dict(negative=negative,passed=result['passed'],local_anon=result['bound_anon_pages'],remote_anon=result['remote_pages_in_bound_anon'],ambiguous=result['ambiguous_anonymous_pages'],mixed_file_page_nodes=dict(__import__('collections').Counter(file_nodes)),private_cow_nodes=[1],reader_result={k:v for k,v in result.items() if k!='mappings'})))
'''
results = []
try:
    for mode in ['positive', 'negative']:
        child = subprocess.run(['numactl', '--cpunodebind=1', '--membind=1', sys.executable, '-c', child_code, str(fixture), mode],
            cwd=ROOT / 'scripts', env={**os.environ, 'CUDA_VISIBLE_DEVICES': '-1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn'}, capture_output=True, text=True, check=True, timeout=30)
        results.append(json.loads(child.stdout))
finally:
    fixture.unlink()
result = dict(status='numa222_cpu_mixed_mapping_and_real_remote_negative_passed', utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    script_sha256=sha(Path(__file__).resolve()), reader_sha256=sha(ROOT / 'scripts/query_numa_anon221.py'), fixtures=results,
    model_requests_sent=0, serving_source_mutation=False, live_worker_mutation=False, gpu_work=False,
    scope='Isolated CPUchildren: cached file pages onnode0 + privateCOW/pool onnode1 accepted; actual16MiB anonymous allocation onnode0 followedby policybind1 rejected. Queries nodes=NULL/flags0. Policy changes confinedtoCPUnegativefixture; originalmodel workers unmodified.')
output.write_text(json.dumps(result, indent=2))
print(json.dumps(dict(status=result['status'], output=str(output), sha256=sha(output), fixtures=results)), flush=True)
