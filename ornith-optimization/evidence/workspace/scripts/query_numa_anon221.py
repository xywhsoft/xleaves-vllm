"""Read-only NUMA page queries: mixed file mappings must not count all N pages as anon."""
from array import array
from collections import Counter
import ctypes
import hashlib
import json
import os
from pathlib import Path
import time
from pipeline_numa189 import ROOT, process

PAGE = os.sysconf('SC_PAGE_SIZE')
assert PAGE == 4096
LIB = ctypes.CDLL('libnuma.so.1', use_errno=True)
MOVE = LIB.numa_move_pages
MOVE.restype = ctypes.c_int
MOVE.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.c_int]


def query_nodes(pid, addresses):
    result = []
    for start in range(0, len(addresses), 1024):
        batch = addresses[start:start + 1024]
        pointers = (ctypes.c_void_p * len(batch))(*batch)
        statuses = (ctypes.c_int * len(batch))(*([-9999] * len(batch)))
        # nodes=NULL and flags=0: status query only, never migration.
        status = MOVE(pid, len(batch), pointers, None, statuses, 0)
        if status < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        result.extend(statuses)
    return result


def private_anonymous_addresses(pid, begin, end):
    addresses = []
    with open(f'/proc/{pid}/pagemap', 'rb', buffering=0) as stream:
        for start in range(begin, end, PAGE * 65536):
            stop = min(end, start + PAGE * 65536)
            count = (stop - start) // PAGE
            stream.seek(start // PAGE * 8)
            raw = stream.read(count * 8)
            assert len(raw) == count * 8
            entries = array('Q')
            entries.frombytes(raw)
            for index, entry in enumerate(entries):
                # bit63 present; bit61 file-page or shared-anon.
                if entry & (1 << 63) and not entry & (1 << 61):
                    addresses.append(start + index * PAGE)
    return addresses


def inspect(pid, rank, birth):
    started = time.perf_counter()
    state, actual_birth = process(pid)
    assert actual_birth == birth and state not in ['missing', 'Z', 'X']
    folder = Path(f'/proc/{pid}')
    status = {x.split(':', 1)[0]: x.split(':', 1)[1].strip() for x in (folder / 'status').read_text().splitlines() if ':' in x}
    expected = Path(f'/sys/devices/system/node/node{rank}/cpulist').read_text().strip()
    # Persist only the worker start method from environ, never credentials.
    env = dict(x.split('=', 1) for x in (folder / 'environ').read_bytes().decode().split('\0') if '=' in x)
    method = env.get('VLLM_WORKER_MULTIPROC_METHOD')
    maps = {int(x.split()[0].split('-')[0], 16): x for x in (folder / 'maps').read_text().splitlines()}
    selected, policies = [], set()
    all_anon, local_anon, remote_anon, ambiguous = 0, 0, 0, 0
    for line in (folder / 'numa_maps').read_text().splitlines():
        parts = line.split()
        values = dict(x.split('=', 1) for x in parts[2:] if '=' in x)
        anon = int(values.get('anon', '0'))
        if not anon:
            continue
        policy = parts[1]
        policies.add(policy)
        all_anon += anon
        if policy != f'bind:{rank}':
            continue
        nodes = {int(k[1:]): int(v) for k, v in values.items() if k.startswith('N') and k[1:].isdigit()}
        remote_all_pages = sum(v for node, v in nodes.items() if node != rank)
        if not remote_all_pages:
            local_anon += anon
            selected.append(dict(address=parts[0], anon=anon, local_anon=anon, remote_anon=0, method='all_mapped_pages_local'))
            continue
        begin = int(parts[0], 16)
        mapping = maps[begin]
        end = int(mapping.split()[0].split('-')[1], 16)
        addresses = private_anonymous_addresses(pid, begin, end)
        located = query_nodes(pid, addresses)
        counts = Counter(located)
        local = counts.get(rank, 0)
        remote = sum(n for node, n in counts.items() if node >= 0 and node != rank)
        errors = {node: n for node, n in counts.items() if node < 0}
        complete = len(addresses) == anon and not errors
        if not complete:
            ambiguous += max(anon, len(addresses))
        local_anon += local
        remote_anon += remote
        selected.append(dict(address=parts[0], mapping=mapping, numa_map=line, anon=anon,
            original_all_page_remote_count=remote_all_pages, private_present_pages=len(addresses),
            queried_node_counts=dict(counts), local_anon=local, remote_anon=remote, query_complete=complete,
            method='pagemap_private_anon_plus_readonly_node_query'))
    state2, birth2 = process(pid)
    assert birth2 == birth and state2 not in ['missing', 'Z', 'X']
    passed = status['Cpus_allowed_list'] == expected and method == 'spawn' and local_anon >= 4096 and remote_anon == 0 and ambiguous == 0
    return dict(pid=pid, rank=rank, birth=birth, cpus=status['Cpus_allowed_list'], expected_cpus=expected,
        mems=status['Mems_allowed_list'], worker_method=method, bound_anon_pages=local_anon,
        remote_pages_in_bound_anon=remote_anon, all_anon_pages=all_anon, anonymous_policies=sorted(policies),
        ambiguous_anonymous_pages=ambiguous, passed=passed, mappings=selected,
        query_nodes_argument=None, query_flags=0, observation_elapsed_seconds=time.perf_counter() - started)


def main():
    origin_path = ROOT / 'results/numa220-remote-mappings.json'
    original = json.loads(origin_path.read_text())
    output = ROOT / 'results/numa221-actual-anonymous-page-query.json'
    assert not output.exists()
    workers = []
    for item in original['workers']:
        try:
            workers.append(inspect(item['pid'], item['rank'], item['birth']))
        except Exception as error:
            workers.append(dict(pid=item['pid'], rank=item['rank'], birth=item['birth'], passed=False,
                error_type=type(error).__name__, error=str(error), node_query_incomplete=True))
    result = dict(status='actual_anonymous_page_query221', utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        case=original['case'], passed=all(w['passed'] for w in workers), workers=workers,
        original220_sha256=hashlib.sha256(origin_path.read_bytes()).hexdigest(), script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        model_requests_sent=0, memory_migration_performed=False, source_mutation_performed=False,
        scope='Strict zero-remote bound anonymous-page check. All-page locality suffices for wholly local mappings; mixed remote mappings require per-page private-anon classification and nodes=NULL read-only query. Query errors or count mismatches fail closed. No live189 files or source modified.',
        references=['https://www.man7.org/linux/man-pages/man5/numa_maps.5.html', 'https://www.man7.org/linux/man-pages/man2/move_pages.2.html', 'https://www.kernel.org/doc/html/latest/admin-guide/mm/pagemap.html'])
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'workers'}), flush=True)
    for worker in workers:
        print(json.dumps({k:v for k,v in worker.items() if k != 'mappings'}), flush=True)


if __name__ == '__main__':
    main()
