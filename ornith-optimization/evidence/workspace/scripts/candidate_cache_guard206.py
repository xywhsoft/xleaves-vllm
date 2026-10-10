"""CPU-only instance prototype. No serving source is changed."""
import functools
from vllm.v1.request import RequestStatus


def prototype(scheduler, inject_exception=False):
    manager = scheduler.kv_cache_manager
    pool = manager.block_pool
    original_schedule = scheduler.schedule
    original_allocate = manager.allocate_slots
    held = []
    protected = False
    allocations = 0
    history = []

    def allocate(request, *args, **kwargs):
        nonlocal protected, allocations
        if (request.status == RequestStatus.WAITING and not protected
                and manager.enable_caching and scheduler.connector is None):
            protected = True
            seen = set()
            capacity = max(0, scheduler.max_num_running_reqs - len(scheduler.running))
            candidates = list(scheduler.waiting) + list(scheduler.skipped_waiting)
            old_boundary = manager.coordinator.shared_prefix_boundary
            try:
                for pending in candidates[:capacity]:
                    if (pending.status != RequestStatus.WAITING
                            or pending.num_computed_tokens
                            or pending.skip_reading_prefix_cache):
                        continue
                    blocks, _ = manager.coordinator.find_longest_cache_hit(
                        pending.block_hashes, pending.num_tokens - 1)
                    for group in blocks:
                        for block in group:
                            if block.is_null or block.block_id in seen:
                                continue
                            seen.add(block.block_id)
                            old_ref = block.ref_cnt
                            try:
                                pool.touch([block])
                            finally:
                                # touch increments the ref before its optional
                                # metrics callback. Track partial success too.
                                if block.ref_cnt > old_ref:
                                    held.append(block)
            finally:
                manager.coordinator.shared_prefix_boundary = old_boundary
            history.append(dict(reserved_blocks=len(held), running=len(scheduler.running),
                                candidate_count=min(capacity, len(candidates))))
        if request.status == RequestStatus.WAITING:
            allocations += 1
            if inject_exception and allocations == 2:
                raise RuntimeError('injected_cpu_allocation_exception204')
        result = original_allocate(request, *args, **kwargs)
        if result is None and held:
            # Capacity must take precedence over best-effort cache retention.
            # Retry once after releasing this step's temporary references.
            released = len(held)
            pool.free_blocks(held)
            held.clear()
            history.append(dict(pressure_retry=True, released_blocks=released))
            result = original_allocate(request, *args, **kwargs)
        return result

    @functools.wraps(original_schedule)
    def schedule():
        nonlocal protected, allocations
        assert not held
        protected = False
        allocations = 0
        manager.allocate_slots = allocate
        try:
            return original_schedule()
        finally:
            manager.allocate_slots = original_allocate
            if held:
                pool.free_blocks(held)
            held.clear()
    scheduler.schedule = schedule
    return history
