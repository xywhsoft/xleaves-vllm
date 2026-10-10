# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise actual scheduler admission and preemption before the first token."""

import json
import time

import pytest

from vllm.v1.request import RequestStatus

from .utils import create_requests, create_scheduler

pytestmark = pytest.mark.cpu_test


@pytest.fixture
def local_opt(tmp_path):
    config = dict(
        architectures=["OPTForCausalLM"],
        model_type="opt",
        hidden_size=64,
        ffn_dim=256,
        num_attention_heads=4,
        num_hidden_layers=2,
        vocab_size=128,
        max_position_embeddings=128,
        eos_token_id=2,
    )
    (tmp_path / "config.json").write_text(json.dumps(config))
    return str(tmp_path)


@pytest.mark.parametrize("prefill_chunks", [1, 2])
def test_partial_prefill_resume_retains_initial_cache_accounting(
    local_opt, prefill_chunks
):
    scheduler = create_scheduler(
        model=local_opt,
        max_num_batched_tokens=16,
        max_model_len=128,
        enable_prefix_caching=True,
        skip_tokenizer_init=True,
        num_blocks=64,
    )
    (request,) = create_requests(
        num_requests=1, num_tokens=64, block_size=16, max_tokens=8
    )
    scheduler.add_request(request)
    for _ in range(prefill_chunks):
        scheduled = scheduler.schedule()
        assert scheduled.num_scheduled_tokens[request.request_id] == 16
    assert request.prefill_stats.num_cached_tokens == 0
    scheduler.running.remove(request)
    scheduler._preempt_request(request, time.monotonic())
    scheduler.prev_step_scheduled_req_ids.clear()
    _, cached = scheduler.kv_cache_manager.get_computed_blocks(request)
    assert cached >= 16, "The regression must actually reuse its partial prefill"
    scheduler.schedule()
    assert request.num_preemptions == 1
    assert request.num_computed_tokens > cached
    # This work was computed inside this cold request, before preemption.
    assert request.prefill_stats.num_cached_tokens == 0


def test_new_request_reports_genuine_prior_request_cache(local_opt):
    scheduler = create_scheduler(
        model=local_opt,
        max_num_batched_tokens=16,
        max_model_len=128,
        enable_prefix_caching=True,
        skip_tokenizer_init=True,
        num_blocks=64,
    )
    old, new = create_requests(
        num_requests=2, num_tokens=64, same_prompt=True, block_size=16, max_tokens=8
    )
    scheduler.add_request(old)
    scheduler.schedule()
    scheduler.schedule()
    scheduler.finish_requests(old.request_id, RequestStatus.FINISHED_ABORTED)
    _, cached = scheduler.kv_cache_manager.get_computed_blocks(new)
    assert cached >= 16
    scheduler.add_request(new)
    scheduler.schedule()
    assert new.num_preemptions == 0
    assert new.prefill_stats.num_cached_tokens == cached
