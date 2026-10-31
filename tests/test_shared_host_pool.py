# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared-host-pool geometry tests through the persistent lifecycle."""

import torch

from shared_pool_utils import destroy_pool_if_present
from shared_pool_utils import make_pool_config
from shared_pool_utils import unique_pool_name
from torch_spyre._C import PoolResultCode
from torch_spyre._C import SharedDataPool
from torch_spyre._C import SharedHostPool
from torch_spyre._C import SharedPool
from torch_spyre._C import MetadataConfig
from torch_spyre._C import PoolBackend
from torch_spyre._C import PoolCompatibility
from torch_spyre._C import PoolConfig
from torch_spyre._C import SharedDataPartitionConfig
from torch_spyre._C import SharedDataPoolConfig
from torch_spyre._C import copy_tensor_raw


def test_attached_host_pool_geometry_and_name():
    """Eager attachment must return the configured child and geometry."""
    name = unique_pool_name("host-geometry")
    created = SharedPool.create_pool(make_pool_config(name, num_slots=5, slot_bytes=5))
    assert created.code is PoolResultCode.OK
    attached = SharedPool.attach_pool(created.descriptor, 0)
    assert attached.code is PoolResultCode.OK
    attachment = attached.attachment
    entry = None
    pool = None
    try:
        entry = attachment.data_pools[0]
        assert entry.registration.name == "host-0"
        pool = entry.data_pool

        assert pool.name() == "host-0"
        assert pool.slot_count() == 5
        assert pool.slot_bytes() >= 5
        assert pool.total_bytes() == pool.slot_count() * pool.slot_bytes()
        assert not hasattr(attachment.metadata, "resolve_pool")
    finally:
        del pool
        del entry
        SharedPool.detach_pool(attachment)
        destroy_pool_if_present(name)


def test_attached_host_pool_preserves_data_pool_hierarchy():
    """A host child must remain usable through the shared-data base API."""
    name = unique_pool_name("host-hierarchy")
    created = SharedPool.create_pool(
        make_pool_config(name, num_slots=2, slot_bytes=128)
    )
    assert created.code is PoolResultCode.OK
    attached = SharedPool.attach_pool(created.descriptor, 0)
    assert attached.code is PoolResultCode.OK
    attachment = attached.attachment
    pool = None
    try:
        pool = attachment.data_pools[0].data_pool

        assert isinstance(pool, SharedHostPool)
        assert isinstance(pool, SharedDataPool)
        assert not hasattr(pool, "slot_ptr")
        assert not hasattr(pool, "create_or_attach")
        assert not hasattr(pool, "unlink_by_name")
    finally:
        del pool
        SharedPool.detach_pool(attachment)
        destroy_pool_if_present(name)


def test_partition_views_write_distinct_arena_offsets():
    """Two logical partitions must not alias the same physical arena bytes."""
    name = unique_pool_name("partition-offsets")
    compatibility = PoolCompatibility("test-model", "float16", 16, 1)
    config = PoolConfig(
        name,
        MetadataConfig(1),
        [
            SharedDataPoolConfig(
                "host-0",
                PoolBackend.HOST_DRAM,
                [
                    SharedDataPartitionConfig("host-r0-c0", 0, 0, 1, 128, None),
                    SharedDataPartitionConfig("host-r0-c1", 0, 1, 1, 128, None),
                ],
                compatibility,
            )
        ],
    )
    created = SharedPool.create_pool(config)
    assert created.code is PoolResultCode.OK
    attached = SharedPool.attach_pool(created.descriptor, 0)
    assert attached.code is PoolResultCode.OK
    attachment = attached.attachment
    entries = attachment.data_pools
    by_name = {}
    pools = []
    try:
        by_name = {entry.registration.name: entry.data_pool for entry in entries}
        pools = [by_name["host-r0-c0"], by_name["host-r0-c1"]]
        assert pools[0].name() != pools[1].name()

        first = torch.arange(64, device="spyre", dtype=torch.float16)
        second = first + 1000
        copy_tensor_raw(first, pools[0], 0, to_device=False)
        copy_tensor_raw(second, pools[1], 0, to_device=False)
        first_result = torch.empty_like(first)
        second_result = torch.empty_like(second)
        copy_tensor_raw(first_result, pools[0], 0, to_device=True)
        copy_tensor_raw(second_result, pools[1], 0, to_device=True)
        assert torch.equal(first_result.cpu(), first.cpu())
        assert torch.equal(second_result.cpu(), second.cpu())
    finally:
        pools.clear()
        by_name.clear()
        entries.clear()
        SharedPool.detach_pool(attachment)
        destroy_pool_if_present(name)
