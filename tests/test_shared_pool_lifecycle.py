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

"""Persistent shared-pool management binding tests."""

import mmap
import pickle

import pytest

from shared_pool_utils import destroy_pool_if_present
from shared_pool_utils import make_pool_config
from shared_pool_utils import unique_pool_name
from torch_spyre._C import ChunkDescriptorEntry
from torch_spyre._C import CompatibleBlockKey
from torch_spyre._C import PoolResultCode
from torch_spyre._C import PoolState
from torch_spyre._C import PoolBackend
from torch_spyre._C import PoolCompatibility
from torch_spyre._C import PoolConfig
from torch_spyre._C import MetadataConfig
from torch_spyre._C import SharedDataPartitionConfig
from torch_spyre._C import SharedDataPoolConfig
from torch_spyre._C import SharedHostPool
from torch_spyre._C import SharedPool
from torch_spyre._C import _decode_pool_config_state
from torch_spyre._C import _decode_pool_descriptor_state
from torch_spyre._C import _decode_reservation_token_state


def make_two_rank_config(name: str) -> PoolConfig:
    """Build two rank-homogeneous arenas under one metadata directory."""
    compatibility = PoolCompatibility("test-model", "float16", 16, 2)
    rank0 = SharedDataPoolConfig(
        "host-r0",
        PoolBackend.HOST_DRAM,
        [
            SharedDataPartitionConfig("host-r0-c0", 0, 0, 2, 128, None),
            SharedDataPartitionConfig("host-r0-c1", 0, 1, 2, 128, None),
        ],
        compatibility,
    )
    rank1 = SharedDataPoolConfig(
        "host-r1",
        PoolBackend.HOST_DRAM,
        [SharedDataPartitionConfig("host-r1-c0", 1, 0, 2, 128, None)],
        compatibility,
    )
    return PoolConfig(name, MetadataConfig(1), [rank0, rank1])


def test_attach_returns_only_ready_requested_rank_handles():
    """Dropping rank filtering or eager handles must fail this test."""
    name = unique_pool_name("ranked-attach")
    created = SharedPool.create_pool(make_two_rank_config(name))
    assert created.code is PoolResultCode.OK
    attachment = None
    try:
        result = SharedPool.attach_pool(created.descriptor, 1)

        assert result.code is PoolResultCode.OK
        assert result.attachment is not None
        attachment = result.attachment
        assert attachment.tp_rank == 1
        assert attachment.metadata.pool_count() == 3
        assert [item.registration.name for item in attachment.data_pools] == [
            "host-r1-c0"
        ]
        assert isinstance(attachment.data_pools[0].data_pool, SharedHostPool)
    finally:
        SharedPool.detach_pool(attachment)
        destroy_pool_if_present(name)


def test_attach_rejects_unconfigured_rank_without_partial_attachment():
    """Accepting an absent TP rank must fail with no usable aggregate."""
    name = unique_pool_name("unknown-rank")
    created = SharedPool.create_pool(make_two_rank_config(name))
    assert created.code is PoolResultCode.OK
    try:
        result = SharedPool.attach_pool(created.descriptor, 2)

        assert result.code is PoolResultCode.CONFIGURATION_MISMATCH
        assert result.attachment is None
    finally:
        destroy_pool_if_present(name)


def test_close_invalidates_every_pool_attachment_alias():
    """Closing either Python alias must close their shared aggregate state."""
    name = unique_pool_name("attachment-alias")
    created = SharedPool.create_pool(make_pool_config(name))
    assert created.code is PoolResultCode.OK
    result = SharedPool.attach_pool(created.descriptor, 0)
    assert result.code is PoolResultCode.OK
    first = result.attachment
    second = result.attachment
    try:
        first.close()

        with pytest.raises(RuntimeError, match="closed"):
            _ = second.metadata
        second.close()
        assert SharedPool.destroy_pool(created.descriptor).code is PoolResultCode.OK
    finally:
        SharedPool.detach_pool(second)
        destroy_pool_if_present(name)


def test_create_open_destroy_persistent_generation():
    """Create must publish one persistent generation until explicit destroy."""
    name = unique_pool_name("management")
    config = make_pool_config(name, num_slots=2, slot_bytes=128)
    try:
        created = SharedPool.create_pool(config)
        assert created.code is PoolResultCode.OK
        assert created.descriptor.state is PoolState.READY
        assert created.descriptor.effective_config.data_pools[0].slot_bytes >= 128

        opened = SharedPool.open_pool(name)
        assert opened.code is PoolResultCode.OK
        assert opened.descriptor == created.descriptor

        duplicate = SharedPool.create_pool(config)
        assert duplicate.code is PoolResultCode.ALREADY_EXISTS
        assert duplicate.descriptor.generation == created.descriptor.generation

        destroyed = SharedPool.destroy_pool(created.descriptor)
        assert destroyed.code is PoolResultCode.OK
        assert SharedPool.open_pool(name).code is PoolResultCode.NOT_FOUND
    finally:
        destroy_pool_if_present(name)


def test_pool_descriptor_pickle_round_trip():
    """Descriptor transport must retain generation and concrete compatibility."""
    name = unique_pool_name("pickle")
    config = make_pool_config(name, num_slots=3, slot_bytes=257, max_chunks=2)
    try:
        created = SharedPool.create_pool(config)
        assert created.code is PoolResultCode.OK

        restored = pickle.loads(pickle.dumps(created.descriptor))

        assert restored == created.descriptor
        assert restored.generation.bytes == created.descriptor.generation.bytes
        assert len(restored.generation.bytes) == 16
        compatibility = restored.effective_config.data_pools[0].compatibility
        assert compatibility.model == "test-model"
        assert compatibility.kv_dtype == "float16"
        assert compatibility.block_size == 16
        assert compatibility.tensor_parallel_size == 1
    finally:
        destroy_pool_if_present(name)


def test_partitioned_descriptor_and_config_pickle_round_trip():
    """Pickle transport must retain every effective arena partition."""
    name = unique_pool_name("partition-pickle")
    compatibility = PoolCompatibility(
        "test-model", "float16", 16, 2, "layout-signature"
    )
    rank0_partitions = [
        SharedDataPartitionConfig("host-r0-c0", 0, 0, 4, 128, None),
        SharedDataPartitionConfig("host-r0-c1", 0, 1, 4, 4097, None),
    ]
    rank1_partitions = [
        SharedDataPartitionConfig("host-r1-c0", 1, 0, 4, 8192, None),
    ]
    config = PoolConfig(
        name,
        MetadataConfig(2),
        [
            SharedDataPoolConfig(
                "host-r0", PoolBackend.HOST_DRAM, rank0_partitions, compatibility
            ),
            SharedDataPoolConfig(
                "host-r1", PoolBackend.HOST_DRAM, rank1_partitions, compatibility
            ),
        ],
    )
    try:
        assert pickle.loads(pickle.dumps(config)) == config
        created = SharedPool.create_pool(config)
        assert created.code is PoolResultCode.OK

        restored = pickle.loads(pickle.dumps(created.descriptor))

        assert restored == created.descriptor
        assert (
            restored.effective_config.data_pools[0].compatibility.layout_signature
            == "layout-signature"
        )
        effective = [
            partition
            for arena in restored.effective_config.data_pools
            for partition in arena.partitions
        ]
        assert [partition.name for partition in effective] == [
            "host-r0-c0",
            "host-r0-c1",
            "host-r1-c0",
        ]
        assert [partition.tp_rank for partition in effective] == [0, 0, 1]
        assert [partition.cache_index for partition in effective] == [0, 1, 0]
        assert all(partition.offset_bytes is not None for partition in effective)
    finally:
        destroy_pool_if_present(name)


def test_partitioned_metadata_survives_the_last_detach():
    """Zero attachments must preserve partition records and published entries."""
    name = unique_pool_name("partition-persistent")
    compatibility = PoolCompatibility("test-model", "float16", 16, 2)
    config = PoolConfig(
        name,
        MetadataConfig(2),
        [
            SharedDataPoolConfig(
                "host-r0",
                PoolBackend.HOST_DRAM,
                [SharedDataPartitionConfig("host-r0-c0", 0, 0, 4, 128, None)],
                compatibility,
            ),
            SharedDataPoolConfig(
                "host-r1",
                PoolBackend.HOST_DRAM,
                [SharedDataPartitionConfig("host-r1-c0", 1, 0, 4, 4097, None)],
                compatibility,
            ),
        ],
    )
    first = None
    second = None
    try:
        created = SharedPool.create_pool(config)
        assert created.code is PoolResultCode.OK
        first_result = SharedPool.attach_pool(created.descriptor, 1)
        assert first_result.code is PoolResultCode.OK
        first = first_result.attachment
        metadata = first.metadata
        registered = metadata.find_pool("host-r1-c0")
        key = CompatibleBlockKey(registered.compatibility, 0x4455)
        reservation = metadata.claim(registered.pool_ref, key)
        metadata.publish(reservation, [ChunkDescriptorEntry(1, 4097)])
        metadata = None
        SharedPool.detach_pool(first)
        first = None

        opened = SharedPool.open_pool(name)
        assert opened.code is PoolResultCode.OK
        assert opened.descriptor == created.descriptor
        second_result = SharedPool.attach_pool(opened.descriptor, 1)
        assert second_result.code is PoolResultCode.OK
        second = second_result.attachment
        metadata = second.metadata
        restored = metadata.find_pool("host-r1-c0")
        assert restored.pool_ref == registered.pool_ref
        entry = metadata.lookup(CompatibleBlockKey(restored.compatibility, 0x4455))
        assert entry is not None
        assert [(chunk.domain_id, chunk.size) for chunk in entry.chunks] == [(1, 4097)]
        metadata = None
        SharedPool.detach_pool(second)
        second = None

        assert SharedPool.destroy_pool(created.descriptor).code is PoolResultCode.OK
    finally:
        if first is not None:
            SharedPool.detach_pool(first)
        if second is not None:
            SharedPool.detach_pool(second)
        destroy_pool_if_present(name)


def test_partitioned_pickle_state_rejects_malformed_geometry_and_tokens():
    """Versioned states must reject mixed geometry and invalid token integers."""
    compatibility = ("test-model", "float16", 16, 2)
    partitions = [
        ("host-r0-c0", 0, 0, 4, 4096, 0),
        ("host-r0-c1", 0, 1, 4, 4096, 0),
    ]
    mixed_child = (
        "host-0",
        PoolBackend.HOST_DRAM.value,
        8,
        4096,
        *compatibility,
        partitions,
    )

    with pytest.raises(ValueError, match="offsets are not canonical"):
        _decode_pool_config_state((2, "/malformed", 2, [mixed_child]))
    with pytest.raises(ValueError, match="pickle version"):
        _decode_pool_config_state((1, "/malformed", 2, []))
    with pytest.raises(ValueError, match="must contain 9 fields"):
        _decode_reservation_token_state((1, 2, 3))
    with pytest.raises(ValueError, match="pickle version"):
        _decode_reservation_token_state((2, 0, 0, 0, 0, 0, 0, 0, 0))
    with pytest.raises(ValueError, match="out of range|unsigned integer"):
        _decode_reservation_token_state((1, -1, 0, 0, 0, 0, 0, 0, 0))


def test_config_pickle_accepts_raw_partition_geometry_and_preserves_order():
    """Requested config is validated canonically without rewriting its facts."""
    partitions = [
        ("host-r0-c1", 0, 1, 1, 128, mmap.PAGESIZE),
        ("host-r0-c0", 0, 0, 1, 128, 0),
    ]
    child = (
        "host-0",
        PoolBackend.HOST_DRAM.value,
        0,
        0,
        "test-model",
        "float16",
        16,
        1,
        partitions,
    )

    config = _decode_pool_config_state((2, "/raw-config", 1, [child]))

    decoded = config.data_pools[0].partitions
    assert [partition.name for partition in decoded] == [
        "host-r0-c1",
        "host-r0-c0",
    ]
    assert [partition.slot_bytes for partition in decoded] == [128, 128]


def test_descriptor_pickle_rejects_noncanonical_partition_geometry():
    """Effective descriptors must already contain Flex's canonical geometry."""
    generation = b"generation-00001"
    compatibility = ("test-model", "float16", 16, 1)
    reversed_partitions = [
        ("host-r0-c1", 0, 1, 1, mmap.PAGESIZE, mmap.PAGESIZE),
        ("host-r0-c0", 0, 0, 1, mmap.PAGESIZE, 0),
    ]
    unaligned_partitions = [("host-r0-c0", 0, 0, 1, 128, 0)]

    for partitions, num_slots in [
        (reversed_partitions, 2),
        (unaligned_partitions, 1),
    ]:
        child = (
            "host-0",
            PoolBackend.HOST_DRAM.value,
            num_slots,
            mmap.PAGESIZE,
            *compatibility,
            partitions,
        )
        state = (2, "/bad-descriptor", generation, PoolState.READY.value, 1, [child])
        with pytest.raises(ValueError, match="canonical|aligned"):
            _decode_pool_descriptor_state(state)


def test_descriptor_pickle_rejects_flat_geometry_overflow():
    """Legacy-style effective geometry must not wrap native arena bytes."""
    child = (
        "host-0",
        PoolBackend.HOST_DRAM.value,
        (1 << 64) - 1,
        2,
        "test-model",
        "float16",
        16,
        1,
        [],
    )
    state = (
        2,
        "/overflow-descriptor",
        b"generation-00001",
        PoolState.READY.value,
        1,
        [child],
    )

    with pytest.raises(ValueError, match="overflows"):
        _decode_pool_descriptor_state(state)


def test_legacy_uniform_descriptor_pickle_normalizes_to_one_partition():
    """Version-1 descriptor state remains readable as one logical partition."""
    state = (
        1,
        "/legacy",
        b"generation-00001",
        PoolState.READY.value,
        2,
        [
            (
                "host-0",
                PoolBackend.HOST_DRAM.value,
                4,
                4096,
                "test-model",
                "float16",
                16,
                1,
            )
        ],
    )

    descriptor = _decode_pool_descriptor_state(state)

    child = descriptor.effective_config.data_pools[0]
    assert child.num_slots == 0
    assert child.slot_bytes == 0
    assert len(child.partitions) == 1
    assert child.partitions[0].name == "host-0"
    assert child.partitions[0].num_slots == 4
    assert child.partitions[0].slot_bytes == 4096
    assert child.partitions[0].offset_bytes == 0


def test_pool_descriptor_rejects_malformed_pickle_state():
    """Descriptor decoding must reject a non-opaque generation value."""
    malformed_state = (
        1,
        "malformed-generation",
        b"too-short",
        PoolState.READY.value,
        1,
        [],
    )

    with pytest.raises(ValueError, match="generation"):
        _decode_pool_descriptor_state(malformed_state)


def test_retained_metadata_keeps_generation_busy_after_attachment_close():
    """Closing the aggregate must not invalidate a retained metadata handle."""
    name = unique_pool_name("metadata-lease")
    created = SharedPool.create_pool(make_pool_config(name))
    assert created.code is PoolResultCode.OK
    attached = SharedPool.attach_pool(created.descriptor, 0)
    assert attached.code is PoolResultCode.OK
    attachment = attached.attachment
    metadata = attachment.metadata
    try:
        SharedPool.detach_pool(attachment)

        assert metadata.pool_count() == 1
        assert (
            SharedPool.destroy_pool(created.descriptor).code is PoolResultCode.POOL_BUSY
        )

        del metadata
        assert SharedPool.destroy_pool(created.descriptor).code is PoolResultCode.OK
    finally:
        SharedPool.detach_pool(attachment)
        destroy_pool_if_present(name)


def test_attached_pool_keeps_generation_busy_after_attachment_close():
    """A retained eager data handle must keep its root lifecycle lease."""
    name = unique_pool_name("data-lease")
    created = SharedPool.create_pool(make_pool_config(name))
    assert created.code is PoolResultCode.OK
    attached = SharedPool.attach_pool(created.descriptor, 0)
    assert attached.code is PoolResultCode.OK
    attachment = attached.attachment
    entry = attachment.data_pools[0]
    data = entry.data_pool
    del entry
    try:
        SharedPool.detach_pool(attachment)
        assert (
            SharedPool.destroy_pool(created.descriptor).code is PoolResultCode.POOL_BUSY
        )

        del data
        assert SharedPool.destroy_pool(created.descriptor).code is PoolResultCode.OK
    finally:
        SharedPool.detach_pool(attachment)
        destroy_pool_if_present(name)


def test_attach_result_attachment_is_none_on_stale_generation():
    """A stale capability must not expose a replacement-generation attachment."""
    name = unique_pool_name("stale-attach")
    config = make_pool_config(name)
    first = SharedPool.create_pool(config)
    assert first.code is PoolResultCode.OK
    assert SharedPool.destroy_pool(first.descriptor).code is PoolResultCode.OK
    replacement = SharedPool.create_pool(config)
    assert replacement.code is PoolResultCode.OK
    try:
        stale = SharedPool.attach_pool(first.descriptor, 0)

        assert stale.code is PoolResultCode.STALE_GENERATION
        assert stale.attachment is None
    finally:
        destroy_pool_if_present(name)


def test_attach_by_identity_is_primary_worker_flow():
    """A worker must attach using only the controller-provided identity."""
    name = unique_pool_name("identity-attach")
    created = SharedPool.create_pool(make_pool_config(name))
    assert created.code is PoolResultCode.OK
    try:
        attached = SharedPool.attach_pool(
            created.descriptor.name,
            created.descriptor.generation.bytes,
            0,
        )

        assert attached.code is PoolResultCode.OK
        assert attached.attachment is not None
        SharedPool.detach_pool(attached.attachment)
    finally:
        destroy_pool_if_present(name)


def test_attach_by_identity_attests_expected_effective_config():
    """Identity attach must reject a worker plan for another physical layout."""
    name = unique_pool_name("attested-attach")
    created = SharedPool.create_pool(make_pool_config(name))
    assert created.code is PoolResultCode.OK
    incompatible = make_pool_config(name, slot_bytes=8192)
    try:
        attached = SharedPool.attach_pool(
            name,
            created.descriptor.generation.bytes,
            incompatible,
            0,
        )

        assert attached.code is PoolResultCode.CONFIGURATION_MISMATCH
        assert attached.attachment is None
    finally:
        destroy_pool_if_present(name)


def test_attach_by_identity_reports_missing_pool():
    """Identity attach must report absence rather than create a pool."""
    name = unique_pool_name("missing-attach")

    attached = SharedPool.attach_pool(name, bytes(16), 0)

    assert attached.code is PoolResultCode.NOT_FOUND
    assert attached.attachment is None


def test_attach_by_identity_rejects_replacement_generation():
    """An old worker capability must not attach a same-name replacement."""
    name = unique_pool_name("identity-stale")
    config = make_pool_config(name)
    first = SharedPool.create_pool(config)
    assert first.code is PoolResultCode.OK
    assert SharedPool.destroy_pool(first.descriptor).code is PoolResultCode.OK
    replacement = SharedPool.create_pool(config)
    assert replacement.code is PoolResultCode.OK
    try:
        attached = SharedPool.attach_pool(name, first.descriptor.generation.bytes, 0)

        assert attached.code is PoolResultCode.STALE_GENERATION
        assert attached.attachment is None
    finally:
        destroy_pool_if_present(name)


@pytest.mark.parametrize("generation", [bytes(15), bytes(17)])
def test_attach_by_identity_rejects_malformed_generation(generation):
    """Generation length must be rejected before the pool is opened."""
    with pytest.raises(ValueError, match="16 bytes"):
        SharedPool.attach_pool(unique_pool_name("malformed-attach"), generation, 0)


def test_detach_is_idempotent():
    """Repeated detach calls must be safe for unconditional cleanup."""
    name = unique_pool_name("idempotent-detach")
    created = SharedPool.create_pool(make_pool_config(name))
    assert created.code is PoolResultCode.OK
    attached = SharedPool.attach_pool(created.descriptor, 0)
    assert attached.code is PoolResultCode.OK
    try:
        SharedPool.detach_pool(attached.attachment)
        SharedPool.detach_pool(attached.attachment)

        assert SharedPool.destroy_pool(created.descriptor).code is PoolResultCode.OK
    finally:
        SharedPool.detach_pool(attached.attachment)
        destroy_pool_if_present(name)
