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

import pickle

import pytest

from shared_pool_utils import destroy_pool_if_present
from shared_pool_utils import make_pool_config
from shared_pool_utils import unique_pool_name
from torch_spyre._C import PoolResultCode
from torch_spyre._C import PoolState
from torch_spyre._C import SharedPool
from torch_spyre._C import _decode_pool_descriptor_state


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
