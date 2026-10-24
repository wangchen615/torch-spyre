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

"""Helpers for lifecycle-managed shared-pool tests."""

import os
import uuid

from torch_spyre._C import MetadataConfig
from torch_spyre._C import PoolBackend
from torch_spyre._C import PoolCompatibility
from torch_spyre._C import PoolConfig
from torch_spyre._C import PoolResultCode
from torch_spyre._C import SharedDataPoolConfig
from torch_spyre._C import SharedPool


def unique_pool_name(label: str) -> str:
    """Return a process-unique POSIX shared-pool name."""
    return f"torch-spyre-{label}-{os.getpid()}-{uuid.uuid4().hex}"


def make_pool_config(
    name: str, *, num_slots: int = 2, slot_bytes: int = 128, max_chunks: int = 1
) -> PoolConfig:
    """Build the fixed one-child configuration used by focused tests."""
    compatibility = PoolCompatibility("test-model", "float16", 16, 1)
    child = SharedDataPoolConfig(
        "host-0",
        PoolBackend.HOST_DRAM,
        num_slots,
        slot_bytes,
        compatibility,
    )
    return PoolConfig(name, MetadataConfig(max_chunks), [child])


def destroy_pool_if_present(name: str) -> None:
    """Destroy a recoverable test generation when one exists."""
    opened = SharedPool.open_pool(name)
    if opened.code is PoolResultCode.OK:
        result = SharedPool.destroy_pool(opened.descriptor)
        if result.code not in (PoolResultCode.OK, PoolResultCode.NOT_FOUND):
            raise RuntimeError(
                f"failed to clean {name}: {result.code}: {result.detail}"
            )
