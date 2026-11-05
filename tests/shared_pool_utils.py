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
from torch_spyre._C import SharedDataPool
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


class LifecyclePool:
    """Own one test attachment and its eagerly attached host child."""

    def __init__(self, label: str, *, num_slots: int, slot_bytes: int) -> None:
        self.name = unique_pool_name(label)
        self.created = SharedPool.create_pool(
            make_pool_config(
                self.name,
                num_slots=num_slots,
                slot_bytes=slot_bytes,
            )
        )
        if self.created.code is not PoolResultCode.OK:
            raise RuntimeError(f"failed to create {self.name}: {self.created.detail}")

        attached = SharedPool.attach_pool(self.created.descriptor, 0)
        if attached.code is not PoolResultCode.OK or attached.attachment is None:
            SharedPool.destroy_pool(self.created.descriptor)
            raise RuntimeError(f"failed to attach {self.name}: {attached.detail}")
        self.attachment = attached.attachment
        self.metadata = self.attachment.metadata
        by_name = {item.registration.name: item for item in self.attachment.data_pools}
        entry = by_name.get("host-0")
        if entry is None:
            SharedPool.detach_pool(self.attachment)
            SharedPool.destroy_pool(self.created.descriptor)
            raise RuntimeError(f"host-0 is missing from {self.name}")
        if entry.registration.slot_count != num_slots:
            SharedPool.detach_pool(self.attachment)
            SharedPool.destroy_pool(self.created.descriptor)
            raise RuntimeError(f"host-0 has unexpected geometry in {self.name}")
        self._pool: SharedDataPool | None = entry.data_pool
        if self._pool.slot_count() != num_slots:
            self.close()
            raise RuntimeError(f"host-0 mapping has unexpected geometry in {self.name}")

    @property
    def pool(self) -> SharedDataPool:
        """Return the eager child while this lifecycle owner is attached."""
        if self._pool is None:
            raise RuntimeError(f"pool {self.name} is closed")
        return self._pool

    def close(self) -> None:
        """Release child and metadata leases before destroying the generation."""
        if self._pool is not None:
            self._pool = None
            del self.metadata
            SharedPool.detach_pool(self.attachment)
        destroyed = SharedPool.destroy_pool(self.created.descriptor)
        if destroyed.code not in (PoolResultCode.OK, PoolResultCode.NOT_FOUND):
            raise RuntimeError(
                f"failed to destroy {self.name}: {destroyed.code}: {destroyed.detail}"
            )
