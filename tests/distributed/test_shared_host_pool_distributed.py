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

"""Two-rank fixed-membership host-pool lifecycle verification."""

import os

import pytest
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_utils import run_tests

from torch_spyre._C import MetadataConfig
from torch_spyre._C import PoolBackend
from torch_spyre._C import PoolCompatibility
from torch_spyre._C import PoolConfig
from torch_spyre._C import PoolResultCode
from torch_spyre._C import SharedDataPoolConfig
from torch_spyre._C import SharedPool


if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
    pytest.skip("distributed environment is not configured", allow_module_level=True)
if int(os.environ["WORLD_SIZE"]) != 2:
    pytest.skip("this test requires exactly two ranks", allow_module_level=True)


C10D_BACKEND = "spyreccl"


def destroy_pool_if_present(name):
    opened = SharedPool.open_pool(name)
    if opened.code is PoolResultCode.OK:
        result = SharedPool.destroy_pool(opened.descriptor)
        if result.code not in (PoolResultCode.OK, PoolResultCode.NOT_FOUND):
            raise RuntimeError(f"failed to clean {name}: {result.code}")


def make_config(name, num_slots, slot_bytes):
    compatibility = PoolCompatibility("distributed-test", "float16", 64, 2)
    child = SharedDataPoolConfig(
        "host-0",
        PoolBackend.HOST_DRAM,
        num_slots,
        slot_bytes,
        compatibility,
    )
    return PoolConfig(name, MetadataConfig(1), [child])


class TestSharedHostPoolCrossProcess(TestCase):
    @classmethod
    def setUpClass(cls):
        if not dist.distributed_c10d.is_backend_available(C10D_BACKEND):
            raise RuntimeError(f"missing C10 backend {C10D_BACKEND}")
        if not dist.is_initialized():
            dist.init_process_group(f"cpu:gloo,spyre:{C10D_BACKEND}")
        cls.comm_rank = dist.get_rank()

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    def test_cross_process_shared_pool(self):
        name = f"torch-spyre-dist-host-{self.id()}"
        descriptor_box = [None]
        attached = None
        pool = None
        descriptor = None
        try:
            if self.comm_rank == 0:
                destroy_pool_if_present(name)
                created = SharedPool.create_pool(make_config(name, 1, 128))
                self.assertIs(created.code, PoolResultCode.OK)
                opened = SharedPool.open_pool(name)
                self.assertIs(opened.code, PoolResultCode.OK)
                self.assertEqual(opened.descriptor, created.descriptor)
                descriptor_box[0] = opened.descriptor

            dist.broadcast_object_list(descriptor_box, src=0)
            descriptor = descriptor_box[0]
            attached = SharedPool.attach_pool(descriptor)
            self.assertIs(attached.code, PoolResultCode.OK)
            registered = attached.metadata.find_pool("host-0")
            self.assertIsNotNone(registered)
            pool = attached.metadata.resolve_pool(registered.pool_ref)

            self.assertEqual(pool.slot_count(), 1)
            self.assertEqual(pool.name(), "host-0")
            self.assertEqual(pool.total_bytes(), pool.slot_count() * pool.slot_bytes())
            dist.barrier()
        finally:
            pool = None
            if attached is not None:
                SharedPool.detach_pool(attached.metadata)
            if dist.is_initialized():
                dist.barrier()
            if self.comm_rank == 0 and descriptor is not None:
                result = SharedPool.destroy_pool(descriptor)
                self.assertIn(
                    result.code,
                    (PoolResultCode.OK, PoolResultCode.NOT_FOUND),
                )
            if dist.is_initialized():
                dist.barrier()


if __name__ == "__main__":
    run_tests()
