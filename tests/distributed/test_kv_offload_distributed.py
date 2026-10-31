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

"""Two-rank raw-copy verification through a persistent host pool."""

import os

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_utils import run_tests

from torch_spyre._C import MetadataConfig
from torch_spyre._C import PoolBackend
from torch_spyre._C import PoolCompatibility
from torch_spyre._C import PoolConfig
from torch_spyre._C import PoolResultCode
from torch_spyre._C import SharedDataPartitionConfig
from torch_spyre._C import SharedDataPoolConfig
from torch_spyre._C import SharedPool
from torch_spyre._C import copy_tensor_raw
from torch_spyre._C import get_composite_address


if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
    pytest.skip("distributed environment is not configured", allow_module_level=True)
if int(os.environ["WORLD_SIZE"]) != 2:
    pytest.skip("this test requires exactly two ranks", allow_module_level=True)


DEVICE = torch.device(f"spyre:{os.environ['RANK']}")
C10D_BACKEND = "spyreccl"


def destroy_pool_if_present(name):
    opened = SharedPool.open_pool(name)
    if opened.code is PoolResultCode.OK:
        result = SharedPool.destroy_pool(opened.descriptor)
        if result.code not in (PoolResultCode.OK, PoolResultCode.NOT_FOUND):
            raise RuntimeError(f"failed to clean {name}: {result.code}")


def make_config(name, slot_bytes):
    compatibility = PoolCompatibility("distributed-test", "float16", 64, 2)
    arenas = [
        SharedDataPoolConfig(
            f"arena-r{rank}",
            PoolBackend.HOST_DRAM,
            [SharedDataPartitionConfig(f"host-r{rank}", rank, 0, 1, slot_bytes)],
            compatibility,
        )
        for rank in range(2)
    ]
    return PoolConfig(name, MetadataConfig(1), arenas)


class TestKVOffloadCrossProcess(TestCase):
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

    def test_cross_process_reload(self):
        expected = torch.arange(64, device=DEVICE, dtype=torch.float16)
        slot_bytes = get_composite_address(expected).total_size
        name = f"torch-spyre-dist-kv-{self.id()}"
        descriptor_box = [None]
        attachment = None
        pool = None
        descriptor = None
        try:
            if self.comm_rank == 0:
                destroy_pool_if_present(name)
                created = SharedPool.create_pool(make_config(name, slot_bytes))
                self.assertIs(created.code, PoolResultCode.OK)
                opened = SharedPool.open_pool(name)
                self.assertIs(opened.code, PoolResultCode.OK)
                self.assertEqual(opened.descriptor, created.descriptor)
                descriptor_box[0] = opened.descriptor

            dist.broadcast_object_list(descriptor_box, src=0)
            descriptor = descriptor_box[0]
            result = SharedPool.attach_pool(descriptor, self.comm_rank)
            self.assertIs(result.code, PoolResultCode.OK)
            attachment = result.attachment
            self.assertEqual(attachment.tp_rank, self.comm_rank)
            entries = attachment.data_pools
            self.assertEqual(
                [item.registration.name for item in entries],
                [f"host-r{self.comm_rank}"],
            )
            pool = entries[0].data_pool
            entries.clear()

            copy_tensor_raw(expected, pool, 0, to_device=False)
            dist.barrier()

            reloaded = torch.empty_like(expected)
            copy_tensor_raw(reloaded, pool, 0, to_device=True)
            self.assertEqual(expected.cpu(), reloaded.cpu())
            dist.barrier()
        finally:
            pool = None
            if attachment is not None:
                SharedPool.detach_pool(attachment)
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
