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

"""M1 raw-copy regression coverage through lifecycle-managed pools."""

import torch
from torch.testing._internal.common_utils import TestCase

from shared_pool_utils import LifecyclePool
from torch_spyre._C import copy_tensor_raw
from torch_spyre._C import get_composite_address


class TestKvOffload(TestCase):
    def _kv_offload_reload(self, source, destination, label):
        """Round-trip one tensor through a fixed shared-host slot."""
        slot_bytes = get_composite_address(source).total_size
        managed = LifecyclePool(label, num_slots=1, slot_bytes=slot_bytes)
        try:
            copy_tensor_raw(source, managed.pool, 0, to_device=False)
            copy_tensor_raw(destination, managed.pool, 0, to_device=True)
            self.assertEqual(source.to("cpu"), destination.to("cpu"))
        finally:
            managed.close()

    def test_kv_offload_reload_zeroed(self):
        source = torch.randn(64, device="spyre", dtype=torch.float16)
        self._kv_offload_reload(source, torch.zeros_like(source), self.id())

    def test_kv_offload_reload_different_tensor(self):
        source = torch.randn(64, device="spyre", dtype=torch.float16)
        self._kv_offload_reload(source, torch.empty_like(source), self.id())

    def test_raw_copy_preserves_padded_contiguous_tensor(self):
        for shape in ((64, 96), (2, 32)):
            with self.subTest(shape=shape):
                source = (
                    torch.arange(
                        torch.Size(shape).numel(),
                        dtype=torch.float16,
                    )
                    .reshape(shape)
                    .to("spyre")
                )
                self._kv_offload_reload(source, torch.zeros_like(source), self.id())

    def test_raw_copy_rejects_strided_view(self):
        base = torch.arange(
            4 * 64,
            device="spyre",
            dtype=torch.float16,
        ).reshape(4, 64)
        view = base[::2]
        self.assertFalse(view.is_contiguous())
        managed = LifecyclePool(
            self.id(),
            num_slots=1,
            slot_bytes=get_composite_address(base).total_size,
        )
        try:
            with self.assertRaisesRegex(RuntimeError, "contiguous"):
                copy_tensor_raw(view, managed.pool, 0, to_device=False)
        finally:
            managed.close()

    def test_raw_copy_rejects_cpu_tensor_before_address_lookup(self):
        source = torch.zeros(64, dtype=torch.float16)
        managed = LifecyclePool(self.id(), num_slots=1, slot_bytes=128)
        try:
            with self.assertRaisesRegex(
                RuntimeError,
                "copy_tensor_raw: tensor must be a Spyre tensor",
            ):
                copy_tensor_raw(source, managed.pool, 0, to_device=False)
        finally:
            managed.close()

    def test_normal_copy_tensor_unaffected(self):
        source = torch.randn(10, dtype=torch.float16)
        reloaded = source.to("spyre").to("cpu")
        self.assertEqual(source, reloaded)

    def test_realistic_small_kv_page(self):
        source = torch.randn((2, 16, 8, 128), device="spyre", dtype=torch.float16)
        self._kv_offload_reload(source, torch.zeros_like(source), self.id())

    def test_realistic_large_kv_page(self):
        source = torch.randn((2, 1024, 8, 128), device="spyre", dtype=torch.float16)
        self._kv_offload_reload(source, torch.zeros_like(source), self.id())

    def test_page_view_offload(self):
        cache = torch.arange(4 * 64, device="spyre", dtype=torch.float16).reshape(4, 64)
        page = cache[2]
        managed = LifecyclePool(
            self.id(),
            num_slots=1,
            slot_bytes=page.numel() * page.element_size(),
        )
        try:
            copy_tensor_raw(page, managed.pool, 0, to_device=False)
            reloaded = torch.empty_like(page)
            copy_tensor_raw(reloaded, managed.pool, 0, to_device=True)
            self.assertEqual(page.to("cpu"), reloaded.to("cpu"))
        finally:
            managed.close()
