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

"""Validation tests for the lifecycle-managed KV-page raw-copy contract."""

import torch
from torch.testing._internal.common_utils import TestCase

from shared_pool_utils import LifecyclePool
from torch_spyre._C import SpyreTensorLayout
from torch_spyre._C import copy_kv_page_raw
from torch_spyre._C import get_composite_address
from torch_spyre._C import get_device_dtype
from torch_spyre._C import get_elem_in_stick

DTYPE = torch.float16


def slot_major_layout(num_slots, num_kv_heads, head_size, dtype=DTYPE):
    """Mirror spyre-inference's token-major KV layout."""
    elements_per_stick = get_elem_in_stick(dtype)
    sticks = (head_size + elements_per_stick - 1) // elements_per_stick
    return SpyreTensorLayout(
        device_size=[num_slots, num_kv_heads, sticks, elements_per_stick],
        stride_map=[
            num_kv_heads * sticks * elements_per_stick,
            sticks * elements_per_stick,
            elements_per_stick,
            1,
        ],
        device_dtype=get_device_dtype(dtype),
    )


def head_major_layout(num_pages, block_size, head_size, dtype=DTYPE):
    """Mirror spyre-inference's head-major KV layout."""
    elements_per_stick = get_elem_in_stick(dtype)
    sticks = (head_size + elements_per_stick - 1) // elements_per_stick
    return SpyreTensorLayout(
        device_size=[num_pages, block_size, sticks, elements_per_stick],
        stride_map=[block_size * head_size, head_size, elements_per_stick, 1],
        device_dtype=get_device_dtype(dtype),
    )


class TestKvPageValidator(TestCase):
    def setUp(self):
        super().setUp()
        torch.spyre._impl._lazy_init()
        self.managed_pool = None

    def tearDown(self):
        if self.managed_pool is not None:
            self.managed_pool.close()
        super().tearDown()

    def _pool(self, label, slot_bytes):
        if self.managed_pool is not None:
            self.managed_pool.close()
        self.managed_pool = LifecyclePool(label, num_slots=2, slot_bytes=slot_bytes)
        return self.managed_pool.pool

    def _pool_for(self, cache, num_blocks, label):
        composite = get_composite_address(cache)
        return self._pool(label, composite.total_size // num_blocks)

    def _token_major(self, num_blocks=8, block_size=128, num_kv_heads=8, head_size=128):
        layout = slot_major_layout(num_blocks * block_size, num_kv_heads, head_size)
        shape = (num_blocks, block_size, num_kv_heads, head_size)
        return (
            torch.zeros(shape, dtype=DTYPE).to("spyre", device_layout=layout),
            num_blocks,
        )

    def _head_major(self, num_blocks=8, block_size=128, num_kv_heads=8, head_size=128):
        layout = head_major_layout(num_blocks * num_kv_heads, block_size, head_size)
        shape = (num_blocks, num_kv_heads, block_size, head_size)
        return (
            torch.zeros(shape, dtype=DTYPE).to("spyre", device_layout=layout),
            num_blocks,
        )

    def test_token_major_page_geometry(self):
        for head_size in (64, 128):
            cache, num_blocks = self._token_major(head_size=head_size)
            composite = get_composite_address(cache)
            self.assertEqual(composite.num_chunks, 1)
            page_bytes = composite.total_size // num_blocks
            self.assertEqual(composite.total_size % num_blocks, 0)
            self.assertEqual(page_bytes % 128, 0)
            self.assertEqual(page_bytes, 128 * 8 * head_size * 2)

    def test_head_major_page_geometry(self):
        for head_size in (64, 128):
            cache, num_blocks = self._head_major(head_size=head_size)
            composite = get_composite_address(cache)
            self.assertEqual(composite.num_chunks, 1)
            self.assertEqual(composite.total_size % num_blocks, 0)
            self.assertEqual((composite.total_size // num_blocks) % 128, 0)

    def test_rejects_generic_layout(self):
        cache = torch.zeros((8, 128, 8, 128), dtype=DTYPE).to("spyre")
        pool = self._pool(self.id(), get_composite_address(cache).total_size)
        with self.assertRaisesRegex(RuntimeError, "rank-4 device layout"):
            copy_kv_page_raw(cache, 0, pool, 0, False)

    def test_rejects_page_view(self):
        cache, num_blocks = self._token_major()
        pool = self._pool_for(cache, num_blocks, self.id())
        with self.assertRaisesRegex(RuntimeError, "rank-4 KV cache"):
            copy_kv_page_raw(cache[1], 0, pool, 0, False)

    def test_rejects_out_of_range_block(self):
        cache, num_blocks = self._token_major()
        pool = self._pool_for(cache, num_blocks, self.id())
        with self.assertRaisesRegex(RuntimeError, "out of range"):
            copy_kv_page_raw(cache, num_blocks, pool, 0, False)

    def test_rejects_unaligned_head_size(self):
        cache, num_blocks = self._token_major(head_size=96)
        pool = self._pool_for(cache, num_blocks, self.id())
        with self.assertRaisesRegex(RuntimeError, "not a multiple of the stick width"):
            copy_kv_page_raw(cache, 0, pool, 0, False)

    def test_rejects_cpu_tensor(self):
        cache, num_blocks = self._token_major()
        pool = self._pool_for(cache, num_blocks, self.id())
        cpu_cache = torch.zeros((8, 128, 8, 128), dtype=DTYPE)
        with self.assertRaisesRegex(RuntimeError, "Spyre"):
            copy_kv_page_raw(cpu_cache, 0, pool, 0, False)

    def test_rejects_wrong_rank(self):
        layout = slot_major_layout(8 * 128, 8, 128)
        cache = torch.zeros((8 * 128, 8, 128), dtype=DTYPE).to(
            "spyre", device_layout=layout
        )
        pool = self._pool(self.id(), get_composite_address(cache).total_size // 8)
        with self.assertRaisesRegex(RuntimeError, "rank-4"):
            copy_kv_page_raw(cache, 0, pool, 0, False)

    def test_rejects_prefix_and_offset_views(self):
        cache, num_blocks = self._token_major()
        pool = self._pool_for(cache, num_blocks, self.id())
        for low, high in ((0, 4), (2, 6), (4, 8)):
            view = cache[low:high]
            self.assertEqual(view.dim(), 4)
            with self.subTest(slice=(low, high)):
                with self.assertRaisesRegex(RuntimeError, "is not num_blocks"):
                    copy_kv_page_raw(view, 0, pool, 0, False)

    def test_rejects_non_contiguous(self):
        cache, num_blocks = self._token_major()
        pool = self._pool_for(cache, num_blocks, self.id())
        permuted = cache.permute(0, 2, 1, 3)
        self.assertFalse(permuted.is_contiguous())
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            copy_kv_page_raw(permuted, 0, pool, 0, False)

    def test_page_bytes_matches_geometry_across_shapes(self):
        for num_blocks, block_size, num_heads, head_size in (
            (8, 128, 8, 128),
            (4, 64, 4, 64),
            (16, 128, 2, 128),
            (3, 128, 8, 128),
        ):
            cache, _ = self._token_major(num_blocks, block_size, num_heads, head_size)
            composite = get_composite_address(cache)
            page_bytes = composite.total_size // num_blocks
            self.assertEqual(
                page_bytes,
                block_size * num_heads * head_size * 2,
            )
            pool = self._pool_for(cache, num_blocks, self.id())
            for block_id in range(num_blocks):
                copy_kv_page_raw(cache, block_id, pool, 0, False)
            del pool

    def test_head_major_page_bytes_matches_geometry(self):
        for num_blocks, block_size, num_heads, head_size in (
            (8, 128, 8, 128),
            (4, 64, 4, 64),
        ):
            cache, _ = self._head_major(num_blocks, block_size, num_heads, head_size)
            composite = get_composite_address(cache)
            self.assertEqual(
                composite.total_size // num_blocks,
                num_heads * block_size * head_size * 2,
            )
            pool = self._pool_for(cache, num_blocks, self.id())
            for block_id in range(num_blocks):
                copy_kv_page_raw(cache, block_id, pool, 0, False)
            del pool
