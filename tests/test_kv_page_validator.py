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

"""Metadata tests for copy_kv_page_raw's KV-page contract.

These are the no-hardware half of the M1 test matrix: they check that the
validator accepts exactly the production KV layouts and rejects everything
else BEFORE any DMA is enqueued. Byte-movement tests live in test_kv_offload.py
and need a card.

The layouts are built the way spyre-inference builds them, so a drift in
slot_major_kv_layout / head_major_kv_layout shows up here.
"""

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_spyre._C import (  # type: ignore[attr-defined]
    SharedHostPool,
    SpyreTensorLayout,
    copy_kv_page_raw,
    get_composite_address,
    get_device_dtype,
    get_elem_in_stick,
)

DT = torch.float16


def slot_major_layout(num_slots, num_kv_heads, head_size, dtype=DT):
    """Mirror of spyre_inference...ops.layout.slot_major_kv_layout."""
    eps = get_elem_in_stick(dtype)
    sticks = (head_size + eps - 1) // eps
    return SpyreTensorLayout(
        device_size=[num_slots, num_kv_heads, sticks, eps],
        stride_map=[num_kv_heads * sticks * eps, sticks * eps, eps, 1],
        device_dtype=get_device_dtype(dtype),
    )


def head_major_layout(num_pages, block_size, head_size, dtype=DT):
    """Mirror of spyre_inference...ops.layout.head_major_kv_layout."""
    eps = get_elem_in_stick(dtype)
    sticks = (head_size + eps - 1) // eps
    return SpyreTensorLayout(
        device_size=[num_pages, block_size, sticks, eps],
        stride_map=[block_size * head_size, head_size, eps, 1],
        device_dtype=get_device_dtype(dtype),
    )


class TestKvPageValidator(TestCase):
    def setUp(self):
        super().setUp()
        torch.spyre._impl._lazy_init()
        self.pool = None

    def tearDown(self):
        if self.pool is not None:
            del self.pool
        super().tearDown()

    def _pool_for(self, cache, num_blocks, name):
        """A pool with one slot the size of one page."""
        ca = get_composite_address(cache)
        page_bytes = ca.total_size // num_blocks
        SharedHostPool.unlink_by_name(name)
        return SharedHostPool.create_or_attach(name, 2, page_bytes)

    def _token_major(self, num_blocks=8, block_size=128, num_kv_heads=8, head_size=128):
        lay = slot_major_layout(num_blocks * block_size, num_kv_heads, head_size)
        shape = (num_blocks, block_size, num_kv_heads, head_size)
        return torch.zeros(shape, dtype=DT).to("spyre", device_layout=lay), num_blocks

    def _head_major(self, num_blocks=8, block_size=128, num_kv_heads=8, head_size=128):
        lay = head_major_layout(num_blocks * num_kv_heads, block_size, head_size)
        shape = (num_blocks, num_kv_heads, block_size, head_size)
        return torch.zeros(shape, dtype=DT).to("spyre", device_layout=lay), num_blocks

    # ---- positive: page geometry is derived correctly ----------------------

    def test_token_major_page_geometry(self):
        for head_size in (64, 128):
            cache, nb = self._token_major(head_size=head_size)
            ca = get_composite_address(cache)
            self.assertEqual(ca.num_chunks, 1)
            page_bytes = ca.total_size // nb
            self.assertEqual(ca.total_size % nb, 0, "cache has inter-page padding")
            self.assertEqual(page_bytes % 128, 0, "page not 128B aligned")
            expected = 128 * 8 * head_size * 2  # block*heads*D*fp16
            self.assertEqual(page_bytes, expected, f"head_size={head_size}")

    def test_head_major_page_geometry(self):
        for head_size in (64, 128):
            cache, nb = self._head_major(head_size=head_size)
            ca = get_composite_address(cache)
            self.assertEqual(ca.num_chunks, 1)
            self.assertEqual(ca.total_size % nb, 0)
            self.assertEqual((ca.total_size // nb) % 128, 0)

    # ---- rejections: all must raise before any DMA -------------------------

    def test_rejects_generic_layout(self):
        """A plain .to('spyre') gets the generic tiled layout, rank 5."""
        t = torch.zeros((8, 128, 8, 128), dtype=DT).to("spyre")
        pool = self._pool_for_generic(t)
        with self.assertRaisesRegex(RuntimeError, "rank-4 device layout"):
            copy_kv_page_raw(t, 0, pool, 0, False)

    def _pool_for_generic(self, t):
        ca = get_composite_address(t)
        SharedHostPool.unlink_by_name("kvval_generic")
        return SharedHostPool.create_or_attach("kvval_generic", 2, ca.total_size)

    def test_rejects_page_view(self):
        """A page view has nonzero storage_offset; pass the cache + block_id."""
        cache, nb = self._token_major()
        pool = self._pool_for(cache, nb, "kvval_view")
        page = cache[1]
        with self.assertRaisesRegex(RuntimeError, "rank-4 KV cache"):
            copy_kv_page_raw(page, 0, pool, 0, False)

    def test_rejects_out_of_range_block(self):
        cache, nb = self._token_major()
        pool = self._pool_for(cache, nb, "kvval_oob")
        with self.assertRaisesRegex(RuntimeError, "out of range"):
            copy_kv_page_raw(cache, nb, pool, 0, False)

    def test_rejects_unaligned_head_size(self):
        """head_size not a multiple of the stick width: device image is padded,
        so no single host-derived range describes a page."""
        cache, nb = self._token_major(head_size=96)
        pool = self._pool_for(cache, nb, "kvval_pad")
        with self.assertRaisesRegex(RuntimeError, "not a multiple of the stick width"):
            copy_kv_page_raw(cache, 0, pool, 0, False)

    def test_rejects_cpu_tensor(self):
        cache, nb = self._token_major()
        pool = self._pool_for(cache, nb, "kvval_cpu")
        cpu = torch.zeros((8, 128, 8, 128), dtype=DT)
        with self.assertRaisesRegex(RuntimeError, "Spyre"):
            copy_kv_page_raw(cpu, 0, pool, 0, False)

    def test_rejects_wrong_rank(self):
        lay = slot_major_layout(8 * 128, 8, 128)
        t = torch.zeros((8 * 128, 8, 128), dtype=DT).to("spyre", device_layout=lay)
        ca = get_composite_address(t)
        SharedHostPool.unlink_by_name("kvval_rank")
        pool = SharedHostPool.create_or_attach("kvval_rank", 2, ca.total_size // 8)
        with self.assertRaisesRegex(RuntimeError, "rank-4"):
            copy_kv_page_raw(t, 0, pool, 0, False)

    def test_rejects_prefix_and_offset_views(self):
        """Views keep the whole cache's device layout (spyre_views.cpp copies
        spyre_layout verbatim) while shrinking size(0). A zero-offset prefix
        view is the dangerous one: storage_offset() is 0, so only the exact
        num_blocks*inner == device_size[0] identity catches it."""
        cache, nb = self._token_major()
        pool = self._pool_for(cache, nb, "kvval_slices")
        for lo, hi in ((0, 4), (2, 6), (4, 8)):
            view = cache[lo:hi]
            self.assertEqual(view.dim(), 4)
            with self.subTest(slice=(lo, hi)):
                with self.assertRaisesRegex(RuntimeError, "is not num_blocks"):
                    copy_kv_page_raw(view, 0, pool, 0, False)

    def test_rejects_non_contiguous(self):
        cache, nb = self._token_major()
        pool = self._pool_for(cache, nb, "kvval_perm")
        perm = cache.permute(0, 2, 1, 3)
        self.assertFalse(perm.is_contiguous())
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            copy_kv_page_raw(perm, 0, pool, 0, False)

    def test_page_bytes_matches_geometry_across_shapes(self):
        """page_bytes must equal one page's worth of logical bytes, and every
        block_id in range must validate."""
        for nb, bs, h, d in ((8, 128, 8, 128), (4, 64, 4, 64),
                             (16, 128, 2, 128), (3, 128, 8, 128)):
            cache, _ = self._token_major(nb, bs, h, d)
            ca = get_composite_address(cache)
            page_bytes = ca.total_size // nb
            self.assertEqual(page_bytes, bs * h * d * 2, f"B={nb} S={bs} H={h} D={d}")
            pool = self._pool_for(cache, nb, "kvval_geom")
            for block_id in range(nb):
                copy_kv_page_raw(cache, block_id, pool, 0, False)

    def test_head_major_page_bytes_matches_geometry(self):
        for nb, bs, h, d in ((8, 128, 8, 128), (4, 64, 4, 64)):
            cache, _ = self._head_major(nb, bs, h, d)
            ca = get_composite_address(cache)
            self.assertEqual(ca.total_size // nb, h * bs * d * 2)
            pool = self._pool_for(cache, nb, "kvval_hmgeom")
            for block_id in range(nb):
                copy_kv_page_raw(cache, block_id, pool, 0, False)


if __name__ == "__main__":
    run_tests()
