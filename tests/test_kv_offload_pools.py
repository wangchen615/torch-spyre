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

"""Byte-movement tests for copy_kv_page_raw / copy_tensor_raw over each pool kind.

The same page-level contract is checked against a SharedHostPool (slots in host
POSIX SHM) and a SharedMarvellPool (slots in a Marvell card's BAR2, reached
peer-to-peer from Spyre HBM). copy_kv_page_raw only sees a flex::SharedPool, so
any difference between the two would be a transport defect in flex.

Every assertion is bitwise on the raw fp16 bit patterns, not a tolerance: a
page reloaded at the wrong offset, from the wrong slot, or not at all must fail.

Gating:
- A real card is required (FLEX_DEVICE must not be MOCK*): the mock accepts a
  DMA without moving a byte, so these would pass vacuously.
- The Marvell cases additionally need FLEX_TEST_PCI_BDF (the Marvell card's
  BDF) and a Spyre card that reaches it peer-to-peer. FLEX_TEST_BAR2_OFFSET
  (default 0) selects the BAR2 window, as in flex's own tests.

Two device write-path properties shape the helpers (both measured on hardware,
neither an offload defect): host->device writes are not bit-preserving, so the
reference is always the device's own image read back after a write; and only
whole-tensor writes are accepted (a page-slice copy_ is rejected, a page-slice
zero_() wipes the whole allocation), so every write rewrites the full cache.
"""

import os
import unittest

import torch
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_spyre._C import (  # type: ignore[attr-defined]
    SharedHostPool,
    SharedMarvellPool,
    SpyreTensorLayout,
    copy_kv_page_raw,
    copy_tensor_raw,
    get_composite_address,
    get_device_dtype,
    get_elem_in_stick,
)

DT = torch.float16
NUM_BLOCKS = 8
BLOCK_SIZE = 128
NUM_KV_HEADS = 8
HEAD_SIZE = 128

MARVELL_BDF = os.environ.get("FLEX_TEST_PCI_BDF", "")
BAR2_OFFSET = int(os.environ.get("FLEX_TEST_BAR2_OFFSET", "0"), 0)
REAL_DEVICE = not os.environ.get("FLEX_DEVICE", "").upper().startswith("MOCK")

requires_hardware = unittest.skipUnless(
    REAL_DEVICE, "byte fidelity requires a real device (FLEX_DEVICE is MOCK*)"
)


def token_major(num_blocks=NUM_BLOCKS):
    """[B, S, H, D] cache in spyre-inference's slot-major layout."""
    eps = get_elem_in_stick(DT)
    sticks = HEAD_SIZE // eps
    lay = SpyreTensorLayout(
        device_size=[num_blocks * BLOCK_SIZE, NUM_KV_HEADS, sticks, eps],
        stride_map=[NUM_KV_HEADS * sticks * eps, sticks * eps, eps, 1],
        device_dtype=get_device_dtype(DT),
    )
    shape = (num_blocks, BLOCK_SIZE, NUM_KV_HEADS, HEAD_SIZE)
    return torch.zeros(shape, dtype=DT).to("spyre", device_layout=lay)


def head_major(num_blocks=NUM_BLOCKS):
    """[B, H, S, D] cache in spyre-inference's head-major layout."""
    eps = get_elem_in_stick(DT)
    sticks = HEAD_SIZE // eps
    lay = SpyreTensorLayout(
        device_size=[num_blocks * NUM_KV_HEADS, BLOCK_SIZE, sticks, eps],
        stride_map=[BLOCK_SIZE * HEAD_SIZE, HEAD_SIZE, eps, 1],
        device_dtype=get_device_dtype(DT),
    )
    shape = (num_blocks, NUM_KV_HEADS, BLOCK_SIZE, HEAD_SIZE)
    return torch.zeros(shape, dtype=DT).to("spyre", device_layout=lay)


LAYOUTS = {"token_major": token_major, "head_major": head_major}


def pattern(shape, block_id, seed):
    """Distinct per (block, seed), non-constant, and free of fp16 subnormals
    (the device write path halves a subnormal on every rewrite)."""
    g = torch.Generator().manual_seed(seed * 1000 + block_id)
    vals = torch.randn(shape, generator=g, dtype=torch.float32)
    tiny = vals.abs() < 2.0**-13
    vals = torch.where(tiny, torch.full_like(vals, 0.5), vals)
    return vals.to(DT)


def bits(t):
    return t.to("cpu").contiguous().view(torch.int16)


class _PoolRoundTrip:
    """Mixin: the tests, run against the pool returned by make_pool()."""

    POOL_CLS: type

    def make_pool(self, name, num_slots, slot_bytes):
        raise NotImplementedError

    def setUp(self):
        super().setUp()
        torch.spyre._impl._lazy_init()
        self.names = []
        self.pools = []

    def tearDown(self):
        # Drop the handles only after every copy has completed.
        torch.spyre.synchronize()
        self.pools.clear()
        for name in self.names:
            self.POOL_CLS.unlink_by_name(name)
        super().tearDown()

    def pool(self, num_slots, slot_bytes):
        name = f"tkvp_{os.getpid()}_{self._testMethodName}_{len(self.names)}"
        self.POOL_CLS.unlink_by_name(name)
        self.names.append(name)
        p = self.make_pool(name, num_slots, slot_bytes)
        self.pools.append(p)
        return p

    # ---- helpers ------------------------------------------------------------

    def fill(self, cache, seeds):
        """Write every page in one whole-tensor copy; return the device image.

        Blocks not in `seeds` get a default pattern, so no neighbour is left
        zero (a spill into a zero page would be invisible)."""
        nb = cache.shape[0]
        host = torch.empty(tuple(cache.shape), dtype=DT)
        for b in range(nb):
            host[b] = pattern(tuple(cache.shape[1:]), b, seeds.get(b, 900 + b))
        cache.copy_(host)
        torch.spyre.synchronize()
        dev = cache.to("cpu")
        return [dev[b].clone() for b in range(nb)]

    def poison(self, cache, blocks):
        """Zero only `blocks` via a whole-tensor rewrite; return the new image."""
        host = cache.to("cpu").clone()
        for b in blocks:
            host[b].zero_()
        cache.copy_(host)
        torch.spyre.synchronize()
        dev = cache.to("cpu")
        return [dev[b].clone() for b in range(cache.shape[0])]

    def assert_bit_exact(self, actual, expected, what):
        a, e = bits(actual), bits(expected)
        if torch.equal(a, e):
            return
        bad = (a != e).sum().item()
        idx = (a != e).nonzero()[0].tolist()
        self.fail(f"{what}: {bad}/{a.numel()} fp16 values differ; first at {idx}")

    def page_bytes(self, cache):
        return get_composite_address(cache).total_size // cache.shape[0]

    # ---- tests --------------------------------------------------------------

    def test_round_trip_is_bit_exact(self):
        """Offload page k, destroy it on the device, reload into page k."""
        for kind, make in LAYOUTS.items():
            with self.subTest(layout=kind):
                cache = make()
                pool = self.pool(2, self.page_bytes(cache))
                expected = self.fill(cache, {3: 1})

                copy_kv_page_raw(cache, 3, pool, 1, to_device=False)
                after = self.poison(cache, [3])
                self.assertFalse(bits(after[3]).any(), "poisoning page 3 failed")

                copy_kv_page_raw(cache, 3, pool, 1, to_device=True)
                image = cache.to("cpu")
                self.assert_bit_exact(image[3], expected[3], f"{kind} page 3")

    def test_reload_relocates_to_another_block(self):
        """A page may come back into a different block; the source is left
        intact and the destination's previous content is fully replaced."""
        for kind, make in LAYOUTS.items():
            with self.subTest(layout=kind):
                cache = make()
                pool = self.pool(2, self.page_bytes(cache))
                expected = self.fill(cache, {1: 2, 6: 3})

                copy_kv_page_raw(cache, 1, pool, 0, to_device=False)
                copy_kv_page_raw(cache, 6, pool, 0, to_device=True)
                image = cache.to("cpu")
                self.assert_bit_exact(image[6], expected[1], f"{kind} 1 -> 6")
                self.assert_bit_exact(image[1], expected[1], f"{kind} source")

    def test_siblings_untouched(self):
        """A wrong page stride or length would spill into a neighbour."""
        for kind, make in LAYOUTS.items():
            with self.subTest(layout=kind):
                cache = make()
                pool = self.pool(2, self.page_bytes(cache))
                target = 4
                expected_target = self.fill(
                    cache, {b: 10 + b for b in range(NUM_BLOCKS)}
                )[target]

                copy_kv_page_raw(cache, target, pool, 0, to_device=False)
                expected = self.poison(cache, [target])
                expected[target] = expected_target
                copy_kv_page_raw(cache, target, pool, 0, to_device=True)

                image = cache.to("cpu")
                for b in range(NUM_BLOCKS):
                    self.assert_bit_exact(
                        image[b], expected[b], f"{kind} page {b} (target={target})"
                    )

    def test_slots_are_independent(self):
        """Pages offloaded to different slots come back from their own slot."""
        cache = token_major()
        pool = self.pool(3, self.page_bytes(cache))
        expected = self.fill(cache, {0: 20, 5: 21, 7: 22})

        for slot, block in enumerate((0, 5, 7)):
            copy_kv_page_raw(cache, block, pool, slot, to_device=False)
        self.fill(cache, {0: 30, 5: 31, 7: 32})  # overwrite the device copies
        # Reload in a permuted order and into permuted blocks.
        copy_kv_page_raw(cache, 7, pool, 0, to_device=True)
        copy_kv_page_raw(cache, 0, pool, 1, to_device=True)
        copy_kv_page_raw(cache, 5, pool, 2, to_device=True)

        image = cache.to("cpu")
        self.assert_bit_exact(image[7], expected[0], "slot 0 -> block 7")
        self.assert_bit_exact(image[0], expected[5], "slot 1 -> block 0")
        self.assert_bit_exact(image[5], expected[7], "slot 2 -> block 5")

    def test_non_blocking_round_trip(self):
        cache = token_major()
        pool = self.pool(2, self.page_bytes(cache))
        expected = self.fill(cache, {2: 40})

        copy_kv_page_raw(cache, 2, pool, 0, to_device=False, non_blocking=True)
        torch.spyre.synchronize()
        self.poison(cache, [2])
        copy_kv_page_raw(cache, 2, pool, 0, to_device=True, non_blocking=True)
        torch.spyre.synchronize()

        self.assert_bit_exact(cache.to("cpu")[2], expected[2], "non-blocking page 2")

    def test_copy_tensor_raw_whole_cache(self):
        """copy_tensor_raw moves a whole (aligned) allocation through a slot."""
        cache = token_major()
        pool = self.pool(1, get_composite_address(cache).total_size)
        expected = self.fill(cache, {})

        copy_tensor_raw(cache, pool, 0, to_device=False)
        self.poison(cache, range(NUM_BLOCKS))
        copy_tensor_raw(cache, pool, 0, to_device=True)

        image = cache.to("cpu")
        for b in range(NUM_BLOCKS):
            self.assert_bit_exact(image[b], expected[b], f"whole cache page {b}")

    def test_slot_out_of_range_rejected(self):
        cache = token_major()
        pool = self.pool(2, self.page_bytes(cache))
        with self.assertRaises(IndexError):  # std::out_of_range
            copy_kv_page_raw(cache, 0, pool, 2, to_device=False)


@requires_hardware
class TestKvOffloadSharedHostPool(_PoolRoundTrip, TestCase):
    POOL_CLS = SharedHostPool

    def make_pool(self, name, num_slots, slot_bytes):
        return SharedHostPool.create_or_attach(name, num_slots, slot_bytes)


@requires_hardware
@unittest.skipUnless(MARVELL_BDF, "FLEX_TEST_PCI_BDF (Marvell card BDF) not set")
class TestKvOffloadSharedMarvellPool(_PoolRoundTrip, TestCase):
    POOL_CLS = SharedMarvellPool

    def setUp(self):
        super().setUp()
        self.next_offset = BAR2_OFFSET

    def make_pool(self, name, num_slots, slot_bytes):
        # Pools created within one test can be alive together, and flex does
        # not detect overlapping BAR2 windows, so give each its own window.
        offset = self.next_offset
        pool = SharedMarvellPool.create_or_attach(
            name, MARVELL_BDF, num_slots, slot_bytes, bar_offset=offset
        )
        self.next_offset = offset + pool.total_bytes()
        return pool


if __name__ == "__main__":
    run_tests()
