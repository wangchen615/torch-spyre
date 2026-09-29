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

"""Binding tests for SharedMarvellPool.

A SharedMarvellPool's slots are a window of a Marvell card's PCI BAR2. These
tests only open pools (which reads sysfs and creates the host-side "<name>.ctl"
control segment); none of them issues a DMA, so they run on any Spyre card.

They need the Marvell card's PCI BDF in FLEX_TEST_PCI_BDF, the same variable
flex's own SharedMarvellPool tests use, and are skipped without it.
"""

import os
import unittest

from torch.testing._internal.common_utils import TestCase, run_tests

from torch_spyre._C import (  # type: ignore[attr-defined]
    SharedHostPool,
    SharedMarvellPool,
    SharedPool,
)

MARVELL_BDF = os.environ.get("FLEX_TEST_PCI_BDF", "")
DEVICE_ALIGNMENT = 128


def read_bar2(bdf):
    """(start, size) of BAR2 from /sys/bus/pci/devices/<bdf>/resource.

    Line i of that file is "start end flags" for resource i, in hex. This is
    the same source flex reads, so bus_address must agree with it exactly.
    """
    with open(f"/sys/bus/pci/devices/{bdf}/resource") as f:
        start, end, _flags = (int(x, 16) for x in f.read().splitlines()[2].split())
    return start, end - start + 1


def ctl_exists(name):
    return os.path.exists(f"/dev/shm/{name.lstrip('/')}.ctl")


@unittest.skipUnless(MARVELL_BDF, "FLEX_TEST_PCI_BDF (Marvell card BDF) not set")
class TestSharedMarvellPool(TestCase):
    def setUp(self):
        super().setUp()
        # Unique per test and process; unlinked on the way in as well as out,
        # so a segment leaked by a crashed run cannot be attached by mistake.
        self.name = f"tsmp_{os.getpid()}_{self._testMethodName}"
        SharedMarvellPool.unlink_by_name(self.name)
        SharedHostPool.unlink_by_name(self.name)

    def tearDown(self):
        SharedMarvellPool.unlink_by_name(self.name)
        SharedHostPool.unlink_by_name(self.name)
        super().tearDown()

    def test_is_a_shared_pool(self):
        pool = SharedMarvellPool.create_or_attach(self.name, MARVELL_BDF, 2, 128)
        self.assertIsInstance(pool, SharedPool)
        self.assertFalse(hasattr(pool, "slot_ptr"))

    def test_geometry(self):
        # slot_bytes is rounded up to DEVICE_ALIGNMENT.
        pool = SharedMarvellPool.create_or_attach(
            self.name, MARVELL_BDF, num_slots=3, slot_bytes=DEVICE_ALIGNMENT + 1
        )
        self.assertEqual(pool.name(), self.name)
        self.assertEqual(pool.slot_count(), 3)
        self.assertEqual(pool.slot_bytes(), 2 * DEVICE_ALIGNMENT)
        self.assertEqual(pool.total_bytes(), 6 * DEVICE_ALIGNMENT)

    def test_bus_address_matches_sysfs_bar2(self):
        bar_start, _ = read_bar2(MARVELL_BDF)
        for offset in (0, 2 * DEVICE_ALIGNMENT, 1 << 30):
            name = f"{self.name}_{offset}"
            SharedMarvellPool.unlink_by_name(name)
            try:
                pool = SharedMarvellPool.create_or_attach(
                    name, MARVELL_BDF, 2, DEVICE_ALIGNMENT, bar_offset=offset
                )
                self.assertEqual(pool.bus_address, bar_start + offset)
                del pool
            finally:
                SharedMarvellPool.unlink_by_name(name)

    def test_bus_address_is_read_only(self):
        pool = SharedMarvellPool.create_or_attach(self.name, MARVELL_BDF, 2, 128)
        with self.assertRaises(AttributeError):
            pool.bus_address = 0

    def test_attach_existing_pool(self):
        creator = SharedMarvellPool.create_or_attach(self.name, MARVELL_BDF, 4, 256)
        attacher = SharedMarvellPool.create_or_attach(self.name, MARVELL_BDF, 4, 256)
        self.assertEqual(attacher.bus_address, creator.bus_address)
        self.assertEqual(attacher.total_bytes(), creator.total_bytes())

    def test_geometry_or_window_mismatch(self):
        _ = SharedMarvellPool.create_or_attach(self.name, MARVELL_BDF, 4, 128)
        with self.assertRaises(RuntimeError):
            SharedMarvellPool.create_or_attach(self.name, MARVELL_BDF, 5, 128)
        with self.assertRaises(RuntimeError):
            SharedMarvellPool.create_or_attach(self.name, MARVELL_BDF, 4, 256)
        with self.assertRaises(RuntimeError):
            SharedMarvellPool.create_or_attach(
                self.name, MARVELL_BDF, 4, 128, bar_offset=DEVICE_ALIGNMENT
            )

    def test_host_pool_under_same_name_rejected(self):
        pool = SharedMarvellPool.create_or_attach(self.name, MARVELL_BDF, 4, 128)
        with self.assertRaises(RuntimeError):
            SharedHostPool.create_or_attach(
                self.name, pool.slot_count(), pool.slot_bytes()
            )

    def test_invalid_arguments(self):
        # std::invalid_argument surfaces as ValueError.
        with self.assertRaises(ValueError):
            SharedMarvellPool.create_or_attach(self.name, MARVELL_BDF, 0, 128)
        with self.assertRaises(ValueError):
            SharedMarvellPool.create_or_attach(self.name, MARVELL_BDF, 4, 0)
        with self.assertRaises(ValueError):
            SharedMarvellPool.create_or_attach(
                self.name, MARVELL_BDF, 4, 128, bar_offset=1
            )
        _, bar_size = read_bar2(MARVELL_BDF)
        with self.assertRaises(ValueError):
            SharedMarvellPool.create_or_attach(
                self.name, MARVELL_BDF, 1, DEVICE_ALIGNMENT, bar_offset=bar_size
            )
        self.assertFalse(ctl_exists(self.name))

    def test_unknown_bdf(self):
        with self.assertRaises(RuntimeError):
            SharedMarvellPool.create_or_attach(self.name, "ffff:ff:1f.7", 4, 128)
        self.assertFalse(ctl_exists(self.name))

    def test_last_handle_unlinks(self):
        pool = SharedMarvellPool.create_or_attach(self.name, MARVELL_BDF, 4, 128)
        self.assertTrue(ctl_exists(self.name))
        # No data segment in /dev/shm: the slots are on the card.
        self.assertFalse(os.path.exists(f"/dev/shm/{self.name}"))
        del pool
        self.assertFalse(ctl_exists(self.name))

    def test_unlink_by_name(self):
        """unlink_by_name frees a name still held by a live handle, so a new
        geometry can be created under it."""
        pool = SharedMarvellPool.create_or_attach(self.name, MARVELL_BDF, 4, 128)
        self.assertTrue(ctl_exists(self.name))
        SharedMarvellPool.unlink_by_name(self.name)
        self.assertFalse(ctl_exists(self.name))
        # The live handle keeps working on its (now nameless) segment.
        self.assertEqual(pool.slot_count(), 4)
        del pool
        recreated = SharedMarvellPool.create_or_attach(self.name, MARVELL_BDF, 2, 256)
        self.assertEqual(recreated.slot_bytes(), 256)
        # Unlinking an absent name is not an error.
        SharedMarvellPool.unlink_by_name(self.name + "_absent")


if __name__ == "__main__":
    run_tests()
