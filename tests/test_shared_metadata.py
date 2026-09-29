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


import os
import multiprocessing
import threading
import time
import unittest
from queue import Empty

from torch.testing._internal.common_utils import TestCase, run_tests

from torch_spyre._C import (  # type: ignore[attr-defined]
    ChunkDescriptorEntry,
    CompatibilityDescriptor,
    CompatibleBlockKey,
    ExistingClaim,
    NoSpace,
    Reservation,
    SharedDataPool,
    SharedDataPoolConfig,
    SharedHostPool,
    SharedMetadata,
    SharedMetadataCapacity,
    SharedMetadataConfig,
    SharedPoolKind,
    SlotReadPin,
    Unavailable,
)


def _run_gil_progress_test(name, ready, result_queue):
    pool_name = f"{name}.pool"
    try:
        compatibility = CompatibilityDescriptor(1, [1])
        config = SharedDataPoolConfig(
            pool_name, SharedPoolKind.HOST, 1, 128, compatibility
        )
        md = SharedMetadata.create_or_attach(
            name, SharedMetadataConfig(1, [config], None)
        )
        registered = md.find_pool(pool_name)
        key = CompatibleBlockKey(registered.compatibility, 0x3001)
        reservation = md.claim(registered.pool_ref, key)
        md.publish(reservation, [ChunkDescriptorEntry(0, 128)])
        entry = md.lookup(key)
        pin = md.pin_read(entry)
        ready.set()

        started = threading.Event()
        finished = threading.Event()
        progress = []
        evicted = []

        def evict_pinned_entry():
            started.set()
            evicted.append(md.evict(entry))
            finished.set()

        worker = threading.Thread(target=evict_pinned_entry)
        worker.start()
        if not started.wait(5):
            raise AssertionError("eviction worker did not start")
        time.sleep(0.1)
        blocked_before_release = not finished.is_set()
        progress.append("main-thread-ran")
        del pin
        worker.join(5)
        result_queue.put(
            (
                "ok",
                blocked_before_release,
                progress,
                worker.is_alive(),
                evicted,
            )
        )
    except BaseException as error:
        result_queue.put(("error", repr(error)))
        ready.set()


class TestSharedMetadataGIL(unittest.TestCase):
    def test_blocked_evict_releases_gil(self):
        name = f"{self.id()}.{os.getpid()}"
        pool_name = f"{name}.pool"
        SharedMetadata.unlink_by_name(name)
        SharedHostPool.unlink_by_name(pool_name)
        self.addCleanup(SharedMetadata.unlink_by_name, name)
        self.addCleanup(SharedHostPool.unlink_by_name, pool_name)

        context = multiprocessing.get_context("spawn")
        ready = context.Event()
        result_queue = context.Queue()
        process = context.Process(
            target=_run_gil_progress_test, args=(name, ready, result_queue)
        )
        process.start()
        if not ready.wait(60):
            process.terminate()
            process.join(5)
            self.fail("child did not finish shared-metadata setup")
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(5)
            self.fail("child deadlocked while eviction waited for a read pin")
        self.assertEqual(process.exitcode, 0)
        try:
            result = result_queue.get(timeout=2)
        except Empty as error:
            self.fail(f"child returned no GIL progress result: {error}")
        finally:
            result_queue.close()
            result_queue.join_thread()

        self.assertEqual(result[0], "ok", result)
        _, blocked_before_release, progress, worker_alive, evicted = result
        self.assertTrue(blocked_before_release)
        self.assertEqual(progress, ["main-thread-ran"])
        self.assertFalse(worker_alive)
        self.assertEqual(evicted, [True])


class TestSharedMetadata(TestCase):
    def make_config(self, test_name: str, *, slots: int = 2) -> SharedMetadataConfig:
        compatibility = CompatibilityDescriptor(1, [0x53, 0x50, 0x59, 0x52, 0x45])
        pool = SharedDataPoolConfig(
            f"{test_name}.pool",
            SharedPoolKind.HOST,
            slots,
            128,
            compatibility,
        )
        return SharedMetadataConfig(1, [pool], None)

    def create_metadata(self, *, slots: int = 2, suffix: str = ""):
        name = f"{self.id()}.{os.getpid()}{suffix}"
        SharedMetadata.unlink_by_name(name)
        self.addCleanup(SharedMetadata.unlink_by_name, name)
        md = SharedMetadata.create_or_attach(name, self.make_config(name, slots=slots))
        registered = md.find_pool(f"{name}.pool")
        self.assertIsNotNone(registered)
        return md, registered

    def test_create_find_and_resolve_pool(self):
        name = f"{self.id()}.{os.getpid()}"
        self.addCleanup(SharedMetadata.unlink_by_name, name)
        md = SharedMetadata.create_or_attach(name, self.make_config(name))
        registered = md.find_pool(f"{name}.pool")
        self.assertIsNotNone(registered)
        self.assertEqual(md.pool_count(), 1)
        self.assertEqual(registered.pool_ref.metadata_version, md.version())
        pool = md.resolve_pool(registered.pool_ref)
        self.assertIsInstance(pool, SharedDataPool)
        self.assertEqual(pool.slot_count(), 2)

    def test_geometry_mismatch_is_rejected(self):
        name = f"{self.id()}.{os.getpid()}"
        self.addCleanup(SharedMetadata.unlink_by_name, name)
        first = SharedMetadata.create_or_attach(name, self.make_config(name, slots=2))
        self.assertIsNotNone(first)
        with self.assertRaises(RuntimeError):
            SharedMetadata.create_or_attach(name, self.make_config(name, slots=3))

    def test_dynamic_register_resolve_and_retire(self):
        name = f"{self.id()}.{os.getpid()}"
        SharedMetadata.unlink_by_name(name)
        self.addCleanup(SharedMetadata.unlink_by_name, name)
        capacity = SharedMetadataCapacity(1, 2, 1)
        md = SharedMetadata.create_or_attach(
            name, SharedMetadataConfig(1, [], capacity)
        )
        compatibility = CompatibilityDescriptor(1, [1])
        config = SharedDataPoolConfig(
            f"{name}.pool", SharedPoolKind.HOST, 2, 128, compatibility
        )
        registered = md.register_or_attach_pool(config)
        resolved = md.resolve_pool(registered.pool_ref)
        self.assertIsInstance(resolved, SharedDataPool)
        self.assertEqual(md.pool_count(), 1)
        self.assertTrue(md.retire_pool(registered.pool_ref))
        self.assertEqual(md.pool_count(), 0)
        self.assertIsNone(md.find_pool(config.name))
        self.assertIsNone(md.resolve_pool(registered.pool_ref))

    def test_claim_publish_lookup_evict(self):
        md, registered = self.create_metadata(slots=2)
        key = CompatibleBlockKey(registered.compatibility, 0x1234)

        reservation = md.claim(registered.pool_ref, key)
        self.assertIsInstance(reservation, Reservation)
        self.assertIsNone(md.lookup(key))

        slot = reservation.slot
        md.publish(reservation, [ChunkDescriptorEntry(0, 128)])
        entry = md.lookup(key)
        self.assertIsNotNone(entry)
        self.assertEqual(entry.slot.pool.pool_id, slot.pool.pool_id)
        self.assertEqual(entry.slot.slot_id, slot.slot_id)

        self.assertTrue(md.evict(entry))
        self.assertIsNone(md.lookup(key))

    def test_abort_returns_reserved_slot(self):
        md, registered = self.create_metadata(slots=1)
        key_a = CompatibleBlockKey(registered.compatibility, 0x2001)
        key_b = CompatibleBlockKey(registered.compatibility, 0x2002)
        reservation_a = md.claim(registered.pool_ref, key_a)
        self.assertIsInstance(reservation_a, Reservation)
        first_slot = reservation_a.slot.slot_id
        md.abort(reservation_a)
        self.assertIsNone(md.lookup(key_a))
        reservation_b = md.claim(registered.pool_ref, key_b)
        self.assertIsInstance(reservation_b, Reservation)
        self.assertEqual(reservation_b.slot.slot_id, first_slot)
        md.abort(reservation_b)

    def test_existing_reserved_and_no_space_are_distinct(self):
        md, registered = self.create_metadata(slots=1)
        key_a = CompatibleBlockKey(registered.compatibility, 0x2101)
        key_b = CompatibleBlockKey(registered.compatibility, 0x2102)
        reservation = md.claim(registered.pool_ref, key_a)
        self.assertIsInstance(reservation, Reservation)
        existing = md.claim(registered.pool_ref, key_a)
        self.assertIsInstance(existing, ExistingClaim)
        self.assertFalse(existing.valid)
        self.assertIsInstance(md.claim(registered.pool_ref, key_b), NoSpace)
        md.abort(reservation)

    def test_foreign_pool_reference_is_unavailable(self):
        md, registered = self.create_metadata(slots=1)
        _, other_registered = self.create_metadata(slots=1, suffix=".other")
        key = CompatibleBlockKey(registered.compatibility, 0x2201)
        result = md.claim(other_registered.pool_ref, key)
        self.assertIsInstance(result, Unavailable)

    def test_failed_publish_leaves_reservation_abortable(self):
        md, registered = self.create_metadata(slots=1)
        key = CompatibleBlockKey(registered.compatibility, 0x2301)
        reservation = md.claim(registered.pool_ref, key)
        self.assertIsInstance(reservation, Reservation)
        with self.assertRaises(ValueError):
            md.publish(reservation, [])
        md.abort(reservation)
        self.assertIsNone(md.lookup(key))

    def test_published_key_is_unique_across_compatible_pools(self):
        name = f"{self.id()}.{os.getpid()}"
        SharedMetadata.unlink_by_name(name)
        self.addCleanup(SharedMetadata.unlink_by_name, name)
        compatibility = CompatibilityDescriptor(1, [1, 2, 3])
        pool_a_config = SharedDataPoolConfig(
            f"{name}.pool-a", SharedPoolKind.HOST, 1, 128, compatibility
        )
        pool_b_config = SharedDataPoolConfig(
            f"{name}.pool-b", SharedPoolKind.HOST, 1, 128, compatibility
        )
        md = SharedMetadata.create_or_attach(
            name,
            SharedMetadataConfig(1, [pool_a_config, pool_b_config], None),
        )
        pool_a = md.find_pool(pool_a_config.name)
        pool_b = md.find_pool(pool_b_config.name)
        self.assertIsNotNone(pool_a)
        self.assertIsNotNone(pool_b)

        key = CompatibleBlockKey(pool_a.compatibility, 0x2401)
        reservation = md.claim(pool_a.pool_ref, key)
        self.assertIsInstance(reservation, Reservation)
        md.publish(reservation, [ChunkDescriptorEntry(0, 128)])

        existing = md.claim(pool_b.pool_ref, key)
        self.assertIsInstance(existing, ExistingClaim)
        self.assertTrue(existing.valid)
        self.assertEqual(existing.slot.pool.pool_id, pool_a.pool_ref.pool_id)
        self.assertEqual(md.lookup(key).slot.pool.pool_id, pool_a.pool_ref.pool_id)

    def test_stale_entry_cannot_pin_or_evict_reused_slot(self):
        md, registered = self.create_metadata(slots=1)
        old_key = CompatibleBlockKey(registered.compatibility, 0x2501)
        old_reservation = md.claim(registered.pool_ref, old_key)
        self.assertIsInstance(old_reservation, Reservation)
        md.publish(old_reservation, [ChunkDescriptorEntry(0, 128)])
        old_entry = md.lookup(old_key)
        self.assertIsNotNone(old_entry)
        self.assertTrue(md.evict(old_entry))

        new_key = CompatibleBlockKey(registered.compatibility, 0x2502)
        new_reservation = md.claim(registered.pool_ref, new_key)
        self.assertIsInstance(new_reservation, Reservation)
        self.assertEqual(new_reservation.slot.slot_id, old_entry.slot.slot_id)
        self.assertNotEqual(
            new_reservation.slot.slot_version, old_entry.slot.slot_version
        )
        md.publish(new_reservation, [ChunkDescriptorEntry(0, 128)])

        self.assertIsNone(md.pin_read(old_entry))
        self.assertFalse(md.evict(old_entry))
        new_entry = md.lookup(new_key)
        self.assertIsNotNone(new_entry)
        pin = md.pin_read(new_entry)
        self.assertIsInstance(pin, SlotReadPin)
        del pin


if __name__ == "__main__":
    run_tests()
