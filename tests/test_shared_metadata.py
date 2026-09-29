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
import sys
import threading
import time
from queue import Empty

import pytest
import torch
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


def test_blocked_evict_releases_gil():
    if torch.spyre.is_initialized():
        pytest.skip("requires a fresh process so the child can open the same card")

    name = f"test_blocked_evict_releases_gil.{os.getpid()}"
    pool_name = f"{name}.pool"
    SharedMetadata.unlink_by_name(name)
    SharedHostPool.unlink_by_name(pool_name)

    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    result_queue = context.Queue()
    process = context.Process(
        target=_run_gil_progress_test,
        args=(name, ready, result_queue),
    )
    try:
        process.start()
        if not ready.wait(60):
            pytest.fail("child did not finish shared-metadata setup")
        process.join(10)
        if process.is_alive():
            pytest.fail("child deadlocked while eviction waited for a read pin")
        assert process.exitcode == 0
        try:
            result = result_queue.get(timeout=2)
        except Empty as error:
            pytest.fail(f"child returned no GIL progress result: {error}")

        assert result[0] == "ok", result
        _, blocked_before_release, progress, worker_alive, evicted = result
        assert blocked_before_release
        assert progress == ["main-thread-ran"]
        assert not worker_alive
        assert evicted == [True]
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        result_queue.close()
        result_queue.join_thread()
        SharedMetadata.unlink_by_name(name)
        SharedHostPool.unlink_by_name(pool_name)


def _run_config_snapshot_test(name, result_queue):
    original_pool_name = f"{name}.pool-a"
    mutated_pool_name = f"{name}.pool-b"
    metadata = None
    try:
        creation_pool_config = SharedDataPoolConfig(
            original_pool_name,
            SharedPoolKind.HOST,
            1,
            128,
            CompatibilityDescriptor(1, [1]),
        )
        creation_config = SharedMetadataConfig(
            1,
            [creation_pool_config],
            SharedMetadataCapacity(1, 1, 2),
        )
        entered_creation = threading.Event()
        creation_result = []

        def profile_creation(frame, event, arg):
            if event == "c_call" and getattr(arg, "__name__", "") == "create_or_attach":
                entered_creation.set()

        def create_metadata():
            sys.setprofile(profile_creation)
            try:
                creation_result.append(
                    ("ok", SharedMetadata.create_or_attach(name, creation_config))
                )
            except BaseException as error:
                creation_result.append(("error", repr(error)))
            finally:
                sys.setprofile(None)

        creation_worker = threading.Thread(target=create_metadata)
        creation_worker.start()
        if not entered_creation.wait(5):
            raise AssertionError("creation worker did not enter the binding")
        creation_config.pools = []
        creation_worker.join(30)
        if creation_worker.is_alive():
            raise AssertionError("creation worker deadlocked")
        if not creation_result or creation_result[0][0] != "ok":
            raise AssertionError(f"metadata creation failed: {creation_result}")
        metadata = creation_result[0][1]
        created_pool = metadata.find_pool(original_pool_name)
        if created_pool is None:
            raise AssertionError(
                "creation did not use the entry configuration snapshot"
            )
        if not metadata.retire_pool(created_pool.pool_ref):
            raise AssertionError(
                "created pool could not be retired for registration test"
            )

        config = SharedDataPoolConfig(
            original_pool_name,
            SharedPoolKind.HOST,
            1,
            128,
            CompatibilityDescriptor(1, [1]),
        )
        entered_call = threading.Event()
        registration_result = []

        def profile_call(frame, event, arg):
            if (
                event == "c_call"
                and getattr(arg, "__name__", "") == "register_or_attach_pool"
            ):
                entered_call.set()

        def register_pool():
            sys.setprofile(profile_call)
            try:
                registration_result.append(
                    ("ok", metadata.register_or_attach_pool(config).name)
                )
            except BaseException as error:
                registration_result.append(("error", repr(error)))
            finally:
                sys.setprofile(None)

        worker = threading.Thread(target=register_pool)
        worker.start()
        if not entered_call.wait(5):
            raise AssertionError("registration worker did not enter the binding")
        config.name = mutated_pool_name
        worker.join(30)
        result_queue.put(
            (
                "ok",
                creation_worker.is_alive(),
                worker.is_alive(),
                registration_result,
                metadata.find_pool(original_pool_name) is not None,
                metadata.find_pool(mutated_pool_name) is not None,
            )
        )
    except BaseException as error:
        result_queue.put(("error", repr(error)))
    finally:
        metadata = None
        SharedMetadata.unlink_by_name(name)


def test_mutable_config_is_snapshotted_before_gil_release():
    if torch.spyre.is_initialized():
        pytest.skip("requires a fresh process so the child can open the card")

    name = f"test_mutable_config_is_snapshotted_before_gil_release.{os.getpid()}"
    SharedMetadata.unlink_by_name(name)
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_run_config_snapshot_test,
        args=(name, result_queue),
    )
    try:
        process.start()
        process.join(60)
        if process.is_alive():
            pytest.fail("child deadlocked during pool registration")
        assert process.exitcode == 0
        try:
            result = result_queue.get(timeout=2)
        except Empty as error:
            pytest.fail(f"child returned no config snapshot result: {error}")

        assert result[0] == "ok", result
        (
            _,
            creation_worker_alive,
            worker_alive,
            registration_result,
            found_original,
            found_mutated,
        ) = result
        assert not creation_worker_alive
        assert not worker_alive
        assert registration_result == [("ok", f"{name}.pool-a")]
        assert found_original
        assert not found_mutated
    finally:
        if process.is_alive():
            process.terminate()
            process.join(5)
        result_queue.close()
        result_queue.join_thread()
        SharedMetadata.unlink_by_name(name)


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
        key = CompatibleBlockKey(registered.compatibility, 0x2501)
        old_reservation = md.claim(registered.pool_ref, key)
        self.assertIsInstance(old_reservation, Reservation)
        md.publish(old_reservation, [ChunkDescriptorEntry(0, 128)])
        old_entry = md.lookup(key)
        self.assertIsNotNone(old_entry)
        self.assertTrue(md.evict(old_entry))

        new_reservation = md.claim(registered.pool_ref, key)
        self.assertIsInstance(new_reservation, Reservation)
        self.assertEqual(new_reservation.slot.slot_id, old_entry.slot.slot_id)
        self.assertNotEqual(
            new_reservation.slot.slot_version, old_entry.slot.slot_version
        )
        md.publish(new_reservation, [ChunkDescriptorEntry(0, 128)])

        self.assertIsNone(md.pin_read(old_entry))
        self.assertFalse(md.evict(old_entry))
        new_entry = md.lookup(key)
        self.assertIsNotNone(new_entry)
        pin = md.pin_read(new_entry)
        self.assertIsInstance(pin, SlotReadPin)
        del pin

    def test_stale_pool_row_references_cannot_access_replacement(self):
        name = f"{self.id()}.{os.getpid()}"
        SharedMetadata.unlink_by_name(name)
        self.addCleanup(SharedMetadata.unlink_by_name, name)
        metadata = SharedMetadata.create_or_attach(
            name,
            SharedMetadataConfig(1, [], SharedMetadataCapacity(1, 1, 1)),
        )
        config = SharedDataPoolConfig(
            f"{name}.pool",
            SharedPoolKind.HOST,
            1,
            128,
            CompatibilityDescriptor(1, [1]),
        )
        original = metadata.register_or_attach_pool(config)
        old_key = CompatibleBlockKey(original.compatibility, 0x2511)
        reservation = metadata.claim(original.pool_ref, old_key)
        self.assertIsInstance(reservation, Reservation)
        metadata.publish(reservation, [ChunkDescriptorEntry(0, 128)])
        old_entry = metadata.lookup(old_key)
        self.assertIsNotNone(old_entry)

        self.assertTrue(metadata.retire_pool(original.pool_ref))
        replacement = metadata.register_or_attach_pool(config)
        self.assertEqual(replacement.pool_ref.pool_id, original.pool_ref.pool_id)
        self.assertNotEqual(
            replacement.pool_ref.pool_version,
            original.pool_ref.pool_version,
        )

        fresh_key = CompatibleBlockKey(replacement.compatibility, 0x2512)
        self.assertIsNone(metadata.resolve_pool(original.pool_ref))
        self.assertIsNone(metadata.pin_read(old_entry))
        self.assertFalse(metadata.evict(old_entry))
        self.assertIsInstance(metadata.claim(original.pool_ref, fresh_key), Unavailable)
        fresh_reservation = metadata.claim(replacement.pool_ref, fresh_key)
        self.assertIsInstance(fresh_reservation, Reservation)
        metadata.abort(fresh_reservation)

    def test_stale_metadata_references_cannot_access_recreated_directory(self):
        name = f"{self.id()}.{os.getpid()}"
        SharedMetadata.unlink_by_name(name)
        self.addCleanup(SharedMetadata.unlink_by_name, name)
        config = self.make_config(name, slots=1)
        old_metadata = SharedMetadata.create_or_attach(name, config)
        old_registered = old_metadata.find_pool(f"{name}.pool")
        self.assertIsNotNone(old_registered)
        old_key = CompatibleBlockKey(old_registered.compatibility, 0x2521)
        old_reservation = old_metadata.claim(old_registered.pool_ref, old_key)
        self.assertIsInstance(old_reservation, Reservation)
        old_metadata.publish(old_reservation, [ChunkDescriptorEntry(0, 128)])
        old_entry = old_metadata.lookup(old_key)
        self.assertIsNotNone(old_entry)
        old_version = old_metadata.version()

        del old_reservation
        SharedMetadata.unlink_by_name(name)
        del old_metadata
        metadata = SharedMetadata.create_or_attach(name, config)
        registered = metadata.find_pool(f"{name}.pool")
        self.assertIsNotNone(registered)
        self.assertNotEqual(metadata.version(), old_version)

        fresh_key = CompatibleBlockKey(registered.compatibility, 0x2521)
        self.assertIsNone(metadata.lookup(old_key))
        self.assertIsNone(metadata.resolve_pool(old_registered.pool_ref))
        self.assertIsNone(metadata.pin_read(old_entry))
        self.assertFalse(metadata.evict(old_entry))
        self.assertIsInstance(
            metadata.claim(old_registered.pool_ref, fresh_key), Unavailable
        )
        fresh_reservation = metadata.claim(registered.pool_ref, fresh_key)
        self.assertIsInstance(fresh_reservation, Reservation)
        metadata.abort(fresh_reservation)

    def test_protocol_objects_do_not_expose_raw_addresses(self):
        md, registered = self.create_metadata(slots=1)
        pool = md.resolve_pool(registered.pool_ref)
        key = CompatibleBlockKey(registered.compatibility, 0x2601)
        reservation = md.claim(registered.pool_ref, key)
        self.assertIsInstance(reservation, Reservation)
        md.publish(reservation, [ChunkDescriptorEntry(0, 128)])
        entry = md.lookup(key)
        self.assertIsNotNone(entry)
        pin = md.pin_read(entry)
        self.assertIsNotNone(pin)

        for value in (md, pool, reservation, entry, pin):
            self.assertFalse(hasattr(value, "slot_ptr"))
            self.assertFalse(hasattr(value, "host_address"))
            self.assertFalse(hasattr(value, "device_address"))

        del pin


if __name__ == "__main__":
    run_tests()
