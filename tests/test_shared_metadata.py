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

"""Fixed-membership shared-metadata protocol tests."""

import multiprocessing
import os
import threading
import time
import uuid
from queue import Empty

import pytest
import torch
from torch.testing._internal.common_utils import TestCase

from torch_spyre._C import ChunkDescriptorEntry
from torch_spyre._C import CompatibleBlockKey
from torch_spyre._C import ExistingClaim
from torch_spyre._C import MetadataConfig
from torch_spyre._C import NoSpace
from torch_spyre._C import PoolBackend
from torch_spyre._C import PoolCompatibility
from torch_spyre._C import PoolConfig
from torch_spyre._C import PoolResultCode
from torch_spyre._C import Reservation
from torch_spyre._C import SharedDataPool
from torch_spyre._C import SharedDataPoolConfig
from torch_spyre._C import SharedPool
from torch_spyre._C import SlotReadPin
from torch_spyre._C import Unavailable
from torch_spyre._C import copy_tensor_raw


def _pool_name(label):
    return f"torch-spyre-metadata-{label}-{os.getpid()}-{uuid.uuid4().hex}"


def _compatibility(model="test-model"):
    return PoolCompatibility(model, "float16", 64, 1)


def _config(name, *, slots=2, children=None):
    if children is None:
        children = [
            SharedDataPoolConfig(
                "host-0", PoolBackend.HOST_DRAM, slots, 128, _compatibility()
            )
        ]
    return PoolConfig(name, MetadataConfig(2), children)


def _cleanup_generation(metadata, descriptor):
    SharedPool.detach_pool(metadata)
    result = SharedPool.destroy_pool(descriptor)
    if result.code not in (
        PoolResultCode.OK,
        PoolResultCode.NOT_FOUND,
        PoolResultCode.STALE_GENERATION,
    ):
        raise RuntimeError(f"failed to clean pool: {result.code}: {result.detail}")


def _create_attached(test, label, *, slots=2, children=None):
    name = _pool_name(label)
    created = SharedPool.create_pool(_config(name, slots=slots, children=children))
    test.assertIs(created.code, PoolResultCode.OK)
    attached = SharedPool.attach_pool(created.descriptor)
    test.assertIs(attached.code, PoolResultCode.OK)
    test.assertIsNotNone(attached.metadata)
    test.addCleanup(_cleanup_generation, attached.metadata, created.descriptor)
    return created, attached.metadata


def _registered_and_pool(test, metadata, name="host-0"):
    registered = metadata.find_pool(name)
    test.assertIsNotNone(registered)
    pool = metadata.resolve_pool(registered.pool_ref)
    test.assertIsInstance(pool, SharedDataPool)
    return registered, pool


def _run_gil_progress_test(name, result_queue):
    created = None
    metadata = None
    _pool = None
    try:
        created = SharedPool.create_pool(_config(name, slots=1))
        if created.code is not PoolResultCode.OK:
            raise AssertionError(f"create failed: {created.code}")
        attached = SharedPool.attach_pool(created.descriptor)
        if attached.code is not PoolResultCode.OK or attached.metadata is None:
            raise AssertionError(f"attach failed: {attached.code}")
        metadata = attached.metadata
        registered = metadata.find_pool("host-0")
        _pool = metadata.resolve_pool(registered.pool_ref)
        key = CompatibleBlockKey(registered.compatibility, 0x3001)
        reservation = metadata.claim(registered.pool_ref, key)
        metadata.publish(reservation, [ChunkDescriptorEntry(0, 128)])
        entry = metadata.lookup(key)
        pin = metadata.pin_read(entry)

        started = threading.Event()
        finished = threading.Event()
        evicted = []

        def evict_pinned_entry():
            started.set()
            evicted.append(metadata.evict(entry))
            finished.set()

        worker = threading.Thread(target=evict_pinned_entry)
        worker.start()
        if not started.wait(5):
            raise AssertionError("eviction worker did not start")
        time.sleep(0.1)
        blocked_before_release = not finished.is_set()
        progress = ["main-thread-ran"]
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
    finally:
        _pool = None
        if metadata is not None:
            SharedPool.detach_pool(metadata)
        if created is not None:
            SharedPool.destroy_pool(created.descriptor)


def test_blocked_evict_releases_gil():
    if hasattr(torch, "spyre") and torch.spyre.is_initialized():
        pytest.skip("requires a fresh parent process for isolated card ownership")

    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(
        target=_run_gil_progress_test,
        args=(_pool_name("gil-progress"), result_queue),
    )
    try:
        process.start()
        process.join(60)
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


class TestSharedMetadata(TestCase):
    def test_configured_pool_lookup_and_resolution(self):
        _, metadata = _create_attached(self, self.id(), slots=2)
        registered, pool = _registered_and_pool(self, metadata)

        self.assertEqual(metadata.pool_count(), 1)
        self.assertEqual(registered.slot_count, 2)
        self.assertEqual(pool.slot_count(), 2)
        self.assertFalse(hasattr(metadata, "register_or_attach_pool"))
        self.assertFalse(hasattr(metadata, "retire_pool"))

    def test_compatibility_identity_and_foreign_pool_rejection(self):
        _, metadata = _create_attached(self, self.id(), slots=1)
        registered, _ = _registered_and_pool(self, metadata)
        _, other_metadata = _create_attached(self, f"{self.id()}-other", slots=1)
        other_registered, _ = _registered_and_pool(self, other_metadata)

        self.assertEqual(
            registered.compatibility.metadata_version,
            registered.pool_ref.metadata_version,
        )
        key = CompatibleBlockKey(registered.compatibility, 0x2201)
        self.assertIsInstance(
            metadata.claim(other_registered.pool_ref, key), Unavailable
        )

    def test_claim_copy_publish_lookup_pin_copy_and_evict(self):
        _, metadata = _create_attached(self, self.id(), slots=1)
        registered, pool = _registered_and_pool(self, metadata)
        key = CompatibleBlockKey(registered.compatibility, 0x1234)
        reservation = metadata.claim(registered.pool_ref, key)
        self.assertIsInstance(reservation, Reservation)
        self.assertIsNone(metadata.lookup(key))

        source = torch.randn(64, device="spyre", dtype=torch.float16)
        copy_tensor_raw(source, pool, reservation.slot.slot_id, to_device=False)
        metadata.publish(reservation, [ChunkDescriptorEntry(0, 128)])
        entry = metadata.lookup(key)
        self.assertIsNotNone(entry)

        pin = metadata.pin_read(entry)
        self.assertIsInstance(pin, SlotReadPin)
        destination = torch.zeros_like(source)
        copy_tensor_raw(destination, pool, entry.slot.slot_id, to_device=True)
        self.assertEqual(source.to("cpu"), destination.to("cpu"))
        del pin

        self.assertTrue(metadata.evict(entry))
        self.assertIsNone(metadata.lookup(key))

    def test_abort_returns_reserved_slot(self):
        _, metadata = _create_attached(self, self.id(), slots=1)
        registered, _ = _registered_and_pool(self, metadata)
        key_a = CompatibleBlockKey(registered.compatibility, 0x2001)
        key_b = CompatibleBlockKey(registered.compatibility, 0x2002)

        reservation_a = metadata.claim(registered.pool_ref, key_a)
        first_slot = reservation_a.slot.slot_id
        metadata.abort(reservation_a)
        reservation_b = metadata.claim(registered.pool_ref, key_b)
        self.assertEqual(reservation_b.slot.slot_id, first_slot)
        metadata.abort(reservation_b)

    def test_reserved_existing_claim_and_no_space_are_distinct(self):
        _, metadata = _create_attached(self, self.id(), slots=1)
        registered, _ = _registered_and_pool(self, metadata)
        key_a = CompatibleBlockKey(registered.compatibility, 0x2101)
        key_b = CompatibleBlockKey(registered.compatibility, 0x2102)

        reservation = metadata.claim(registered.pool_ref, key_a)
        existing = metadata.claim(registered.pool_ref, key_a)
        self.assertIsInstance(existing, ExistingClaim)
        self.assertFalse(existing.valid)
        self.assertIsInstance(metadata.claim(registered.pool_ref, key_b), NoSpace)
        metadata.abort(reservation)

    def test_failed_publish_remains_abortable(self):
        _, metadata = _create_attached(self, self.id(), slots=1)
        registered, _ = _registered_and_pool(self, metadata)
        key = CompatibleBlockKey(registered.compatibility, 0x2301)
        reservation = metadata.claim(registered.pool_ref, key)

        with self.assertRaises(ValueError):
            metadata.publish(reservation, [])
        metadata.abort(reservation)
        self.assertIsNone(metadata.lookup(key))

    def test_published_key_is_unique_across_compatible_configured_pools(self):
        compatibility = _compatibility()
        children = [
            SharedDataPoolConfig(
                "host-a", PoolBackend.HOST_DRAM, 1, 128, compatibility
            ),
            SharedDataPoolConfig(
                "host-b", PoolBackend.HOST_DRAM, 1, 128, compatibility
            ),
        ]
        _, metadata = _create_attached(self, self.id(), children=children)
        pool_a, _ = _registered_and_pool(self, metadata, "host-a")
        pool_b, _ = _registered_and_pool(self, metadata, "host-b")
        key = CompatibleBlockKey(pool_a.compatibility, 0x2401)

        reservation = metadata.claim(pool_a.pool_ref, key)
        metadata.publish(reservation, [ChunkDescriptorEntry(0, 128)])
        existing = metadata.claim(pool_b.pool_ref, key)

        self.assertIsInstance(existing, ExistingClaim)
        self.assertTrue(existing.valid)
        self.assertEqual(existing.slot.pool.pool_id, pool_a.pool_ref.pool_id)

    def test_stale_entry_cannot_pin_or_evict_reused_slot(self):
        _, metadata = _create_attached(self, self.id(), slots=1)
        registered, _ = _registered_and_pool(self, metadata)
        key = CompatibleBlockKey(registered.compatibility, 0x2501)
        old_reservation = metadata.claim(registered.pool_ref, key)
        metadata.publish(old_reservation, [ChunkDescriptorEntry(0, 128)])
        old_entry = metadata.lookup(key)
        self.assertTrue(metadata.evict(old_entry))

        new_reservation = metadata.claim(registered.pool_ref, key)
        self.assertEqual(new_reservation.slot.slot_id, old_entry.slot.slot_id)
        self.assertNotEqual(
            new_reservation.slot.slot_version, old_entry.slot.slot_version
        )
        metadata.publish(new_reservation, [ChunkDescriptorEntry(0, 128)])

        self.assertIsNone(metadata.pin_read(old_entry))
        self.assertFalse(metadata.evict(old_entry))

    def test_stale_metadata_references_reject_replacement_generation(self):
        name = _pool_name(self.id())
        config = _config(name, slots=1)
        created = SharedPool.create_pool(config)
        old_attached = SharedPool.attach_pool(created.descriptor)
        old_metadata = old_attached.metadata
        old_registered, old_pool = _registered_and_pool(self, old_metadata)
        old_key = CompatibleBlockKey(old_registered.compatibility, 0x2521)
        old_reservation = old_metadata.claim(old_registered.pool_ref, old_key)
        old_metadata.publish(old_reservation, [ChunkDescriptorEntry(0, 128)])
        old_entry = old_metadata.lookup(old_key)

        del old_pool
        SharedPool.detach_pool(old_metadata)
        self.assertIs(
            SharedPool.destroy_pool(created.descriptor).code,
            PoolResultCode.OK,
        )
        replacement = SharedPool.create_pool(config)
        replacement_attached = SharedPool.attach_pool(replacement.descriptor)
        metadata = replacement_attached.metadata
        self.addCleanup(_cleanup_generation, metadata, replacement.descriptor)
        registered, _ = _registered_and_pool(self, metadata)
        fresh_key = CompatibleBlockKey(registered.compatibility, 0x2521)

        self.assertIsNone(metadata.lookup(old_key))
        self.assertIsNone(metadata.resolve_pool(old_registered.pool_ref))
        self.assertIsNone(metadata.pin_read(old_entry))
        self.assertFalse(metadata.evict(old_entry))
        self.assertIsInstance(
            metadata.claim(old_registered.pool_ref, fresh_key), Unavailable
        )

    def test_protocol_objects_expose_no_raw_addresses(self):
        _, metadata = _create_attached(self, self.id(), slots=1)
        registered, pool = _registered_and_pool(self, metadata)
        key = CompatibleBlockKey(registered.compatibility, 0x2601)
        reservation = metadata.claim(registered.pool_ref, key)
        metadata.publish(reservation, [ChunkDescriptorEntry(0, 128)])
        entry = metadata.lookup(key)
        pin = metadata.pin_read(entry)

        for value in (metadata, pool, reservation, entry, pin):
            self.assertFalse(hasattr(value, "slot_ptr"))
            self.assertFalse(hasattr(value, "host_address"))
            self.assertFalse(hasattr(value, "device_address"))
        del pin

    def test_read_pin_retains_native_metadata_after_detach(self):
        created, metadata = _create_attached(self, self.id(), slots=1)
        registered, pool = _registered_and_pool(self, metadata)
        key = CompatibleBlockKey(registered.compatibility, 0x2701)
        reservation = metadata.claim(registered.pool_ref, key)
        metadata.publish(reservation, [ChunkDescriptorEntry(0, 128)])
        entry = metadata.lookup(key)
        pin = metadata.pin_read(entry)

        del pool
        SharedPool.detach_pool(metadata)
        self.assertIs(
            SharedPool.destroy_pool(created.descriptor).code,
            PoolResultCode.POOL_BUSY,
        )
        del pin
        self.assertIs(
            SharedPool.destroy_pool(created.descriptor).code,
            PoolResultCode.OK,
        )
