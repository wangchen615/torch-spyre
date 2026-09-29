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
    SharedMetadata,
    SharedMetadataCapacity,
    SharedMetadataConfig,
    SharedPoolKind,
    Unavailable,
)


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


if __name__ == "__main__":
    run_tests()
