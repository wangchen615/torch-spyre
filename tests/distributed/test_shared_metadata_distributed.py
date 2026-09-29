# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import threading

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase, run_tests

from torch_spyre._C import (  # type: ignore[attr-defined]
    ChunkDescriptorEntry,
    CompatibilityDescriptor,
    CompatibleBlockKey,
    ExistingClaim,
    Reservation,
    SharedDataPoolConfig,
    SharedMetadata,
    SharedMetadataConfig,
    SharedPoolKind,
    copy_tensor_raw,
    get_composite_address,
)


if "RANK" not in os.environ:
    pytest.skip(
        "RANK environment variable not defined, skipping distributed tests",
        allow_module_level=True,
    )

if "WORLD_SIZE" not in os.environ:
    pytest.skip(
        "WORLD_SIZE environment variable not defined, skipping distributed tests",
        allow_module_level=True,
    )

try:
    world_size = int(os.environ["WORLD_SIZE"])
except ValueError:
    pytest.skip(
        "WORLD_SIZE environment variable is not a valid integer",
        allow_module_level=True,
    )

if world_size != 2:
    pytest.skip(
        f"WORLD_SIZE is {world_size}, need exactly 2 for this test",
        allow_module_level=True,
    )


DEVICE = torch.device(f"spyre:{os.environ['RANK']}")
C10D_BACKEND = "spyreccl"
PAGE_ELEMENTS = 64


def expected_page(owner: int, operation: int) -> torch.Tensor:
    offset = owner * 512 + operation * PAGE_ELEMENTS
    return torch.arange(
        offset,
        offset + PAGE_ELEMENTS,
        dtype=torch.float16,
    )


def block_key(compatibility, owner: int, operation: int):
    return CompatibleBlockKey(
        compatibility,
        0x7000 + owner * 0x100 + operation,
    )


class TestSharedMetadataCrossProcess(TestCase):
    @classmethod
    def setUpClass(cls):
        if not dist.distributed_c10d.is_backend_available(C10D_BACKEND):
            raise RuntimeError(f"Error: Missing the C10 Backend {C10D_BACKEND}")
        if C10D_BACKEND != dist.get_default_backend_for_device("spyre"):
            raise RuntimeError(
                f"Error: Missing a C10 Backend for 'spyre'! Expected {C10D_BACKEND}"
            )

        if not dist.is_initialized():
            dist.init_process_group(f"cpu:gloo,spyre:{C10D_BACKEND}")

        cls.comm_size = dist.get_world_size()
        cls.comm_rank = dist.get_rank()
        if cls.comm_size != 2:
            raise RuntimeError(
                f"SharedMetadata test requires exactly 2 ranks, got {cls.comm_size}"
            )
        if torch.spyre.device_count() < 2:
            raise RuntimeError("SharedMetadata test requires two Spyre cards")

    @classmethod
    def tearDownClass(cls):
        if dist.is_initialized():
            dist.destroy_process_group()

    def publish_reserved(self, md, pool, reservation, owner, operation):
        source = expected_page(owner, operation).to(DEVICE)
        address = get_composite_address(source)
        chunks = [
            ChunkDescriptorEntry(chunk.domain_id, chunk.size)
            for chunk in address.chunks()
        ]
        copy_tensor_raw(source, pool, reservation.slot.slot_id, to_device=False)
        md.publish(reservation, chunks)

    def offload(self, md, pool, pool_ref, compatibility, owner, operation):
        key = block_key(compatibility, owner, operation)
        reservation = md.claim(pool_ref, key)
        self.assertIsInstance(reservation, Reservation)
        self.publish_reserved(md, pool, reservation, owner, operation)
        return key, reservation.slot

    def reload(self, md, pool, key, owner, operation):
        entry = md.lookup(key)
        self.assertIsNotNone(entry)
        pin = md.pin_read(entry)
        self.assertIsNotNone(pin)
        destination = torch.empty(PAGE_ELEMENTS, device=DEVICE, dtype=torch.float16)
        copy_tensor_raw(destination, pool, entry.slot.slot_id, to_device=True)
        del pin
        self.assertEqual(expected_page(owner, operation), destination.cpu())
        return entry

    def test_concurrent_cross_process_metadata_and_dma(self):
        metadata_name = self.id()
        pool_name = f"{metadata_name}.pool"
        if self.comm_rank == 0:
            SharedMetadata.unlink_by_name(metadata_name)
        dist.barrier()  # rank 0 removed any stale directory

        compatibility = CompatibilityDescriptor(1, [0x53, 0x50, 0x59, 0x52, 0x45])
        pool_config = SharedDataPoolConfig(
            pool_name,
            SharedPoolKind.HOST,
            2,
            128,
            compatibility,
        )
        md = SharedMetadata.create_or_attach(
            metadata_name,
            SharedMetadataConfig(1, [pool_config], None),
        )
        registered = md.find_pool(pool_name)
        self.assertIsNotNone(registered)
        pool = md.resolve_pool(registered.pool_ref)
        self.assertIsNotNone(pool)

        for operation in range(4):
            dist.barrier()  # start both distinct-key offloads together
            own_key, own_slot = self.offload(
                md,
                pool,
                registered.pool_ref,
                registered.compatibility,
                self.comm_rank,
                operation,
            )
            dist.barrier()  # both pages are published

            peer = 1 - self.comm_rank
            peer_key = block_key(registered.compatibility, peer, operation)
            self.reload(md, pool, peer_key, peer, operation)
            dist.barrier()  # both peer reload pins are released

            own_entry = md.lookup(own_key)
            self.assertIsNotNone(own_entry)
            self.assertEqual(own_entry.slot.slot_id, own_slot.slot_id)
            self.assertTrue(md.evict(own_entry))
            dist.barrier()  # both slots are free for the next round

        same_key = CompatibleBlockKey(registered.compatibility, 0x7F00)
        dist.barrier()  # start both same-key claims together
        claim = md.claim(registered.pool_ref, same_key)
        is_winner = isinstance(claim, Reservation)
        if not is_winner:
            self.assertIsInstance(claim, ExistingClaim)
            self.assertFalse(claim.valid)

        winner_flags = [torch.zeros(1, dtype=torch.int64) for _ in range(2)]
        dist.all_gather(
            winner_flags,
            torch.tensor([is_winner], dtype=torch.int64),
        )
        self.assertEqual(sum(flag.item() for flag in winner_flags), 1)
        winner = next(i for i, flag in enumerate(winner_flags) if flag.item())
        observed_slot = claim.slot
        dist.barrier()  # both ranks observed the reserved entry

        if is_winner:
            self.publish_reserved(md, pool, claim, winner, operation=16)
        dist.barrier()  # the winning page is published

        entry = self.reload(md, pool, same_key, winner, operation=16)
        self.assertEqual(entry.slot.pool.pool_id, observed_slot.pool.pool_id)
        self.assertEqual(entry.slot.slot_id, observed_slot.slot_id)
        self.assertEqual(entry.slot.slot_version, observed_slot.slot_version)
        dist.barrier()  # both same-key reload pins are released
        if self.comm_rank == 0:
            self.assertTrue(md.evict(entry))
        dist.barrier()  # the same-key slot is free

        protected_key = block_key(registered.compatibility, 0, 32)
        unrelated_key = block_key(registered.compatibility, 0, 33)
        replacement_key = block_key(registered.compatibility, 0, 34)

        protected_slot = None
        if self.comm_rank == 0:
            actual_key, protected_slot = self.offload(
                md,
                pool,
                registered.pool_ref,
                registered.compatibility,
                0,
                32,
            )
            self.assertEqual(actual_key.block_hash, protected_key.block_hash)
        dist.barrier()  # the protected page is published

        protected_entry = md.lookup(protected_key)
        self.assertIsNotNone(protected_entry)
        protected_pin = md.pin_read(protected_entry) if self.comm_rank == 1 else None
        if self.comm_rank == 1:
            self.assertIsNotNone(protected_pin)
        dist.barrier()  # rank 1 holds the protected slot pin

        if self.comm_rank == 0:
            started = threading.Event()
            finished = threading.Event()
            evicted = []

            def evict_protected(metadata, entry):
                started.set()
                evicted.append(metadata.evict(entry))
                finished.set()

            worker = threading.Thread(
                target=evict_protected,
                args=(md, protected_entry),
            )
            worker.start()
            self.assertTrue(started.wait(5))
            self.assertFalse(finished.wait(0.1))
            unrelated_key, _ = self.offload(
                md,
                pool,
                registered.pool_ref,
                registered.compatibility,
                0,
                33,
            )
            self.assertFalse(finished.is_set())
        else:
            destination = torch.empty(
                PAGE_ELEMENTS,
                device=DEVICE,
                dtype=torch.float16,
            )
            copy_tensor_raw(
                destination,
                pool,
                protected_entry.slot.slot_id,
                to_device=True,
            )
            self.assertEqual(expected_page(0, 32), destination.cpu())

        dist.barrier()  # protected reload and unrelated publish completed
        if self.comm_rank == 0:
            self.assertFalse(finished.is_set())
        else:
            self.reload(md, pool, unrelated_key, owner=0, operation=33)
        dist.barrier()  # rank 1 still holds the protected pin

        if self.comm_rank == 1:
            del protected_pin
        if self.comm_rank == 0:
            worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(evicted, [True])
            del worker
        dist.barrier()  # pin release allowed eviction to finish

        replacement = None
        source = None
        if self.comm_rank == 0:
            replacement = md.claim(registered.pool_ref, replacement_key)
            self.assertIsInstance(replacement, Reservation)
            self.assertEqual(replacement.slot.slot_id, protected_slot.slot_id)
            self.assertNotEqual(
                replacement.slot.slot_version,
                protected_slot.slot_version,
            )
            source = expected_page(0, 34).to(DEVICE)
            copy_tensor_raw(
                source,
                pool,
                replacement.slot.slot_id,
                to_device=False,
            )
        dist.barrier()  # replacement bytes exist but remain unpublished

        if self.comm_rank == 1:
            self.assertIsNone(md.lookup(replacement_key))
            self.assertIsNone(md.pin_read(protected_entry))
        dist.barrier()  # rank 1 observed both required misses

        if self.comm_rank == 0:
            address = get_composite_address(source)
            md.publish(
                replacement,
                [
                    ChunkDescriptorEntry(chunk.domain_id, chunk.size)
                    for chunk in address.chunks()
                ],
            )
        dist.barrier()  # replacement is visible

        if self.comm_rank == 1:
            self.reload(md, pool, replacement_key, owner=0, operation=34)
        dist.barrier()  # all DMA and pins are complete

        del own_entry
        del protected_entry
        del entry
        del claim
        del replacement
        del source
        del pool
        del md
        dist.barrier()  # both ranks dropped shared-memory handles
        if self.comm_rank == 0:
            SharedMetadata.unlink_by_name(metadata_name)
        dist.barrier()  # rank 0 completed directory cleanup


if __name__ == "__main__":
    run_tests()
