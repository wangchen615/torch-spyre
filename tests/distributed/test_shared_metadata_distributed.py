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

import datetime
import gc
import os
import threading
import time
import traceback

import pytest
import torch
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_utils import run_tests

from torch_spyre._C import ChunkDescriptorEntry
from torch_spyre._C import CompatibleBlockKey
from torch_spyre._C import ExistingClaim
from torch_spyre._C import MetadataConfig
from torch_spyre._C import PoolBackend
from torch_spyre._C import PoolCompatibility
from torch_spyre._C import PoolConfig
from torch_spyre._C import PoolResultCode
from torch_spyre._C import Reservation
from torch_spyre._C import SharedDataPoolConfig
from torch_spyre._C import SharedPool
from torch_spyre._C import copy_tensor_raw
from torch_spyre._C import get_composite_address


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


def destroy_pool_if_present(name):
    opened = SharedPool.open_pool(name)
    if opened.code is PoolResultCode.OK:
        result = SharedPool.destroy_pool(opened.descriptor)
        if result.code not in (PoolResultCode.OK, PoolResultCode.NOT_FOUND):
            raise RuntimeError(f"failed to clean {name}: {result.code}")


def make_config(name):
    compatibility = PoolCompatibility("distributed-test", "float16", 64, 2)
    child = SharedDataPoolConfig("host-0", PoolBackend.HOST_DRAM, 2, 128, compatibility)
    return PoolConfig(name, MetadataConfig(1), [child])


def destroy_generation_when_idle(descriptor, timeout_seconds=15):
    deadline = time.monotonic() + timeout_seconds
    while True:
        result = SharedPool.destroy_pool(descriptor)
        if result.code in (PoolResultCode.OK, PoolResultCode.NOT_FOUND):
            return
        if result.code is not PoolResultCode.POOL_BUSY:
            raise RuntimeError(
                f"failed to destroy {descriptor.name}: {result.code}: {result.detail}"
            )
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"timed out destroying busy generation {descriptor.name}"
            )
        time.sleep(0.05)


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
            dist.init_process_group(
                f"cpu:gloo,spyre:{C10D_BACKEND}",
                timeout=datetime.timedelta(seconds=120),
            )

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

    def _cleanup_test_generation(self, state):
        cleanup_errors = []

        state["protected_pin"] = None
        worker = state.get("worker")
        if worker is not None:
            worker.join(10)
            if worker.is_alive():
                cleanup_errors.append("eviction worker did not finish")
        state["worker"] = None

        state["dependents"].clear()
        state["pool"] = None
        attached = state.get("attached")
        if attached is not None and attached.metadata is not None:
            try:
                SharedPool.detach_pool(attached.metadata)
            except Exception as error:
                cleanup_errors.append(f"metadata detach failed: {error}")
        state["metadata"] = None
        state["attached"] = None
        gc.collect()

        descriptor = state.get("descriptor")
        if self.comm_rank == 0 and descriptor is not None:
            try:
                destroy_generation_when_idle(descriptor)
            except Exception as error:
                cleanup_errors.append(str(error))
        if cleanup_errors:
            raise RuntimeError("; ".join(cleanup_errors))

    def test_concurrent_cross_process_metadata_and_dma(self):
        metadata_name = os.getenv(
            "TORCH_SPYRE_SHARED_METADATA_TEST_NAME",
            f"torch-spyre-dist-metadata-{self.id()}",
        )
        state = {
            "descriptor": None,
            "attached": None,
            "metadata": None,
            "pool": None,
            "protected_pin": None,
            "worker": None,
            "dependents": [],
        }
        try:
            self._run_concurrent_cross_process_metadata_and_dma(
                metadata_name,
                state,
            )
        except BaseException as error:
            traceback.clear_frames(error.__traceback__)
            raise
        finally:
            self._cleanup_test_generation(state)

    def _run_concurrent_cross_process_metadata_and_dma(self, metadata_name, state):
        descriptor_box = [None]
        if self.comm_rank == 0:
            destroy_pool_if_present(metadata_name)
            created = SharedPool.create_pool(make_config(metadata_name))
            self.assertIs(created.code, PoolResultCode.OK)
            state["descriptor"] = created.descriptor
            opened = SharedPool.open_pool(metadata_name)
            self.assertIs(opened.code, PoolResultCode.OK)
            self.assertEqual(opened.descriptor, created.descriptor)
            descriptor_box[0] = opened.descriptor

        dist.broadcast_object_list(descriptor_box, src=0)
        descriptor = descriptor_box[0]
        state["descriptor"] = descriptor
        attached = SharedPool.attach_pool(descriptor)
        state["attached"] = attached
        self.assertIs(attached.code, PoolResultCode.OK)
        self.assertIsNotNone(attached.metadata)
        md = attached.metadata
        state["metadata"] = md
        registered = md.find_pool("host-0")
        self.assertIsNotNone(registered)
        pool = md.resolve_pool(registered.pool_ref)
        state["pool"] = pool
        self.assertIsNotNone(pool)
        if os.getenv("TORCH_SPYRE_SHARED_METADATA_FAIL_AFTER_RESOLVE") == "1":
            raise RuntimeError("injected failure after data-pool resolution")

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
            state["dependents"].append(own_entry)
            self.assertEqual(own_entry.slot.slot_id, own_slot.slot_id)
            self.assertTrue(md.evict(own_entry))
            dist.barrier()  # both slots are free for the next round

        same_key = CompatibleBlockKey(registered.compatibility, 0x7F00)
        dist.barrier()  # start both same-key claims together
        claim = md.claim(registered.pool_ref, same_key)
        state["dependents"].append(claim)
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
        state["dependents"].append(entry)
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
        state["dependents"].append(protected_entry)
        protected_pin = md.pin_read(protected_entry) if self.comm_rank == 1 else None
        state["protected_pin"] = protected_pin
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
            state["worker"] = worker
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
            state["protected_pin"] = None
            del protected_pin
        if self.comm_rank == 0:
            worker.join(5)
            self.assertFalse(worker.is_alive())
            self.assertEqual(evicted, [True])
            state["worker"] = None
            del worker
        dist.barrier()  # pin release allowed eviction to finish

        replacement = None
        source = None
        if self.comm_rank == 0:
            replacement = md.claim(registered.pool_ref, replacement_key)
            state["dependents"].append(replacement)
            self.assertIsInstance(replacement, Reservation)
            self.assertEqual(replacement.slot.slot_id, protected_slot.slot_id)
            self.assertNotEqual(
                replacement.slot.slot_version,
                protected_slot.slot_version,
            )
            source = expected_page(0, 34).to(DEVICE)
            state["dependents"].append(source)
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
        state["dependents"].clear()

        if self.comm_rank == 0:
            state["pool"] = None
            pool = None
            SharedPool.detach_pool(attached.metadata)
            state["metadata"] = None
            state["attached"] = None
        dist.barrier()  # rank 1 deliberately retains both handles

        if self.comm_rank == 0:
            busy = SharedPool.destroy_pool(descriptor)
            self.assertIs(busy.code, PoolResultCode.POOL_BUSY)
        dist.barrier()

        if self.comm_rank == 1:
            state["pool"] = None
            pool = None
            SharedPool.detach_pool(attached.metadata)
            state["metadata"] = None
            state["attached"] = None
        dist.barrier()

        if self.comm_rank == 0:
            destroyed = SharedPool.destroy_pool(descriptor)
            self.assertIs(destroyed.code, PoolResultCode.OK)
            self.assertIs(
                SharedPool.open_pool(metadata_name).code,
                PoolResultCode.NOT_FOUND,
            )
            state["descriptor"] = None
        dist.barrier()


if __name__ == "__main__":
    run_tests()
