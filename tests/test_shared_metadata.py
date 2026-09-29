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
    CompatibilityDescriptor,
    SharedDataPool,
    SharedDataPoolConfig,
    SharedMetadata,
    SharedMetadataCapacity,
    SharedMetadataConfig,
    SharedPoolKind,
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


if __name__ == "__main__":
    run_tests()
