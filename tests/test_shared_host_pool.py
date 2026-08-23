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


from torch.testing._internal.common_utils import run_tests, TestCase

import torch_spyre


class TestSharedHostPool(TestCase):
    """
    Tests for the SharedHostPool functionality in the torch_spyre module.
    """

    def test_create_or_attach(self):
        # Create a shared pool
        shared_pool = torch_spyre._C.SharedHostPool.create_or_attach("Testing", 5, 5)

        # Check if slot count is as expected
        self.assertEqual(shared_pool.slot_count(), 5)

        # Check if greater than or equal because the actual slot bytes may be
        # larger due to alignment of size/stride of the pool
        self.assertGreaterEqual(shared_pool.slot_bytes(), 5)

    def test_attach_existing_pool(self):
        # Create a shared pool
        torch_spyre._C.SharedHostPool.create_or_attach("Testing", 5, 5)

        # Attach to the existing shared pool
        shared_pool_compare = torch_spyre._C.SharedHostPool.create_or_attach(
            "Testing", 10, 10
        )

        self.assertEqual(shared_pool_compare.slot_count(), 5)
        self.assertGreaterEqual(shared_pool_compare.slot_bytes(), 5)

    def no_host_pointer(self):
        # Create a shared pool
        shared_pool = torch_spyre._C.SharedHostPool.create_or_attach("Testing", 5, 5)

        # Check if host pointer is None
        self.assertIsNone(shared_pool.host_pointer())

    def test_pool_real_model(self):
        # KV geometry from ibm-ai-platform/micro-g3.3-8b-instruct-1b (HF config)
        NUM_LAYERS, KV_HEADS, HEAD_DIM, BLOCK_SIZE = 4, 8, 128, 128

        # Multiply by 2 for key and value, and by 2 for float16
        slot_bytes = NUM_LAYERS * KV_HEADS * HEAD_DIM * 2 * 2 * BLOCK_SIZE

        # Choosing common prompt length of 8192 for testing
        slot_count = 8192 / BLOCK_SIZE

        torch_spyre._C.SharedHostPool.create_or_attach(
            "Testing", int(slot_count), int(slot_bytes)
        )


if __name__ == "__main__":
    run_tests()
