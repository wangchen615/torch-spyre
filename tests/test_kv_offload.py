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


import torch

from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
)

# TEMPORARY BLOCK FOR RUFF:
from torch_spyre._C import (  # type: ignore[attr-defined]
    SharedHostPool,
    copy_tensor_raw,
    get_composite_address_handle,
)


class TestSpyre(TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    def test_kv_offload_reload_zeroed(self):
        tensor_on_spyre = torch.randn(10, device="spyre", dtype=torch.float16)

        # Composite address handle for the tensor to determine the size of the slot needed in the shared host pool
        slot_bytes = get_composite_address_handle(tensor_on_spyre).total_size()

        # Create the shared pool with a single slot of the required size
        pool = SharedHostPool.create_or_attach(
            "kv_offload_pool", num_slots=1, slot_bytes=slot_bytes
        )

        # Use the first slot in the pool
        slot_id = 0

        # D2H: Move tensor from spyre to host memory pool
        copy_tensor_raw(tensor_on_spyre, pool, slot_id, to_device=False)

        # H2D: Move tensor back from host memory pool to spyre
        tensor_reloaded = tensor_on_spyre.zero_()
        copy_tensor_raw(tensor_reloaded, pool, slot_id, to_device=True)

        self.assertEqual(tensor_on_spyre, tensor_reloaded)

    def test_kv_offload_reload_diff_tensor(self):
        tensor_on_spyre = torch.randn(10, device="spyre", dtype=torch.float16)

        # Composite address handle for the tensor to determine the size of the slot needed in the shared host pool
        slot_bytes = get_composite_address_handle(tensor_on_spyre).total_size()

        # Create the shared pool with a single slot of the required size
        pool = SharedHostPool.create_or_attach(
            "kv_offload_pool", num_slots=1, slot_bytes=slot_bytes
        )

        # Use the first slot in the pool
        slot_id = 0

        # D2H: Move tensor from spyre to host memory pool
        copy_tensor_raw(tensor_on_spyre, pool, slot_id, to_device=False)

        # H2D: Move tensor back from host memory pool to spyre
        tensor_reloaded = tensor_on_spyre.empty_like(tensor_on_spyre)
        copy_tensor_raw(tensor_reloaded, pool, slot_id, to_device=True)

        self.assertEqual(tensor_on_spyre, tensor_reloaded)

    def test_normal_copy_tensor_unaffected(self):
        """
        Ensure the normal copy_tensor (.to()) function is unaffected.
        """
        tensor = torch.randn(10, dtype=torch.float16)
        tensor_on_spyre = tensor.to("spyre")
        tensor_back = tensor_on_spyre.to("cpu")
        self.assertEqual(tensor, tensor_back)


if __name__ == "__main__":
    run_tests()
