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
        shared_pool = torch_spyre._C.SharedHostPool.create_or_attach("Testing", 5, 5)

        self.assertEqual(shared_pool.slot_count(), 5)
        self.assertGreaterEqual(shared_pool.slot_bytes(), 5)


if __name__ == "__main__":
    run_tests()
