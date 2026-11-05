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

"""Marvell lifecycle binding tests that need no Marvell card.

They cover the Marvell enum values, MarvellPlacement, the SharedDataPoolConfig
marvell field and the pickle versions: host-only state stays version 3, state
with a Marvell placement is version 4, and both decode.
"""

import os
import pickle

import pytest

from shared_pool_utils import make_pool_config
from shared_pool_utils import unique_pool_name
from torch_spyre._C import MarvellPlacement
from torch_spyre._C import MetadataConfig
from torch_spyre._C import PoolBackend
from torch_spyre._C import PoolCompatibility
from torch_spyre._C import PoolConfig
from torch_spyre._C import PoolResultCode
from torch_spyre._C import PoolState
from torch_spyre._C import SharedDataPartitionConfig
from torch_spyre._C import SharedDataPoolConfig
from torch_spyre._C import SharedPool
from torch_spyre._C import SharedPoolKind
from torch_spyre._C import _decode_pool_config_state
from torch_spyre._C import _decode_pool_descriptor_state

BDF = "0000:ed:00.0"
PAGE = os.sysconf("SC_PAGE_SIZE")
COMPATIBILITY = PoolCompatibility("test-model", "float16", 16, 1)


def marvell_arena(name="marvell-r0", bar_offset=None):
    """One rank-0 Marvell arena with one partition."""
    return SharedDataPoolConfig(
        name,
        PoolBackend.MARVELL,
        [SharedDataPartitionConfig(f"{name}-c0", 0, 0, 2, PAGE, None)],
        COMPATIBILITY,
        marvell=MarvellPlacement(BDF, bar_offset),
    )


def host_arena(name="host-r0"):
    return SharedDataPoolConfig(
        name,
        PoolBackend.HOST_DRAM,
        [SharedDataPartitionConfig(f"{name}-c0", 0, 0, 2, PAGE, None)],
        COMPATIBILITY,
    )


def effective_child(name, backend, placement):
    """Version-4 effective child state: one canonical partition at offset 0."""
    return (
        name,
        backend.value,
        2,
        PAGE,
        "test-model",
        "float16",
        16,
        1,
        "",
        [(f"{name}-c0", 0, 0, 2, PAGE, 0)],
        placement,
    )


def descriptor_state(version, children):
    return (
        version,
        "/marvell",
        b"generation-00001",
        PoolState.READY.value,
        1,
        children,
    )


def test_marvell_enum_values():
    assert PoolBackend.MARVELL.value == 2  # flex PoolBackend::Marvell.
    assert SharedPoolKind.MARVELL.value == 1  # flex SharedPoolKind::MARVELL.


def test_placement_fields_and_equality():
    placement = MarvellPlacement(BDF, 4 * PAGE)
    assert placement.pci_bdf == BDF
    assert placement.bar_offset == 4 * PAGE
    assert MarvellPlacement(BDF).bar_offset is None  # flex places the window.
    assert placement == MarvellPlacement(BDF, 4 * PAGE)
    assert placement != MarvellPlacement(BDF)


def test_config_marvell_field_on_every_constructor():
    flat = SharedDataPoolConfig(
        "marvell-r0",
        PoolBackend.MARVELL,
        2,
        PAGE,
        COMPATIBILITY,
        marvell=MarvellPlacement(BDF),
    )
    both = SharedDataPoolConfig(
        "marvell-r0",
        PoolBackend.MARVELL,
        0,
        0,
        [SharedDataPartitionConfig("marvell-r0-c0", 0, 0, 2, PAGE, None)],
        COMPATIBILITY,
        marvell=MarvellPlacement(BDF),
    )
    assert flat.marvell == MarvellPlacement(BDF)
    assert both.marvell == MarvellPlacement(BDF)
    assert marvell_arena().marvell.pci_bdf == BDF
    assert host_arena().marvell is None  # host arenas carry no placement.


def test_host_only_config_still_pickles_as_version_3():
    config = make_pool_config("/host-only")
    state = config.__getstate__()
    assert state[0] == 3  # unchanged host-only format.
    assert len(state[3][0]) == 10  # no placement item.
    assert pickle.loads(pickle.dumps(config)) == config


def test_marvell_config_pickles_as_version_4():
    arenas = [host_arena(), marvell_arena(bar_offset=PAGE)]
    config = PoolConfig("/mixed", MetadataConfig(1), arenas)
    state = config.__getstate__()
    assert state[0] == 4  # a placement is present.
    assert state[3][0][10] is None  # host arena.
    assert state[3][1][10] == (BDF, PAGE)  # Marvell arena.
    assert pickle.loads(pickle.dumps(config)) == config


def test_unplaced_marvell_config_round_trips():
    config = PoolConfig("/unplaced", MetadataConfig(1), [marvell_arena()])
    restored = pickle.loads(pickle.dumps(config))
    assert restored == config
    assert restored.data_pools[0].marvell.bar_offset is None  # stays unplaced.


def test_version_4_descriptor_decodes_placement():
    descriptor = _decode_pool_descriptor_state(
        descriptor_state(
            4,
            [
                effective_child("host-r0", PoolBackend.HOST_DRAM, None),
                effective_child("marvell-r0", PoolBackend.MARVELL, (BDF, 8 * PAGE)),
            ],
        )
    )
    host, marvell = descriptor.effective_config.data_pools
    assert host.backend is PoolBackend.HOST_DRAM and host.marvell is None
    assert marvell.backend is PoolBackend.MARVELL
    assert marvell.marvell == MarvellPlacement(BDF, 8 * PAGE)
    restored = pickle.loads(pickle.dumps(descriptor))
    assert restored == descriptor  # re-encodes as version 4.


@pytest.mark.parametrize(
    "child, match",
    [
        (effective_child("marvell-r0", PoolBackend.MARVELL, None), "placement"),
        (effective_child("host-r0", PoolBackend.HOST_DRAM, (BDF, 0)), "placement"),
        (effective_child("marvell-r0", PoolBackend.MARVELL, (BDF,)), "placement"),
        (effective_child("marvell-r0", PoolBackend.MARVELL, [BDF, 0]), "placement"),
    ],
)
def test_version_4_rejects_inconsistent_placement(child, match):
    with pytest.raises(ValueError, match=match):
        _decode_pool_descriptor_state(descriptor_state(4, [child]))


def test_version_4_requires_the_placement_item():
    child = effective_child("host-r0", PoolBackend.HOST_DRAM, None)[:10]
    with pytest.raises(ValueError, match="must contain 11 fields"):
        _decode_pool_descriptor_state(descriptor_state(4, [child]))


def test_version_3_rejects_a_marvell_backend():
    child = effective_child("marvell-r0", PoolBackend.MARVELL, None)[:10]
    with pytest.raises(ValueError, match="placement"):
        _decode_pool_descriptor_state(descriptor_state(3, [child]))


def test_config_state_rejects_unknown_version():
    config = PoolConfig("/unplaced", MetadataConfig(1), [marvell_arena()])
    state = list(config.__getstate__())
    state[0] = 5
    with pytest.raises(ValueError, match="version"):
        _decode_pool_config_state(tuple(state))


def test_create_rejects_marvell_backend_without_placement():
    name = unique_pool_name("marvell-no-placement")
    arena = SharedDataPoolConfig(
        "marvell-r0",
        PoolBackend.MARVELL,
        [SharedDataPartitionConfig("marvell-r0-c0", 0, 0, 2, PAGE, None)],
        COMPATIBILITY,
    )
    config = PoolConfig(name, MetadataConfig(1), [arena])
    with pytest.raises(ValueError, match="Marvell placement is required"):
        SharedPool.create_pool(config)  # rejected before any card access.
    assert SharedPool.open_pool(name).code is PoolResultCode.NOT_FOUND  # unpublished.
