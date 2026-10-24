"""
Spyre C++ bindings
"""

from __future__ import annotations
import collections.abc
import torch
import typing

__all__: list[str] = [
    "AIUPTI_ACTIVITY_NAME_MAX_BYTES",
    "AttachResult",
    "ChunkDescriptorEntry",
    "CompatibilityRef",
    "CompositeAddressHandle",
    "CompositeChunkInfo",
    "CompatibleBlockKey",
    "DataFormats",
    "DestroyResult",
    "JobPlan",
    "ElementArrangement",
    "ExistingClaim",
    "MetadataConfig",
    "LookupEntry",
    "NoSpace",
    "PoolBackend",
    "PoolCompatibility",
    "PoolConfig",
    "PoolDescriptor",
    "PoolDescriptorResult",
    "PoolGeneration",
    "PoolRef",
    "PoolResultCode",
    "PoolState",
    "RegisteredPool",
    "Reservation",
    "SharedDataPool",
    "SharedDataPoolConfig",
    "SharedHostPool",
    "SharedMetadata",
    "SharedPool",
    "SharedPoolKind",
    "SlotReadPin",
    "SlotRef",
    "SpyreStreamError",
    "SpyreDeviceState",
    "SpyreTensorLayout",
    "SymbolicArg",
    "SymbolicArgKind",
    "Unavailable",
    "_SpyreStreamBase",
    "current_stream",
    "default_stream",
    "get_composite_address",
    "get_device_state",
    "get_stream_from_pool",
    "set_current_stream",
    "stream_get_error",
    "stream_get_error_string",
    "synchronize",
    "as_strided_with_layout",
    "empty_with_layout",
    "copy_tensor",
    "copy_tensor_raw",
    "copy_kv_page_raw",
    "fill_tensor",
    "encode_constant",
    "extract_kernel_provenance_key",
    "free_runtime",
    "get_device_dtype",
    "get_downcast_warning",
    "get_elem_in_stick",
    "get_spyre_tensor_layout",
    "get_device_size_in_bytes",
    "kernel_provenance_registry_stats",
    "launch_jobplan",
    "lookup_kernel_provenance",
    "prepare_kernel",
    "register_kernel_provenance",
    "set_downcast_warning",
    "set_spyre_tensor_layout",
    "spyre_empty_with_layout",
    "start_runtime",
    "to_with_layout",
]

AIUPTI_ACTIVITY_NAME_MAX_BYTES: int

class PoolBackend:
    HOST_DRAM: typing.ClassVar[PoolBackend]
    name: str
    value: int

class PoolState:
    INITIALIZING: typing.ClassVar[PoolState]
    READY: typing.ClassVar[PoolState]
    DESTROYING: typing.ClassVar[PoolState]
    name: str
    value: int

class PoolResultCode:
    OK: typing.ClassVar[PoolResultCode]
    NOT_FOUND: typing.ClassVar[PoolResultCode]
    ALREADY_EXISTS: typing.ClassVar[PoolResultCode]
    INITIALIZING: typing.ClassVar[PoolResultCode]
    DESTROYING: typing.ClassVar[PoolResultCode]
    POOL_BUSY: typing.ClassVar[PoolResultCode]
    STALE_GENERATION: typing.ClassVar[PoolResultCode]
    CONFIGURATION_MISMATCH: typing.ClassVar[PoolResultCode]
    RESOURCE_ERROR: typing.ClassVar[PoolResultCode]
    name: str
    value: int

class SharedPoolKind:
    HOST: typing.ClassVar[SharedPoolKind]
    name: str
    value: int

class PoolCompatibility:
    def __init__(
        self,
        model: str,
        kv_dtype: str,
        block_size: int,
        tensor_parallel_size: int,
    ) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    @property
    def model(self) -> str: ...
    @property
    def kv_dtype(self) -> str: ...
    @property
    def block_size(self) -> int: ...
    @property
    def tensor_parallel_size(self) -> int: ...

class MetadataConfig:
    def __init__(self, max_chunks: int) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    @property
    def max_chunks(self) -> int: ...

class SharedDataPoolConfig:
    def __init__(
        self,
        name: str,
        backend: PoolBackend,
        num_slots: int,
        slot_bytes: int,
        compatibility: PoolCompatibility,
    ) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    @property
    def name(self) -> str: ...
    @property
    def backend(self) -> PoolBackend: ...
    @property
    def num_slots(self) -> int: ...
    @property
    def slot_bytes(self) -> int: ...
    @property
    def compatibility(self) -> PoolCompatibility: ...

class PoolConfig:
    def __init__(
        self,
        name: str,
        metadata: MetadataConfig,
        data_pools: collections.abc.Sequence[SharedDataPoolConfig],
    ) -> None: ...
    def __eq__(self, other: object) -> bool: ...
    @property
    def name(self) -> str: ...
    @property
    def metadata(self) -> MetadataConfig: ...
    @property
    def data_pools(self) -> list[SharedDataPoolConfig]: ...

class PoolGeneration:
    def __eq__(self, other: object) -> bool: ...
    @property
    def bytes(self) -> bytes: ...

class PoolDescriptor:
    def __eq__(self, other: object) -> bool: ...
    @property
    def name(self) -> str: ...
    @property
    def generation(self) -> PoolGeneration: ...
    @property
    def state(self) -> PoolState: ...
    @property
    def effective_config(self) -> PoolConfig: ...

class PoolDescriptorResult:
    @property
    def code(self) -> PoolResultCode: ...
    @property
    def descriptor(self) -> PoolDescriptor: ...
    @property
    def detail(self) -> str: ...

class DestroyResult:
    @property
    def code(self) -> PoolResultCode: ...
    @property
    def detail(self) -> str: ...

class CompatibilityRef:
    def __eq__(self, other: object) -> bool: ...
    @property
    def metadata_version(self) -> int: ...
    @property
    def compatibility_id(self) -> int: ...

class PoolRef:
    def __eq__(self, other: object) -> bool: ...
    @property
    def metadata_version(self) -> int: ...
    @property
    def pool_id(self) -> int: ...
    @property
    def pool_version(self) -> int: ...

class RegisteredPool:
    @property
    def pool_ref(self) -> PoolRef: ...
    @property
    def compatibility(self) -> CompatibilityRef: ...
    @property
    def name(self) -> str: ...
    @property
    def kind(self) -> SharedPoolKind: ...
    @property
    def slot_count(self) -> int: ...
    @property
    def slot_bytes(self) -> int: ...

class CompatibleBlockKey:
    def __init__(self, compatibility: CompatibilityRef, block_hash: int) -> None: ...
    @property
    def compatibility(self) -> CompatibilityRef: ...
    @property
    def block_hash(self) -> int: ...

class SlotRef:
    @property
    def pool(self) -> PoolRef: ...
    @property
    def slot_id(self) -> int: ...
    @property
    def slot_version(self) -> int: ...

class ChunkDescriptorEntry:
    def __init__(self, domain_id: int, size: int) -> None: ...
    @property
    def domain_id(self) -> int: ...
    @property
    def size(self) -> int: ...

class LookupEntry:
    @property
    def key(self) -> CompatibleBlockKey: ...
    @property
    def slot(self) -> SlotRef: ...
    @property
    def chunks(self) -> list[ChunkDescriptorEntry]: ...

class Reservation:
    @property
    def key(self) -> CompatibleBlockKey: ...
    @property
    def slot(self) -> SlotRef: ...

class ExistingClaim:
    @property
    def slot(self) -> SlotRef: ...
    @property
    def valid(self) -> bool: ...

class NoSpace: ...
class Unavailable: ...
class SlotReadPin: ...

class SharedDataPool:
    def name(self) -> str: ...
    def slot_count(self) -> int: ...
    def slot_bytes(self) -> int: ...
    def total_bytes(self) -> int: ...

class SharedHostPool(SharedDataPool): ...

class CompositeChunkInfo:
    @property
    def region_id(self) -> int: ...
    @property
    def offset(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def domain_id(self) -> int: ...

class CompositeAddressHandle:
    @property
    def total_size(self) -> int: ...
    @property
    def num_chunks(self) -> int: ...
    def chunks(self) -> list[CompositeChunkInfo]: ...

class SharedMetadata:
    def pool_count(self) -> int: ...
    def version(self) -> int: ...
    def find_pool(self, name: str) -> RegisteredPool | None: ...
    def resolve_pool(self, pool_ref: PoolRef) -> SharedDataPool: ...
    def lookup(self, key: CompatibleBlockKey) -> LookupEntry | None: ...
    def pin_read(self, entry: LookupEntry) -> SlotReadPin | None: ...
    def claim(
        self, target_pool: PoolRef, key: CompatibleBlockKey
    ) -> Reservation | ExistingClaim | NoSpace | Unavailable: ...
    def publish(
        self, reservation: Reservation, chunks: list[ChunkDescriptorEntry]
    ) -> None: ...
    def abort(self, reservation: Reservation) -> None: ...
    def evict(self, entry: LookupEntry) -> bool: ...

class AttachResult:
    @property
    def code(self) -> PoolResultCode: ...
    @property
    def metadata(self) -> SharedMetadata | None: ...
    @property
    def detail(self) -> str: ...

class SharedPool:
    @staticmethod
    def create_pool(config: PoolConfig) -> PoolDescriptorResult: ...
    @staticmethod
    def open_pool(name: str) -> PoolDescriptorResult: ...
    @staticmethod
    def destroy_pool(descriptor: PoolDescriptor) -> DestroyResult: ...
    @staticmethod
    def attach_pool(descriptor: PoolDescriptor) -> AttachResult: ...
    @staticmethod
    def detach_pool(metadata: SharedMetadata | None) -> None: ...

def _decode_pool_descriptor_state(state: tuple[typing.Any, ...]) -> PoolDescriptor: ...
def extract_kernel_provenance_key(event_name: str) -> str | None: ...
def kernel_provenance_registry_stats() -> dict[str, int]: ...
def lookup_kernel_provenance(key: str) -> list[str] | None: ...
def register_kernel_provenance(
    event_base_name: str, debug_handle_ids: collections.abc.Sequence[str]
) -> bool: ...

class DataFormats:
    """
    Members:

      SEN169_FP16

      IEEE_FP32

      INVALID

      SEN143_FP8

      SEN152_FP8

      SEN153_FP9

      SENINT2

      SENINT4

      SENINT8

      SENINT16

      SENINT24

      IEEE_INT64

      IEEE_INT32

      SENUINT32

      SENUINT2

      IEEE_FP16

      BOOL

      BFLOAT16

      SEN18F_FP24
    """

    BFLOAT16: typing.ClassVar[DataFormats]  # value = <DataFormats.BFLOAT16: 17>
    BOOL: typing.ClassVar[DataFormats]  # value = <DataFormats.BOOL: 16>
    IEEE_FP16: typing.ClassVar[DataFormats]  # value = <DataFormats.IEEE_FP16: 15>
    IEEE_FP32: typing.ClassVar[DataFormats]  # value = <DataFormats.IEEE_FP32: 1>
    IEEE_INT32: typing.ClassVar[DataFormats]  # value = <DataFormats.IEEE_INT32: 12>
    IEEE_INT64: typing.ClassVar[DataFormats]  # value = <DataFormats.IEEE_INT64: 11>
    INVALID: typing.ClassVar[DataFormats]  # value = <DataFormats.INVALID: 2>
    SEN143_FP8: typing.ClassVar[DataFormats]  # value = <DataFormats.SEN143_FP8: 3>
    SEN152_FP8: typing.ClassVar[DataFormats]  # value = <DataFormats.SEN152_FP8: 4>
    SEN153_FP9: typing.ClassVar[DataFormats]  # value = <DataFormats.SEN153_FP9: 5>
    SEN169_FP16: typing.ClassVar[DataFormats]  # value = <DataFormats.SEN169_FP16: 0>
    SEN18F_FP24: typing.ClassVar[DataFormats]  # value = <DataFormats.SEN18F_FP24: 18>
    SENINT16: typing.ClassVar[DataFormats]  # value = <DataFormats.SENINT16: 9>
    SENINT2: typing.ClassVar[DataFormats]  # value = <DataFormats.SENINT2: 6>
    SENINT24: typing.ClassVar[DataFormats]  # value = <DataFormats.SENINT24: 10>
    SENINT4: typing.ClassVar[DataFormats]  # value = <DataFormats.SENINT4: 7>
    SENINT8: typing.ClassVar[DataFormats]  # value = <DataFormats.SENINT8: 8>
    SENUINT2: typing.ClassVar[DataFormats]  # value = <DataFormats.SENUINT2: 14>
    SENUINT32: typing.ClassVar[DataFormats]  # value = <DataFormats.SENUINT32: 13>
    __members__: typing.ClassVar[
        dict[str, DataFormats]
    ]  # value = {'SEN169_FP16': <DataFormats.SEN169_FP16: 0>, 'IEEE_FP32': <DataFormats.IEEE_FP32: 1>, 'INVALID': <DataFormats.INVALID: 2>, 'SEN143_FP8': <DataFormats.SEN143_FP8: 3>, 'SEN152_FP8': <DataFormats.SEN152_FP8: 4>, 'SEN153_FP9': <DataFormats.SEN153_FP9: 5>, 'SENINT2': <DataFormats.SENINT2: 6>, 'SENINT4': <DataFormats.SENINT4: 7>, 'SENINT8': <DataFormats.SENINT8: 8>, 'SENINT16': <DataFormats.SENINT16: 9>, 'SENINT24': <DataFormats.SENINT24: 10>, 'IEEE_INT64': <DataFormats.IEEE_INT64: 11>, 'IEEE_INT32': <DataFormats.IEEE_INT32: 12>, 'SENUINT32': <DataFormats.SENUINT32: 13>, 'SENUINT2': <DataFormats.SENUINT2: 14>, 'IEEE_FP16': <DataFormats.IEEE_FP16: 15>, 'BOOL': <DataFormats.BOOL: 16>, 'BFLOAT16': <DataFormats.BFLOAT16: 17>, 'SEN18F_FP24': <DataFormats.SEN18F_FP24: 18>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: typing.SupportsInt) -> None: ...
    def __str__(self) -> str: ...
    def elems_per_stick(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class ElementArrangement:
    """
    Members:

      STANDARD

      DL16_TO_FP32

      QFP8CH

      EXX2

      QFP8WT
    """

    DL16_TO_FP32: typing.ClassVar[
        ElementArrangement
    ]  # value = <ElementArrangement.DL16_TO_FP32: 1>
    QFP8CH: typing.ClassVar[
        ElementArrangement
    ]  # value = <ElementArrangement.QFP8CH: 2>
    EXX2: typing.ClassVar[ElementArrangement]  # value = <ElementArrangement.EXX2: 3>
    FP32_TO_DL16: typing.ClassVar[
        ElementArrangement
    ]  # value = <ElementArrangement.FP32_TO_DL16: 4>
    QFP8WT: typing.ClassVar[
        ElementArrangement
    ]  # value = <ElementArrangement.QFP8WT: 5>
    STANDARD: typing.ClassVar[
        ElementArrangement
    ]  # value = <ElementArrangement.STANDARD: 0>
    __members__: typing.ClassVar[
        dict[str, ElementArrangement]
    ]  # value = {'STANDARD': <ElementArrangement.STANDARD: 0>, 'DL16_TO_FP32': <ElementArrangement.DL16_TO_FP32: 1>, 'QFP8CH': <ElementArrangement.QFP8CH: 2>, 'EXX2': <ElementArrangement.EXX2: 3>, 'QFP8WT': <ElementArrangement.QFP8WT: 5>}
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class SpyreTensorLayout:
    def __hash__(self) -> int: ...
    def __eq__(self, arg0: SpyreTensorLayout) -> bool: ...  # type: ignore[override]
    @typing.overload
    def __init__(
        self,
        host_size: collections.abc.Sequence[typing.SupportsInt],
        dtype: torch.dtype,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        host_size: collections.abc.Sequence[typing.SupportsInt],
        host_strides: collections.abc.Sequence[typing.SupportsInt],
        dtype: torch.dtype,
        dim_order: collections.abc.Sequence[typing.SupportsInt],
        element_arrangement: ElementArrangement = ElementArrangement.STANDARD,
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        device_size: collections.abc.Sequence[typing.SupportsInt],
        stride_map: collections.abc.Sequence[typing.SupportsInt],
        device_dtype: DataFormats,
        element_arrangement: ElementArrangement = ElementArrangement.STANDARD,
        valid_elements: dict[tuple[int, ...], int] = {},
    ) -> None: ...
    @typing.overload
    def __init__(
        self,
        device_size: collections.abc.Sequence[typing.SupportsInt],
        stride_map: collections.abc.Sequence[typing.SupportsInt],
        device_dtype: DataFormats,
        valid_elements: dict[tuple[int, ...], int],
        element_arrangement: ElementArrangement = ElementArrangement.STANDARD,
    ) -> None: ...
    def __repr__(self) -> str: ...
    def __str__(self) -> str: ...
    def elems_per_stick(self) -> int: ...
    def with_element_arrangement(
        self, element_arrangement: ElementArrangement
    ) -> SpyreTensorLayout: ...
    @property
    def device_dtype(self) -> DataFormats: ...
    @property
    def device_size(self) -> list[int]: ...
    @property
    def element_arrangement(self) -> ElementArrangement: ...
    @property
    def stride_map(self) -> list[int]: ...
    @property
    def valid_elements(self) -> dict[tuple[int, ...], int]: ...

class _SpyreStreamBase:
    """
    C++ SpyreStream wrapper class.

    Represents a stream of execution on a Spyre device.
    """
    def synchronize(self) -> None:
        """Wait for all operations on this stream to complete"""
        ...

    def query(self) -> bool:
        """Check if all operations on this stream have completed"""
        ...

    def device(self) -> torch.device:
        """Get the device associated with this stream"""
        ...

    def id(self) -> int:
        """Get the stream ID"""
        ...

    def priority(self) -> int:
        """Get the stream priority"""
        ...

    def __repr__(self) -> str: ...

def get_stream_from_pool(device: torch.device, priority: int = 0) -> _SpyreStreamBase:
    """
    Get a stream from the pool for the specified device and priority.

    Args:
        device: The device for which to get a stream
        priority: Stream priority (lower = higher priority). Default: 0

    Returns:
        A SpyreStream object from the pool
    """
    ...

def current_stream(device: torch.device) -> _SpyreStreamBase:
    """
    Get the current stream for a device.

    Args:
        device: The device to query

    Returns:
        The current stream for the device
    """
    ...

def default_stream(device: torch.device) -> _SpyreStreamBase:
    """
    Get the default stream for a device.

    Args:
        device: The device to query

    Returns:
        The default stream (stream ID 0) for the device
    """
    ...

def set_current_stream(stream: _SpyreStreamBase) -> None:
    """
    Set the current stream for the stream's device.

    Args:
        stream: The stream to set as current
    """
    ...

def synchronize(device: torch.device | None = None) -> None:
    """
    Synchronize all streams on a device or all devices.

    Args:
        device: The device to synchronize. If None, synchronizes all devices.
    """
    ...

def as_strided_with_layout(
    arg0: torch.Tensor,
    arg1: tuple[int, ...],
    arg2: tuple[int, ...],
    arg3: typing.SupportsInt | None,
    arg4: SpyreTensorLayout,
) -> torch.Tensor: ...
def copy_tensor(
    self: torch.Tensor, dst: torch.Tensor, non_blocking: bool = False
) -> None:
    """
    Copy tensor

    Args:
        self or dst: one of that must be on spyre device
    """
    ...

def copy_tensor_raw(
    dev_tensor: torch.Tensor,
    pool: SharedDataPool,
    slot_id: int,
    to_device: bool,
    non_blocking: bool = False,
) -> None: ...
def copy_kv_page_raw(
    cache: torch.Tensor,
    block_id: int,
    pool: SharedDataPool,
    slot_id: int,
    to_device: bool,
    non_blocking: bool = False,
) -> None: ...
def get_composite_address(tensor: torch.Tensor) -> CompositeAddressHandle: ...
def fill_tensor(self: torch.Tensor, value: float) -> torch.Tensor:
    """
    Fill a spyre tensor with a scalar value using device-side FillDMA.

    Args:
        self: the spyre tensor to fill (in-place)
        value: the fill value (converted to the tensor's dtype pattern)
    """
    ...

def empty_with_layout(
    arg0: tuple[int, ...],
    arg1: SpyreTensorLayout,
    arg2: torch.dtype | None,
    arg3: torch.device | None,
    arg4: bool | None,
    arg5: torch.memory_format | None,
) -> torch.Tensor: ...
def encode_constant(arg0: typing.SupportsFloat, arg1: DataFormats) -> int: ...
def free_runtime() -> None: ...
def get_device_dtype(arg0: torch.dtype) -> DataFormats: ...
def get_downcast_warning() -> bool:
    """
    Return whether downcast warnings are enabled.
    """

def get_elem_in_stick(arg0: torch.dtype) -> int: ...
@typing.overload
def get_device_size_in_bytes(layout: SpyreTensorLayout) -> int: ...
@typing.overload
def get_device_size_in_bytes(
    device_size: typing.Sequence[int], device_dtype: DataFormats
) -> int: ...
def get_spyre_tensor_layout(arg0: torch.Tensor) -> SpyreTensorLayout: ...

class SymbolicArgKind:
    """
    Members:

      kAddress

      kDimension
    """

    kAddress: typing.ClassVar[SymbolicArgKind]
    kDimension: typing.ClassVar[SymbolicArgKind]
    __members__: typing.ClassVar[dict[str, SymbolicArgKind]]
    def __eq__(self, other: typing.Any) -> bool: ...
    def __hash__(self) -> int: ...
    def __init__(self, value: typing.SupportsInt) -> None: ...
    def __int__(self) -> int: ...
    def __repr__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class SymbolicArg:
    kind: SymbolicArgKind
    value: int
    tensor_id: int
    dim_index: int
    def __init__(
        self,
        kind: SymbolicArgKind,
        tensor_id: int,
        dim_index: int = -1,
        value: int = -1,
    ) -> None: ...
    def __repr__(self) -> str: ...

class JobPlan:
    """
    A torch-spyre internal container for executing a unit of work.

    Produced by prepare_kernel() and consumed by launch_jobplan().
    """
    def num_steps(self) -> int:
        """Get the number of steps in the JobPlan"""
        ...

    def job_allocation_size(self) -> int:
        """Get the size of the job allocation"""
        ...

    def get_step_type(self, idx: int) -> str:
        """Get the type of step at the given index (H2D, D2H, Compute, or HostCompute)"""
        ...

    def get_step_name(self, idx: int) -> str | None:
        """Get the profiler-visible name for a compute step, or None"""
        ...

def launch_jobplan(
    job_plan: JobPlan,
    args: collections.abc.Sequence[torch.Tensor],
    symbolic_args: list[SymbolicArg] = ...,
) -> None:
    """
    Launch a prepared JobPlan with the given tensor arguments.

    Args:
        job_plan: The JobPlan to execute
        args: Sequence of input/output tensors
        symbolic_args: Optional typed per-symbol payload. Defaults to empty.
    """
    ...

def prepare_kernel(
    spyrecode_dir: str,
    stream: _SpyreStreamBase | None = None,
    profiler_name: str | None = None,
) -> JobPlan:
    """
    Prepare a kernel from a SpyreCode directory and return a JobPlan.

    Args:
        spyrecode_dir: Path to the SpyreCode directory
        stream: Stream to use for initialization transfers.
            If None, uses the current stream. Defaults to None.
        profiler_name: Bounded base name for profiler-visible compute events.
            If None, uses the existing SpyreCode or directory-derived name.

    Returns:
        Prepared JobPlan ready for execution
    """
    ...

def set_downcast_warning(arg0: bool) -> None:
    """
    Enable/disable downcast warnings for this process.
    """

def set_spyre_tensor_layout(arg0: torch.Tensor, arg1: SpyreTensorLayout) -> None: ...
def spyre_empty_with_layout(
    size: tuple[int, ...],
    stride: tuple[int, ...],
    dtype: torch.dtype,
    device_layout: SpyreTensorLayout,
    device: torch.device | None = None,
) -> torch.Tensor: ...

class SpyreStreamError:
    Success: typing.ClassVar[SpyreStreamError]  # value = <SpyreStreamError.Success: 0>
    Shutdown: typing.ClassVar[
        SpyreStreamError
    ]  # value = <SpyreStreamError.Shutdown: 1>
    __members__: typing.ClassVar[dict[str, SpyreStreamError]]
    def __eq__(self, other: typing.Any) -> bool: ...
    def __hash__(self) -> int: ...
    def __int__(self) -> int: ...
    def __repr__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class SpyreDeviceState:
    Ok: typing.ClassVar[SpyreDeviceState]  # value = <SpyreDeviceState.Ok: 0>
    NotInitialized: typing.ClassVar[
        SpyreDeviceState
    ]  # value = <SpyreDeviceState.NotInitialized: 1>
    StreamError: typing.ClassVar[
        SpyreDeviceState
    ]  # value = <SpyreDeviceState.StreamError: 2>
    __members__: typing.ClassVar[dict[str, SpyreDeviceState]]
    def __eq__(self, other: typing.Any) -> bool: ...
    def __hash__(self) -> int: ...
    def __int__(self) -> int: ...
    def __repr__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

def stream_get_error(stream: _SpyreStreamBase) -> SpyreStreamError: ...
def stream_get_error_string(error: SpyreStreamError) -> str: ...
def get_device_state() -> SpyreDeviceState: ...
def start_runtime() -> None: ...
def to_with_layout(arg0: torch.Tensor, arg1: SpyreTensorLayout) -> torch.Tensor: ...
