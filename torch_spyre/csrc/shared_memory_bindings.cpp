/*
 * Copyright 2026 The Torch-Spyre Authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "shared_memory_bindings.h"

#include <pybind11/native_enum.h>
#include <pybind11/operators.h>
#include <pybind11/stl.h>
#include <unistd.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <flex/memory_interface/shared_host_pool.hpp>
#include <flex/memory_interface/shared_metadata.hpp>
#include <flex/memory_interface/shared_pool.hpp>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "module.h"

namespace py = pybind11;

namespace torch_spyre::shared_memory {
namespace {

constexpr int kLegacyPoolDescriptorPickleVersion = 1;
constexpr int kPartitionedPoolPickleVersion = 2;
constexpr int kPoolPickleVersion = 3;
constexpr size_t kPoolDescriptorStateSize = 6;
constexpr size_t kPoolConfigStateSize = 4;
constexpr size_t kLegacyPoolChildStateSize = 8;
constexpr size_t kPartitionedPoolChildStateSize = 9;
constexpr size_t kPoolChildStateSize = 10;
constexpr size_t kPoolPartitionStateSize = 6;
constexpr size_t kReservationTokenStateSize = 9;

class PySharedMetadata {
 public:
  explicit PySharedMetadata(std::shared_ptr<flex::SharedMetadata> metadata)
      : metadata_(std::move(metadata)) {}

  flex::SharedMetadata& Get() const {
    if (!metadata_) {
      throw std::runtime_error("SharedMetadata attachment is detached");
    }
    return *metadata_;
  }

  std::shared_ptr<flex::SharedMetadata> Owner() const {
    if (!metadata_) {
      throw std::runtime_error("SharedMetadata attachment is detached");
    }
    return metadata_;
  }

 private:
  std::shared_ptr<flex::SharedMetadata> metadata_;
};

struct PyAttachedDataPool {
  flex::RegisteredPool registration;
  std::shared_ptr<flex::SharedDataPool> data_pool;
};

class PyPoolAttachment {
 public:
  explicit PyPoolAttachment(std::shared_ptr<flex::PoolAttachment> attachment)
      : attachment_(std::move(attachment)) {
    if (!attachment_ || !attachment_->metadata) {
      throw std::runtime_error("PoolAttachment is incomplete");
    }
    metadata_ = std::make_shared<PySharedMetadata>(attachment_->metadata);
    data_pools_.reserve(attachment_->data_pools.size());
    for (const auto& attached : attachment_->data_pools) {
      data_pools_.push_back(
          {attached.registration,
           std::const_pointer_cast<flex::SharedDataPool>(attached.data_pool)});
    }
  }

  uint32_t TpRank() const {
    RequireOpen();
    return attachment_->tp_rank;
  }

  std::shared_ptr<PySharedMetadata> Metadata() const {
    RequireOpen();
    return metadata_;
  }

  std::vector<PyAttachedDataPool> DataPools() const {
    RequireOpen();
    return data_pools_;
  }

  void Close() noexcept {
    if (attachment_) {
      metadata_.reset();
      data_pools_.clear();
      flex::SharedPool::DetachPool(attachment_);
    }
  }

  ~PyPoolAttachment() {
    Close();
  }

 private:
  void RequireOpen() const {
    if (!attachment_) {
      throw std::runtime_error("PoolAttachment is closed");
    }
  }

  std::shared_ptr<flex::PoolAttachment> attachment_;
  std::shared_ptr<PySharedMetadata> metadata_;
  std::vector<PyAttachedDataPool> data_pools_;
};

class PySlotReadPin {
 public:
  PySlotReadPin(flex::SlotReadPin pin,
                std::shared_ptr<flex::SharedMetadata> owner)
      : owner_(std::move(owner)), pin_(std::move(pin)) {}

 private:
  std::shared_ptr<flex::SharedMetadata> owner_;
  flex::SlotReadPin pin_;
};

struct PyAttachResult {
  flex::PoolResultCode code = flex::PoolResultCode::ResourceError;
  std::shared_ptr<PyPoolAttachment> attachment;
  std::string detail;
};

flex::PoolGeneration DecodeGeneration(const py::bytes& value) {
  const std::string bytes = value;
  if (bytes.size() != flex::PoolGeneration{}.bytes.size()) {
    throw py::value_error("pool generation must contain exactly 16 bytes");
  }
  flex::PoolGeneration generation;
  for (size_t index = 0; index < generation.bytes.size(); ++index) {
    generation.bytes[index] = static_cast<uint8_t>(bytes[index]);
  }
  return generation;
}

PyAttachResult AttachDescriptor(const flex::PoolDescriptor& descriptor,
                                uint32_t tp_rank) {
  spyre::startRuntime();
  flex::AttachResult attached;
  {
    py::gil_scoped_release release;
    attached = flex::SharedPool::AttachPool(spyre::GlobalRuntime::get(),
                                            descriptor, tp_rank);
  }
  PyAttachResult result{attached.code, nullptr, std::move(attached.detail)};
  if (attached.attachment) {
    result.attachment =
        std::make_shared<PyPoolAttachment>(std::move(attached.attachment));
  }
  return result;
}

size_t HostPageSize() {
  const int64_t value = sysconf(_SC_PAGESIZE);
  if (value <= 0) {
    throw py::value_error("failed to determine host page size");
  }
  return static_cast<size_t>(value);
}

size_t AlignUp(size_t value, size_t alignment, const char* description) {
  const size_t remainder = value % alignment;
  if (remainder != 0 &&
      value > std::numeric_limits<size_t>::max() - (alignment - remainder)) {
    throw py::value_error(std::string(description) + " alignment overflows");
  }
  return remainder == 0 ? value : value + alignment - remainder;
}

void require_tuple_size(const py::tuple& state, size_t expected,
                        const char* description) {
  if (state.size() != expected) {
    throw py::value_error(std::string(description) + " must contain " +
                          std::to_string(expected) + " fields");
  }
}

flex::PoolState decode_pool_state(uint32_t raw) {
  switch (raw) {
    case static_cast<uint32_t>(flex::PoolState::Initializing):
      return flex::PoolState::Initializing;
    case static_cast<uint32_t>(flex::PoolState::Ready):
      return flex::PoolState::Ready;
    case static_cast<uint32_t>(flex::PoolState::Destroying):
      return flex::PoolState::Destroying;
    default:
      throw py::value_error("PoolDescriptor state is invalid");
  }
}

flex::PoolBackend decode_pool_backend(uint32_t raw) {
  if (raw != static_cast<uint32_t>(flex::PoolBackend::HostDram)) {
    throw py::value_error("PoolDescriptor backend is invalid");
  }
  return flex::PoolBackend::HostDram;
}

template <typename T>
T decode_unsigned(const py::handle& value, const char* description) {
  try {
    const auto decoded = value.cast<uint64_t>();
    if (decoded > std::numeric_limits<T>::max()) {
      throw py::value_error(std::string(description) + " is out of range");
    }
    return static_cast<T>(decoded);
  }
  catch (const py::cast_error&) {
    throw py::value_error(std::string(description) +
                          " must be an unsigned integer");
  }
}

py::list encode_pool_children(
    const std::vector<flex::SharedDataPoolConfig>& data_pools) {
  py::list children;
  for (const auto& child : data_pools) {
    py::list partitions;
    for (const auto& partition : child.partitions) {
      py::tuple partition_state(kPoolPartitionStateSize);
      partition_state[0] = partition.name;
      partition_state[1] = partition.tp_rank;
      partition_state[2] = partition.cache_index;
      partition_state[3] = partition.num_slots;
      partition_state[4] = partition.slot_bytes;
      partition_state[5] = partition.offset_bytes.has_value()
                               ? py::cast(*partition.offset_bytes)
                               : py::none();
      partitions.append(std::move(partition_state));
    }
    py::tuple child_state(kPoolChildStateSize);
    child_state[0] = child.name;
    child_state[1] = static_cast<uint32_t>(child.backend);
    child_state[2] = child.num_slots;
    child_state[3] = child.slot_bytes;
    child_state[4] = child.compatibility.model;
    child_state[5] = child.compatibility.kv_dtype;
    child_state[6] = child.compatibility.block_size;
    child_state[7] = child.compatibility.tensor_parallel_size;
    child_state[8] = child.compatibility.layout_signature;
    child_state[9] = std::move(partitions);
    children.append(std::move(child_state));
  }
  return children;
}

enum class PoolGeometryKind {
  RequestedConfig,
  EffectiveDescriptor,
};

std::vector<flex::SharedDataPoolConfig> decode_pool_children(
    const py::handle& value, int version, PoolGeometryKind geometry_kind) {
  if (!py::isinstance<py::list>(value)) {
    throw py::value_error("PoolConfig children must be a list");
  }
  std::vector<flex::SharedDataPoolConfig> children;
  std::unordered_set<std::string> partition_names;
  for (const py::handle child_handle : value.cast<py::list>()) {
    if (!py::isinstance<py::tuple>(child_handle)) {
      throw py::value_error("PoolConfig child must be a tuple");
    }
    const auto child = py::reinterpret_borrow<py::tuple>(child_handle);
    const size_t expected = version == kLegacyPoolDescriptorPickleVersion
                                ? kLegacyPoolChildStateSize
                                : (version == kPartitionedPoolPickleVersion
                                       ? kPartitionedPoolChildStateSize
                                       : kPoolChildStateSize);
    require_tuple_size(child, expected, "PoolConfig child");
    const bool has_layout_signature = version >= kPoolPickleVersion;
    const size_t partitions_index = has_layout_signature ? 9 : 8;
    flex::SharedDataPoolConfig decoded{
        .name = child[0].cast<std::string>(),
        .backend = decode_pool_backend(
            decode_unsigned<uint32_t>(child[1], "PoolConfig backend")),
        .num_slots = decode_unsigned<size_t>(child[2], "PoolConfig num_slots"),
        .slot_bytes =
            decode_unsigned<size_t>(child[3], "PoolConfig slot_bytes"),
        .partitions = {},
        .compatibility = {.model = child[4].cast<std::string>(),
                          .kv_dtype = child[5].cast<std::string>(),
                          .block_size = decode_unsigned<uint64_t>(
                              child[6], "PoolConfig block_size"),
                          .tensor_parallel_size = decode_unsigned<uint32_t>(
                              child[7], "PoolConfig tensor_parallel_size"),
                          .layout_signature = has_layout_signature
                                                  ? child[8].cast<std::string>()
                                                  : std::string{}}};

    if (version == kLegacyPoolDescriptorPickleVersion) {
      if (decoded.num_slots == 0 || decoded.slot_bytes == 0) {
        throw py::value_error("legacy PoolConfig geometry is invalid");
      }
      if (decoded.num_slots >
          std::numeric_limits<size_t>::max() / decoded.slot_bytes) {
        throw py::value_error("legacy PoolConfig arena size overflows size_t");
      }
      decoded.partitions.push_back({.name = decoded.name,
                                    .tp_rank = 0,
                                    .cache_index = 0,
                                    .num_slots = decoded.num_slots,
                                    .slot_bytes = decoded.slot_bytes,
                                    .offset_bytes = 0});
      decoded.num_slots = 0;
      decoded.slot_bytes = 0;
    } else {
      if (!py::isinstance<py::list>(child[partitions_index])) {
        throw py::value_error("PoolConfig partitions must be a list");
      }
      std::unordered_set<uint64_t> identities;
      for (const py::handle partition_handle :
           child[partitions_index].cast<py::list>()) {
        if (!py::isinstance<py::tuple>(partition_handle)) {
          throw py::value_error("PoolConfig partition must be a tuple");
        }
        const auto partition =
            py::reinterpret_borrow<py::tuple>(partition_handle);
        require_tuple_size(partition, kPoolPartitionStateSize,
                           "PoolConfig partition");
        flex::SharedDataPartitionConfig item{
            .name = partition[0].cast<std::string>(),
            .tp_rank =
                decode_unsigned<uint32_t>(partition[1], "partition TP rank"),
            .cache_index = decode_unsigned<uint32_t>(partition[2],
                                                     "partition cache index"),
            .num_slots =
                decode_unsigned<size_t>(partition[3], "partition num_slots"),
            .slot_bytes =
                decode_unsigned<size_t>(partition[4], "partition slot_bytes"),
            .offset_bytes = std::nullopt};
        if (!partition[5].is_none()) {
          item.offset_bytes =
              decode_unsigned<size_t>(partition[5], "partition offset_bytes");
        }
        if (item.name.empty() || item.num_slots == 0 || item.slot_bytes == 0 ||
            !partition_names.insert(item.name).second) {
          throw py::value_error(
              "PoolConfig partition identity or geometry is invalid");
        }
        const uint64_t identity =
            (static_cast<uint64_t>(item.tp_rank) << 32U) | item.cache_index;
        if (!identities.insert(identity).second) {
          throw py::value_error(
              "PoolConfig partition rank/cache identity is duplicated");
        }
        decoded.partitions.push_back(std::move(item));
      }
      if (decoded.partitions.empty()) {
        if (decoded.num_slots == 0 || decoded.slot_bytes == 0) {
          throw py::value_error("PoolConfig geometry is empty");
        }
        if (decoded.num_slots >
            std::numeric_limits<size_t>::max() / decoded.slot_bytes) {
          throw py::value_error("PoolConfig arena size overflows size_t");
        }
        children.push_back(std::move(decoded));
        continue;
      }

      std::vector<const flex::SharedDataPartitionConfig*> canonical;
      canonical.reserve(decoded.partitions.size());
      for (const auto& partition : decoded.partitions) {
        canonical.push_back(&partition);
      }
      std::sort(canonical.begin(), canonical.end(),
                [](const auto* left, const auto* right) {
                  return std::pair{left->tp_rank, left->cache_index} <
                         std::pair{right->tp_rank, right->cache_index};
                });

      const size_t page_size = HostPageSize();
      size_t arena_bytes = 0;
      for (size_t index = 0; index < canonical.size(); ++index) {
        const auto& item = *canonical[index];
        if (geometry_kind == PoolGeometryKind::EffectiveDescriptor &&
            &item != &decoded.partitions[index]) {
          throw py::value_error(
              "PoolDescriptor partition order is not canonical");
        }
        const size_t slot_bytes =
            geometry_kind == PoolGeometryKind::EffectiveDescriptor
                ? item.slot_bytes
                : AlignUp(item.slot_bytes, page_size,
                          "PoolConfig partition slot size");
        if (geometry_kind == PoolGeometryKind::EffectiveDescriptor &&
            slot_bytes % page_size != 0) {
          throw py::value_error(
              "PoolDescriptor partition slot size is not page aligned");
        }
        if (item.offset_bytes.has_value() &&
            *item.offset_bytes != arena_bytes) {
          throw py::value_error(
              "PoolConfig partition offsets are not canonical");
        }
        if (geometry_kind == PoolGeometryKind::EffectiveDescriptor &&
            !item.offset_bytes.has_value()) {
          throw py::value_error(
              "PoolDescriptor partition offset is not canonical");
        }
        if (item.num_slots > std::numeric_limits<size_t>::max() / slot_bytes) {
          throw py::value_error("PoolConfig partition size overflows size_t");
        }
        const size_t partition_bytes = item.num_slots * slot_bytes;
        if (arena_bytes >
            std::numeric_limits<size_t>::max() - partition_bytes) {
          throw py::value_error("PoolConfig arena size overflows size_t");
        }
        arena_bytes += partition_bytes;
      }

      if (geometry_kind == PoolGeometryKind::EffectiveDescriptor) {
        if (decoded.slot_bytes != page_size ||
            decoded.num_slots != arena_bytes / page_size) {
          throw py::value_error(
              "PoolDescriptor arena summary is not canonical");
        }
      } else if (decoded.num_slots != 0 || decoded.slot_bytes != 0) {
        if (decoded.num_slots == 0 || decoded.slot_bytes != page_size ||
            decoded.num_slots != arena_bytes / page_size) {
          throw py::value_error(
              "PoolConfig flat and partition geometry are inconsistent");
        }
      }
    }
    children.push_back(std::move(decoded));
  }
  return children;
}

py::tuple encode_pool_config(const flex::PoolConfig& config) {
  py::tuple state(kPoolConfigStateSize);
  state[0] = kPoolPickleVersion;
  state[1] = config.name;
  state[2] = config.metadata.max_chunks;
  state[3] = encode_pool_children(config.data_pools);
  return state;
}

flex::PoolConfig decode_pool_config(const py::tuple& state) {
  require_tuple_size(state, kPoolConfigStateSize, "PoolConfig state");
  const int version = state[0].cast<int>();
  if (version != kPartitionedPoolPickleVersion &&
      version != kPoolPickleVersion) {
    throw py::value_error("PoolConfig pickle version is invalid");
  }
  const std::string name = state[1].cast<std::string>();
  flex::MetadataConfig metadata{
      decode_unsigned<uint32_t>(state[2], "PoolConfig max_chunks")};
  return {name, metadata,
          decode_pool_children(state[3], version,
                               PoolGeometryKind::RequestedConfig)};
}

py::tuple encode_pool_descriptor(const flex::PoolDescriptor& descriptor) {
  py::tuple state(kPoolDescriptorStateSize);
  state[0] = kPoolPickleVersion;
  state[1] = descriptor.name;
  state[2] = py::bytes(
      reinterpret_cast<const char*>(descriptor.generation.bytes.data()),
      descriptor.generation.bytes.size());
  state[3] = static_cast<uint32_t>(descriptor.state);
  state[4] = descriptor.effective_config.metadata.max_chunks;
  state[5] = encode_pool_children(descriptor.effective_config.data_pools);
  return state;
}

flex::PoolDescriptor decode_pool_descriptor(const py::tuple& state) {
  require_tuple_size(state, kPoolDescriptorStateSize, "PoolDescriptor state");
  const int version = state[0].cast<int>();
  if (version != kLegacyPoolDescriptorPickleVersion &&
      version != kPartitionedPoolPickleVersion &&
      version != kPoolPickleVersion) {
    throw py::value_error("PoolDescriptor pickle version is invalid");
  }
  if (!py::isinstance<py::bytes>(state[2])) {
    throw py::value_error("PoolDescriptor generation must be bytes");
  }

  const std::string generation_bytes = state[2].cast<std::string>();
  if (generation_bytes.size() != flex::PoolGeneration{}.bytes.size()) {
    throw py::value_error("PoolDescriptor generation must contain 16 bytes");
  }

  flex::PoolGeneration generation;
  for (size_t index = 0; index < generation.bytes.size(); ++index) {
    generation.bytes[index] = static_cast<uint8_t>(generation_bytes[index]);
  }

  const std::string name = state[1].cast<std::string>();
  const auto pool_state = decode_pool_state(
      decode_unsigned<uint32_t>(state[3], "PoolDescriptor state"));
  flex::MetadataConfig metadata{
      decode_unsigned<uint32_t>(state[4], "PoolDescriptor max_chunks")};
  auto children = decode_pool_children(state[5], version,
                                       PoolGeometryKind::EffectiveDescriptor);

  flex::PoolConfig config{name, metadata, std::move(children)};
  return {name, generation, pool_state, std::move(config)};
}

py::tuple encode_reservation_token(const flex::ReservationToken& token) {
  return py::make_tuple(1, token.key.compatibility.metadata_version,
                        token.key.compatibility.compatibility_id,
                        token.key.block_hash, token.slot.pool.metadata_version,
                        token.slot.pool.pool_id, token.slot.pool.pool_version,
                        token.slot.slot_id, token.slot.slot_version);
}

flex::ReservationToken decode_reservation_token(const py::tuple& state) {
  require_tuple_size(state, kReservationTokenStateSize,
                     "ReservationToken state");
  if (state[0].cast<int>() != 1) {
    throw py::value_error("ReservationToken pickle version is invalid");
  }
  const flex::CompatibilityRef compatibility{
      decode_unsigned<uint64_t>(state[1], "compatibility metadata version"),
      decode_unsigned<uint32_t>(state[2], "compatibility ID")};
  const flex::CompatibleBlockKey key{
      compatibility, decode_unsigned<uint64_t>(state[3], "block hash")};
  const flex::PoolRef pool{
      decode_unsigned<uint64_t>(state[4], "pool metadata version"),
      decode_unsigned<uint32_t>(state[5], "pool ID"),
      decode_unsigned<uint64_t>(state[6], "pool version")};
  const flex::SlotRef slot{pool, decode_unsigned<uint64_t>(state[7], "slot ID"),
                           decode_unsigned<uint64_t>(state[8], "slot version")};
  return {key, slot};
}

}  // namespace

void init_shared_memory_bindings(py::module_& module) {
  py::native_enum<flex::PoolBackend>(module, "PoolBackend", "enum.Enum")
      .value("HOST_DRAM", flex::PoolBackend::HostDram)
      .finalize();

  py::native_enum<flex::PoolState>(module, "PoolState", "enum.Enum")
      .value("INITIALIZING", flex::PoolState::Initializing)
      .value("READY", flex::PoolState::Ready)
      .value("DESTROYING", flex::PoolState::Destroying)
      .finalize();

  py::native_enum<flex::PoolResultCode>(module, "PoolResultCode", "enum.Enum")
      .value("OK", flex::PoolResultCode::Ok)
      .value("NOT_FOUND", flex::PoolResultCode::NotFound)
      .value("ALREADY_EXISTS", flex::PoolResultCode::AlreadyExists)
      .value("INITIALIZING", flex::PoolResultCode::Initializing)
      .value("DESTROYING", flex::PoolResultCode::Destroying)
      .value("POOL_BUSY", flex::PoolResultCode::PoolBusy)
      .value("STALE_GENERATION", flex::PoolResultCode::StaleGeneration)
      .value("CONFIGURATION_MISMATCH",
             flex::PoolResultCode::ConfigurationMismatch)
      .value("RESOURCE_ERROR", flex::PoolResultCode::ResourceError)
      .finalize();

  py::native_enum<flex::SharedPoolKind>(module, "SharedPoolKind", "enum.Enum")
      .value("HOST", flex::SharedPoolKind::HOST)
      .finalize();

  py::class_<flex::PoolCompatibility>(module, "PoolCompatibility")
      .def(
          py::init<std::string, std::string, uint64_t, uint32_t, std::string>(),
          py::arg("model"), py::arg("kv_dtype"), py::arg("block_size"),
          py::arg("tensor_parallel_size"), py::arg("layout_signature") = "")
      .def_readonly("model", &flex::PoolCompatibility::model)
      .def_readonly("kv_dtype", &flex::PoolCompatibility::kv_dtype)
      .def_readonly("block_size", &flex::PoolCompatibility::block_size)
      .def_readonly("tensor_parallel_size",
                    &flex::PoolCompatibility::tensor_parallel_size)
      .def_readonly("layout_signature",
                    &flex::PoolCompatibility::layout_signature)
      .def(py::self == py::self);

  py::class_<flex::MetadataConfig>(module, "MetadataConfig")
      .def(py::init<uint32_t>(), py::arg("max_chunks"))
      .def_readonly("max_chunks", &flex::MetadataConfig::max_chunks)
      .def(py::self == py::self);

  py::class_<flex::SharedDataPartitionConfig>(module,
                                              "SharedDataPartitionConfig")
      .def(py::init([](std::string name, uint32_t tp_rank, uint32_t cache_index,
                       size_t num_slots, size_t slot_bytes,
                       std::optional<size_t> offset_bytes) {
             return flex::SharedDataPartitionConfig{
                 .name = std::move(name),
                 .tp_rank = tp_rank,
                 .cache_index = cache_index,
                 .num_slots = num_slots,
                 .slot_bytes = slot_bytes,
                 .offset_bytes = offset_bytes};
           }),
           py::arg("name"), py::arg("tp_rank"), py::arg("cache_index"),
           py::arg("num_slots"), py::arg("slot_bytes"),
           py::arg("offset_bytes") = std::nullopt)
      .def_readonly("name", &flex::SharedDataPartitionConfig::name)
      .def_readonly("tp_rank", &flex::SharedDataPartitionConfig::tp_rank)
      .def_readonly("cache_index",
                    &flex::SharedDataPartitionConfig::cache_index)
      .def_readonly("num_slots", &flex::SharedDataPartitionConfig::num_slots)
      .def_readonly("slot_bytes", &flex::SharedDataPartitionConfig::slot_bytes)
      .def_readonly("offset_bytes",
                    &flex::SharedDataPartitionConfig::offset_bytes)
      .def(py::self == py::self);

  py::class_<flex::SharedDataPoolConfig>(module, "SharedDataPoolConfig")
      .def(py::init([](std::string name, flex::PoolBackend backend,
                       size_t num_slots, size_t slot_bytes,
                       flex::PoolCompatibility compatibility) {
             return flex::SharedDataPoolConfig{
                 .name = std::move(name),
                 .backend = backend,
                 .num_slots = num_slots,
                 .slot_bytes = slot_bytes,
                 .partitions = {},
                 .compatibility = std::move(compatibility)};
           }),
           py::arg("name"), py::arg("backend"), py::arg("num_slots"),
           py::arg("slot_bytes"), py::arg("compatibility"))
      .def(py::init([](std::string name, flex::PoolBackend backend,
                       std::vector<flex::SharedDataPartitionConfig> partitions,
                       flex::PoolCompatibility compatibility) {
             return flex::SharedDataPoolConfig{
                 .name = std::move(name),
                 .backend = backend,
                 .num_slots = 0,
                 .slot_bytes = 0,
                 .partitions = std::move(partitions),
                 .compatibility = std::move(compatibility)};
           }),
           py::arg("name"), py::arg("backend"), py::arg("partitions"),
           py::arg("compatibility"))
      .def(py::init([](std::string name, flex::PoolBackend backend,
                       size_t num_slots, size_t slot_bytes,
                       std::vector<flex::SharedDataPartitionConfig> partitions,
                       flex::PoolCompatibility compatibility) {
             return flex::SharedDataPoolConfig{
                 .name = std::move(name),
                 .backend = backend,
                 .num_slots = num_slots,
                 .slot_bytes = slot_bytes,
                 .partitions = std::move(partitions),
                 .compatibility = std::move(compatibility)};
           }),
           py::arg("name"), py::arg("backend"), py::arg("num_slots"),
           py::arg("slot_bytes"), py::arg("partitions"),
           py::arg("compatibility"))
      .def_readonly("name", &flex::SharedDataPoolConfig::name)
      .def_readonly("backend", &flex::SharedDataPoolConfig::backend)
      .def_readonly("num_slots", &flex::SharedDataPoolConfig::num_slots)
      .def_readonly("slot_bytes", &flex::SharedDataPoolConfig::slot_bytes)
      .def_readonly("partitions", &flex::SharedDataPoolConfig::partitions)
      .def_readonly("compatibility", &flex::SharedDataPoolConfig::compatibility)
      .def(py::self == py::self);

  py::class_<flex::PoolConfig>(module, "PoolConfig")
      .def(py::init<std::string, flex::MetadataConfig,
                    std::vector<flex::SharedDataPoolConfig>>(),
           py::arg("name"), py::arg("metadata"), py::arg("data_pools"))
      .def_readonly("name", &flex::PoolConfig::name)
      .def_readonly("metadata", &flex::PoolConfig::metadata)
      .def_readonly("data_pools", &flex::PoolConfig::data_pools)
      .def(py::self == py::self)
      .def(py::pickle(
          [](const flex::PoolConfig& config) {
            return encode_pool_config(config);
          },
          [](const py::tuple& state) { return decode_pool_config(state); }));

  py::class_<flex::PoolGeneration>(module, "PoolGeneration")
      .def_property_readonly(
          "bytes",
          [](const flex::PoolGeneration& value) {
            return py::bytes(reinterpret_cast<const char*>(value.bytes.data()),
                             value.bytes.size());
          })
      .def(py::self == py::self);

  py::class_<flex::PoolDescriptor>(module, "PoolDescriptor")
      .def_readonly("name", &flex::PoolDescriptor::name)
      .def_readonly("generation", &flex::PoolDescriptor::generation)
      .def_readonly("state", &flex::PoolDescriptor::state)
      .def_readonly("effective_config", &flex::PoolDescriptor::effective_config)
      .def(py::self == py::self)
      .def(py::pickle(
          [](const flex::PoolDescriptor& descriptor) {
            return encode_pool_descriptor(descriptor);
          },
          [](const py::tuple& state) {
            return decode_pool_descriptor(state);
          }));

  py::class_<flex::PoolDescriptorResult>(module, "PoolDescriptorResult")
      .def_readonly("code", &flex::PoolDescriptorResult::code)
      .def_readonly("descriptor", &flex::PoolDescriptorResult::descriptor)
      .def_readonly("detail", &flex::PoolDescriptorResult::detail);

  py::class_<flex::DestroyResult>(module, "DestroyResult")
      .def_readonly("code", &flex::DestroyResult::code)
      .def_readonly("detail", &flex::DestroyResult::detail);

  py::class_<flex::CompatibilityRef>(module, "CompatibilityRef")
      .def_readonly("metadata_version",
                    &flex::CompatibilityRef::metadata_version)
      .def_readonly("compatibility_id",
                    &flex::CompatibilityRef::compatibility_id)
      .def(py::self == py::self);

  py::class_<flex::PoolRef>(module, "PoolRef")
      .def_readonly("metadata_version", &flex::PoolRef::metadata_version)
      .def_readonly("pool_id", &flex::PoolRef::pool_id)
      .def_readonly("pool_version", &flex::PoolRef::pool_version)
      .def(py::self == py::self);

  py::class_<flex::RegisteredPool>(module, "RegisteredPool")
      .def_readonly("pool_ref", &flex::RegisteredPool::pool_ref)
      .def_readonly("compatibility", &flex::RegisteredPool::compatibility)
      .def_readonly("name", &flex::RegisteredPool::name)
      .def_readonly("kind", &flex::RegisteredPool::kind)
      .def_readonly("slot_count", &flex::RegisteredPool::slot_count)
      .def_readonly("slot_bytes", &flex::RegisteredPool::slot_bytes);

  py::class_<flex::CompatibleBlockKey>(module, "CompatibleBlockKey")
      .def(py::init([](flex::CompatibilityRef compatibility,
                       flex::BlockHash block_hash) {
             return flex::CompatibleBlockKey{compatibility, block_hash};
           }),
           py::arg("compatibility"), py::arg("block_hash"))
      .def_readonly("compatibility", &flex::CompatibleBlockKey::compatibility)
      .def_readonly("block_hash", &flex::CompatibleBlockKey::block_hash)
      .def(py::self == py::self);

  py::class_<flex::SlotRef>(module, "SlotRef")
      .def_readonly("pool", &flex::SlotRef::pool)
      .def_readonly("slot_id", &flex::SlotRef::slot_id)
      .def_readonly("slot_version", &flex::SlotRef::slot_version)
      .def(py::self == py::self);

  py::class_<flex::ReservationToken>(module, "ReservationToken")
      .def_readonly("key", &flex::ReservationToken::key)
      .def_readonly("slot", &flex::ReservationToken::slot)
      .def(py::self == py::self)
      .def(py::pickle(
          [](const flex::ReservationToken& token) {
            return encode_reservation_token(token);
          },
          [](const py::tuple& state) {
            return decode_reservation_token(state);
          }));

  py::class_<flex::ChunkDescriptorEntry>(module, "ChunkDescriptorEntry")
      .def(py::init([](uint32_t domain_id, uint64_t size) {
             return flex::ChunkDescriptorEntry{domain_id, size};
           }),
           py::arg("domain_id"), py::arg("size"))
      .def_readonly("domain_id", &flex::ChunkDescriptorEntry::domain_id)
      .def_readonly("size", &flex::ChunkDescriptorEntry::size);

  py::class_<flex::LookupEntry>(module, "LookupEntry")
      .def_readonly("key", &flex::LookupEntry::key)
      .def_readonly("slot", &flex::LookupEntry::slot)
      .def_readonly("chunks", &flex::LookupEntry::chunks);

  py::class_<flex::Reservation>(module, "Reservation")
      .def_property_readonly("key", &flex::Reservation::Key,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("slot", &flex::Reservation::Slot,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("token", &flex::Reservation::Token);

  py::class_<flex::ExistingClaim>(module, "ExistingClaim")
      .def_readonly("slot", &flex::ExistingClaim::slot)
      .def_readonly("valid", &flex::ExistingClaim::valid);

  py::class_<flex::NoSpace>(module, "NoSpace");
  py::class_<flex::Unavailable>(module, "Unavailable");
  py::class_<PySlotReadPin>(module, "SlotReadPin");

  py::class_<flex::SharedDataPool, std::shared_ptr<flex::SharedDataPool>>(
      module, "SharedDataPool")
      .def("name", &flex::SharedDataPool::Name)
      .def("slot_count", &flex::SharedDataPool::SlotCount)
      .def("slot_bytes", &flex::SharedDataPool::SlotBytes)
      .def("total_bytes", &flex::SharedDataPool::TotalBytes);

  py::class_<flex::SharedHostPool, flex::SharedDataPool,
             std::shared_ptr<flex::SharedHostPool>>(module, "SharedHostPool");

  py::class_<PySharedMetadata, std::shared_ptr<PySharedMetadata>>(
      module, "SharedMetadata")
      .def("pool_count",
           [](const PySharedMetadata& self) {
             auto owner = self.Owner();
             py::gil_scoped_release release;
             return owner->PoolCount();
           })
      .def("version",
           [](const PySharedMetadata& self) {
             auto owner = self.Owner();
             py::gil_scoped_release release;
             return owner->Version();
           })
      .def("find_pool",
           [](const PySharedMetadata& self, const std::string& name) {
             auto owner = self.Owner();
             py::gil_scoped_release release;
             return owner->FindPool(name);
           })
      .def("lookup",
           [](const PySharedMetadata& self,
              const flex::CompatibleBlockKey& key) {
             auto owner = self.Owner();
             py::gil_scoped_release release;
             return owner->Lookup(key);
           })
      .def("pin_read",
           [](const PySharedMetadata& self,
              const flex::LookupEntry& entry) -> py::object {
             auto owner = self.Owner();
             auto pin = [&]() {
               py::gil_scoped_release release;
               return owner->PinRead(entry);
             }();
             if (!pin.has_value()) {
               return py::none();
             }
             return py::cast(PySlotReadPin(std::move(*pin), std::move(owner)));
           })
      .def("claim",
           [](const PySharedMetadata& self, const flex::PoolRef& target_pool,
              const flex::CompatibleBlockKey& key) {
             auto owner = self.Owner();
             auto result = [&]() {
               py::gil_scoped_release release;
               return owner->Claim(target_pool, key);
             }();
             return std::visit(
                 [](auto&& value) -> py::object {
                   return py::cast(std::forward<decltype(value)>(value));
                 },
                 std::move(result));
           })
      .def("publish",
           [](const PySharedMetadata& self, flex::Reservation& reservation,
              const flex::ChunkDescriptor& chunks) {
             auto owner = self.Owner();
             py::gil_scoped_release release;
             owner->Publish(reservation, chunks);
           })
      .def("publish",
           [](const PySharedMetadata& self, const flex::ReservationToken& token,
              const flex::ChunkDescriptor& chunks) {
             auto owner = self.Owner();
             py::gil_scoped_release release;
             owner->Publish(token, chunks);
           })
      .def("abort",
           [](const PySharedMetadata& self, flex::Reservation& reservation) {
             auto owner = self.Owner();
             py::gil_scoped_release release;
             owner->Abort(reservation);
           })
      .def("abort",
           [](const PySharedMetadata& self,
              const flex::ReservationToken& token) {
             auto owner = self.Owner();
             py::gil_scoped_release release;
             owner->Abort(token);
           })
      .def("evict",
           [](const PySharedMetadata& self, const flex::LookupEntry& entry) {
             auto owner = self.Owner();
             py::gil_scoped_release release;
             return owner->Evict(entry);
           });

  py::class_<PyAttachedDataPool>(module, "AttachedDataPool")
      .def_property_readonly(
          "registration",
          [](const PyAttachedDataPool& self) { return self.registration; })
      .def_readonly("data_pool", &PyAttachedDataPool::data_pool);

  py::class_<PyPoolAttachment, std::shared_ptr<PyPoolAttachment>>(
      module, "PoolAttachment")
      .def_property_readonly("tp_rank", &PyPoolAttachment::TpRank)
      .def_property_readonly("metadata", &PyPoolAttachment::Metadata)
      .def_property_readonly("data_pools", &PyPoolAttachment::DataPools)
      .def("close", &PyPoolAttachment::Close);

  py::class_<PyAttachResult>(module, "AttachResult")
      .def_readonly("code", &PyAttachResult::code)
      .def_readonly("attachment", &PyAttachResult::attachment)
      .def_readonly("detail", &PyAttachResult::detail);

  py::class_<flex::SharedPool>(module, "SharedPool")
      .def_static("create_pool", &flex::SharedPool::CreatePool,
                  py::call_guard<py::gil_scoped_release>())
      .def_static("open_pool", &flex::SharedPool::OpenPool,
                  py::call_guard<py::gil_scoped_release>())
      .def_static("destroy_pool", &flex::SharedPool::DestroyPool,
                  py::call_guard<py::gil_scoped_release>())
      .def_static(
          "attach_pool",
          [](const flex::PoolDescriptor& descriptor, uint32_t tp_rank) {
            return AttachDescriptor(descriptor, tp_rank);
          },
          py::arg("descriptor"), py::arg("tp_rank"))
      .def_static(
          "attach_pool",
          [](const std::string& name, const py::bytes& expected_generation,
             uint32_t tp_rank) {
            const auto expected = DecodeGeneration(expected_generation);
            flex::PoolDescriptorResult opened;
            {
              py::gil_scoped_release release;
              opened = flex::SharedPool::OpenPool(name);
            }
            if (opened.code != flex::PoolResultCode::Ok) {
              return PyAttachResult{opened.code, nullptr,
                                    std::move(opened.detail)};
            }
            if (opened.descriptor.generation != expected) {
              return PyAttachResult{
                  flex::PoolResultCode::StaleGeneration, nullptr, {}};
            }
            return AttachDescriptor(opened.descriptor, tp_rank);
          },
          py::arg("name"), py::arg("expected_generation"), py::arg("tp_rank"))
      .def_static(
          "attach_pool",
          [](const std::string& name, const py::bytes& expected_generation,
             const flex::PoolConfig& expected_config, uint32_t tp_rank) {
            const auto expected = DecodeGeneration(expected_generation);
            flex::PoolDescriptorResult opened;
            {
              py::gil_scoped_release release;
              opened = flex::SharedPool::OpenPool(name);
            }
            if (opened.code != flex::PoolResultCode::Ok) {
              return PyAttachResult{opened.code, nullptr,
                                    std::move(opened.detail)};
            }
            if (opened.descriptor.generation != expected) {
              return PyAttachResult{
                  flex::PoolResultCode::StaleGeneration, nullptr, {}};
            }
            opened.descriptor.effective_config = expected_config;
            return AttachDescriptor(opened.descriptor, tp_rank);
          },
          py::arg("name"), py::arg("expected_generation"),
          py::arg("expected_config"), py::arg("tp_rank"))
      .def_static("detach_pool",
                  [](const std::shared_ptr<PyPoolAttachment>& attachment) {
                    if (attachment) {
                      attachment->Close();
                    }
                  });

  module.def("_decode_pool_descriptor_state", decode_pool_descriptor,
             py::arg("state"));
  module.def("_decode_pool_config_state", decode_pool_config, py::arg("state"));
  module.def("_decode_reservation_token_state", decode_reservation_token,
             py::arg("state"));
}

}  // namespace torch_spyre::shared_memory
