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

#include <cstddef>
#include <cstdint>
#include <flex/memory_interface/shared_host_pool.hpp>
#include <flex/memory_interface/shared_metadata.hpp>
#include <flex/memory_interface/shared_pool.hpp>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "module.h"

namespace py = pybind11;

namespace torch_spyre::shared_memory {
namespace {

constexpr int kPoolDescriptorPickleVersion = 1;
constexpr size_t kPoolDescriptorStateSize = 6;
constexpr size_t kPoolChildStateSize = 8;

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

  void Detach() noexcept {
    if (metadata_) {
      flex::SharedPool::DetachPool(std::move(metadata_));
    }
  }

  ~PySharedMetadata() {
    Detach();
  }

 private:
  std::shared_ptr<flex::SharedMetadata> metadata_;
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
  std::shared_ptr<PySharedMetadata> metadata;
  std::string detail;
};

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

py::tuple encode_pool_descriptor(const flex::PoolDescriptor& descriptor) {
  py::list children;
  for (const auto& child : descriptor.effective_config.data_pools) {
    py::tuple child_state(kPoolChildStateSize);
    child_state[0] = child.name;
    child_state[1] = static_cast<uint32_t>(child.backend);
    child_state[2] = child.num_slots;
    child_state[3] = child.slot_bytes;
    child_state[4] = child.compatibility.model;
    child_state[5] = child.compatibility.kv_dtype;
    child_state[6] = child.compatibility.block_size;
    child_state[7] = child.compatibility.tensor_parallel_size;
    children.append(std::move(child_state));
  }

  py::tuple state(kPoolDescriptorStateSize);
  state[0] = kPoolDescriptorPickleVersion;
  state[1] = descriptor.name;
  state[2] = py::bytes(
      reinterpret_cast<const char*>(descriptor.generation.bytes.data()),
      descriptor.generation.bytes.size());
  state[3] = static_cast<uint32_t>(descriptor.state);
  state[4] = descriptor.effective_config.metadata.max_chunks;
  state[5] = std::move(children);
  return state;
}

flex::PoolDescriptor decode_pool_descriptor(const py::tuple& state) {
  require_tuple_size(state, kPoolDescriptorStateSize, "PoolDescriptor state");
  if (state[0].cast<int>() != kPoolDescriptorPickleVersion) {
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
  const auto pool_state = decode_pool_state(state[3].cast<uint32_t>());
  flex::MetadataConfig metadata{state[4].cast<uint32_t>()};

  if (!py::isinstance<py::list>(state[5])) {
    throw py::value_error("PoolDescriptor children must be a list");
  }
  std::vector<flex::SharedDataPoolConfig> children;
  for (const py::handle child_handle : state[5].cast<py::list>()) {
    if (!py::isinstance<py::tuple>(child_handle)) {
      throw py::value_error("PoolDescriptor child must be a tuple");
    }
    const auto child = py::reinterpret_borrow<py::tuple>(child_handle);
    require_tuple_size(child, kPoolChildStateSize, "PoolDescriptor child");
    children.push_back({
        child[0].cast<std::string>(),
        decode_pool_backend(child[1].cast<uint32_t>()),
        child[2].cast<size_t>(),
        child[3].cast<size_t>(),
        {child[4].cast<std::string>(), child[5].cast<std::string>(),
         child[6].cast<uint64_t>(), child[7].cast<uint32_t>()},
    });
  }

  flex::PoolConfig config{name, metadata, std::move(children)};
  return {name, generation, pool_state, std::move(config)};
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
      .def(py::init<std::string, std::string, uint64_t, uint32_t>(),
           py::arg("model"), py::arg("kv_dtype"), py::arg("block_size"),
           py::arg("tensor_parallel_size"))
      .def_readonly("model", &flex::PoolCompatibility::model)
      .def_readonly("kv_dtype", &flex::PoolCompatibility::kv_dtype)
      .def_readonly("block_size", &flex::PoolCompatibility::block_size)
      .def_readonly("tensor_parallel_size",
                    &flex::PoolCompatibility::tensor_parallel_size)
      .def(py::self == py::self);

  py::class_<flex::MetadataConfig>(module, "MetadataConfig")
      .def(py::init<uint32_t>(), py::arg("max_chunks"))
      .def_readonly("max_chunks", &flex::MetadataConfig::max_chunks)
      .def(py::self == py::self);

  py::class_<flex::SharedDataPoolConfig>(module, "SharedDataPoolConfig")
      .def(py::init<std::string, flex::PoolBackend, size_t, size_t,
                    flex::PoolCompatibility>(),
           py::arg("name"), py::arg("backend"), py::arg("num_slots"),
           py::arg("slot_bytes"), py::arg("compatibility"))
      .def_readonly("name", &flex::SharedDataPoolConfig::name)
      .def_readonly("backend", &flex::SharedDataPoolConfig::backend)
      .def_readonly("num_slots", &flex::SharedDataPoolConfig::num_slots)
      .def_readonly("slot_bytes", &flex::SharedDataPoolConfig::slot_bytes)
      .def_readonly("compatibility", &flex::SharedDataPoolConfig::compatibility)
      .def(py::self == py::self);

  py::class_<flex::PoolConfig>(module, "PoolConfig")
      .def(py::init<std::string, flex::MetadataConfig,
                    std::vector<flex::SharedDataPoolConfig>>(),
           py::arg("name"), py::arg("metadata"), py::arg("data_pools"))
      .def_readonly("name", &flex::PoolConfig::name)
      .def_readonly("metadata", &flex::PoolConfig::metadata)
      .def_readonly("data_pools", &flex::PoolConfig::data_pools)
      .def(py::self == py::self);

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
      .def_readonly("block_hash", &flex::CompatibleBlockKey::block_hash);

  py::class_<flex::SlotRef>(module, "SlotRef")
      .def_readonly("pool", &flex::SlotRef::pool)
      .def_readonly("slot_id", &flex::SlotRef::slot_id)
      .def_readonly("slot_version", &flex::SlotRef::slot_version);

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
                             py::return_value_policy::reference_internal);

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
      .def("resolve_pool",
           [](const PySharedMetadata& self, const flex::PoolRef& pool_ref) {
             auto owner = self.Owner();
             auto resolved = [&]() {
               py::gil_scoped_release release;
               return owner->ResolvePool(pool_ref);
             }();
             return std::const_pointer_cast<flex::SharedDataPool>(resolved);
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
      .def("abort",
           [](const PySharedMetadata& self, flex::Reservation& reservation) {
             auto owner = self.Owner();
             py::gil_scoped_release release;
             owner->Abort(reservation);
           })
      .def("evict",
           [](const PySharedMetadata& self, const flex::LookupEntry& entry) {
             auto owner = self.Owner();
             py::gil_scoped_release release;
             return owner->Evict(entry);
           });

  py::class_<PyAttachResult>(module, "AttachResult")
      .def_readonly("code", &PyAttachResult::code)
      .def_readonly("metadata", &PyAttachResult::metadata)
      .def_readonly("detail", &PyAttachResult::detail);

  py::class_<flex::SharedPool>(module, "SharedPool")
      .def_static("create_pool", &flex::SharedPool::CreatePool,
                  py::call_guard<py::gil_scoped_release>())
      .def_static("open_pool", &flex::SharedPool::OpenPool,
                  py::call_guard<py::gil_scoped_release>())
      .def_static("destroy_pool", &flex::SharedPool::DestroyPool,
                  py::call_guard<py::gil_scoped_release>())
      .def_static("attach_pool",
                  [](const flex::PoolDescriptor& descriptor) {
                    spyre::startRuntime();
                    flex::AttachResult attached;
                    {
                      py::gil_scoped_release release;
                      attached = flex::SharedPool::AttachPool(
                          spyre::GlobalRuntime::get(), descriptor);
                    }
                    PyAttachResult result{attached.code, nullptr,
                                          std::move(attached.detail)};
                    if (attached.metadata) {
                      result.metadata = std::make_shared<PySharedMetadata>(
                          std::move(attached.metadata));
                    }
                    return result;
                  })
      .def_static("detach_pool",
                  [](const std::shared_ptr<PySharedMetadata>& metadata) {
                    if (metadata) {
                      metadata->Detach();
                    }
                  });

  module.def("_decode_pool_descriptor_state", decode_pool_descriptor,
             py::arg("state"));
}

}  // namespace torch_spyre::shared_memory
