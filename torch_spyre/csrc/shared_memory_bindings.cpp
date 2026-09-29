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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <flex/memory_interface/shared_data_pool.hpp>
#include <flex/memory_interface/shared_host_pool.hpp>
#include <flex/memory_interface/shared_metadata.hpp>
#include <flex/memory_interface/shared_pool.hpp>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "module.h"

namespace py = pybind11;

namespace torch_spyre::shared_memory {

void init_shared_memory_bindings(py::module_& m) {
  py::enum_<flex::SharedPoolKind>(m, "SharedPoolKind")
      .value("HOST", flex::SharedPoolKind::HOST);

  py::class_<flex::CompatibilityDescriptor>(m, "CompatibilityDescriptor")
      .def(py::init([](uint32_t format_version, std::vector<uint8_t> data) {
        return flex::CompatibilityDescriptor{format_version, std::move(data)};
      }))
      .def_readwrite("format_version",
                     &flex::CompatibilityDescriptor::format_version)
      .def_readwrite("data", &flex::CompatibilityDescriptor::data);

  py::class_<flex::SharedDataPoolConfig>(m, "SharedDataPoolConfig")
      .def(py::init([](std::string name, flex::SharedPoolKind kind,
                       size_t num_slots, size_t slot_bytes,
                       flex::CompatibilityDescriptor compatibility) {
        return flex::SharedDataPoolConfig{std::move(name), kind, num_slots,
                                          slot_bytes, std::move(compatibility)};
      }))
      .def_readwrite("name", &flex::SharedDataPoolConfig::name)
      .def_readwrite("kind", &flex::SharedDataPoolConfig::kind)
      .def_readwrite("num_slots", &flex::SharedDataPoolConfig::num_slots)
      .def_readwrite("slot_bytes", &flex::SharedDataPoolConfig::slot_bytes)
      .def_readwrite("compatibility",
                     &flex::SharedDataPoolConfig::compatibility);

  py::class_<flex::SharedMetadataCapacity>(m, "SharedMetadataCapacity")
      .def(py::init([](uint32_t max_pools, uint64_t max_slots_per_pool,
                       uint32_t max_compatibilities) {
        return flex::SharedMetadataCapacity{max_pools, max_slots_per_pool,
                                            max_compatibilities};
      }))
      .def_readwrite("max_pools", &flex::SharedMetadataCapacity::max_pools)
      .def_readwrite("max_slots_per_pool",
                     &flex::SharedMetadataCapacity::max_slots_per_pool)
      .def_readwrite("max_compatibilities",
                     &flex::SharedMetadataCapacity::max_compatibilities);

  py::class_<flex::SharedMetadataConfig>(m, "SharedMetadataConfig")
      .def(py::init([](uint32_t max_chunks,
                       std::vector<flex::SharedDataPoolConfig> pools,
                       std::optional<flex::SharedMetadataCapacity> capacity) {
             return flex::SharedMetadataConfig{max_chunks, std::move(pools),
                                               std::move(capacity)};
           }),
           py::arg("max_chunks"), py::arg("pools"),
           py::arg("capacity") = std::nullopt)
      .def_readwrite("max_chunks", &flex::SharedMetadataConfig::max_chunks)
      .def_readwrite("pools", &flex::SharedMetadataConfig::pools)
      .def_readwrite("capacity", &flex::SharedMetadataConfig::capacity);

  py::class_<flex::CompatibilityRef>(m, "CompatibilityRef")
      .def_readonly("metadata_version",
                    &flex::CompatibilityRef::metadata_version)
      .def_readonly("compatibility_id",
                    &flex::CompatibilityRef::compatibility_id);

  py::class_<flex::PoolRef>(m, "PoolRef")
      .def_readonly("metadata_version", &flex::PoolRef::metadata_version)
      .def_readonly("pool_id", &flex::PoolRef::pool_id)
      .def_readonly("pool_version", &flex::PoolRef::pool_version);

  py::class_<flex::RegisteredPool>(m, "RegisteredPool")
      .def_readonly("pool_ref", &flex::RegisteredPool::pool_ref)
      .def_readonly("compatibility", &flex::RegisteredPool::compatibility)
      .def_readonly("name", &flex::RegisteredPool::name)
      .def_readonly("kind", &flex::RegisteredPool::kind)
      .def_readonly("slot_count", &flex::RegisteredPool::slot_count)
      .def_readonly("slot_bytes", &flex::RegisteredPool::slot_bytes);

  py::class_<flex::CompatibleBlockKey>(m, "CompatibleBlockKey")
      .def(py::init(
          [](flex::CompatibilityRef compatibility, flex::BlockHash block_hash) {
            return flex::CompatibleBlockKey{compatibility, block_hash};
          }))
      .def_readonly("compatibility", &flex::CompatibleBlockKey::compatibility)
      .def_readonly("block_hash", &flex::CompatibleBlockKey::block_hash);

  py::class_<flex::SlotRef>(m, "SlotRef")
      .def_readonly("pool", &flex::SlotRef::pool)
      .def_readonly("slot_id", &flex::SlotRef::slot_id)
      .def_readonly("slot_version", &flex::SlotRef::slot_version);

  py::class_<flex::ChunkDescriptorEntry>(m, "ChunkDescriptorEntry")
      .def(py::init([](uint32_t domain_id, uint64_t size) {
        return flex::ChunkDescriptorEntry{domain_id, size};
      }))
      .def_readonly("domain_id", &flex::ChunkDescriptorEntry::domain_id)
      .def_readonly("size", &flex::ChunkDescriptorEntry::size);

  py::class_<flex::LookupEntry>(m, "LookupEntry")
      .def_readonly("key", &flex::LookupEntry::key)
      .def_readonly("slot", &flex::LookupEntry::slot)
      .def_readonly("chunks", &flex::LookupEntry::chunks);

  py::class_<flex::Reservation>(m, "Reservation")
      .def_property_readonly("key", &flex::Reservation::Key,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("slot", &flex::Reservation::Slot,
                             py::return_value_policy::reference_internal);

  py::class_<flex::ExistingClaim>(m, "ExistingClaim")
      .def_readonly("slot", &flex::ExistingClaim::slot)
      .def_readonly("valid", &flex::ExistingClaim::valid);

  py::class_<flex::NoSpace>(m, "NoSpace");
  py::class_<flex::Unavailable>(m, "Unavailable");

  py::class_<flex::SlotReadPin>(
      m, "SlotReadPin",
      "Pins one slot version through blocking H2D completion. Destroy the "
      "pin on the thread that acquired it.");

  py::class_<flex::SharedPool, std::shared_ptr<flex::SharedPool>>(m,
                                                                  "SharedPool")
      .def("name", &flex::SharedPool::Name);

  py::class_<flex::SharedDataPool, flex::SharedPool,
             std::shared_ptr<flex::SharedDataPool>>(m, "SharedDataPool")
      .def("slot_count", &flex::SharedDataPool::SlotCount)
      .def("slot_bytes", &flex::SharedDataPool::SlotBytes)
      .def("total_bytes", &flex::SharedDataPool::TotalBytes);

  py::class_<flex::SharedHostPool, flex::SharedDataPool,
             std::shared_ptr<flex::SharedHostPool>>(m, "SharedHostPool")
      .def_static(
          "create_or_attach",
          [](const std::string& name, size_t num_slots, size_t slot_bytes) {
            spyre::startRuntime();
            py::gil_scoped_release release;
            auto pool = flex::SharedHostPool::CreateOrAttach(
                spyre::GlobalRuntime::get(), name, num_slots, slot_bytes);
            return std::shared_ptr<flex::SharedHostPool>(std::move(pool));
          },
          py::arg("name"), py::arg("num_slots"), py::arg("slot_bytes"))
      .def_static("unlink_by_name", &flex::SharedHostPool::UnlinkByName,
                  py::arg("name"), py::call_guard<py::gil_scoped_release>());

  py::class_<flex::SharedMetadata, flex::SharedPool,
             std::shared_ptr<flex::SharedMetadata>>(
      m, "SharedMetadata",
      "Thin binding for Flex's process-shared metadata directory.\n\n"
      "Write protocol: claim -> blocking D2H copy -> publish. Read protocol: "
      "lookup -> pin_read -> blocking H2D copy -> destroy the pin on the "
      "acquiring thread. An unpublished reservation is a miss. Abort only "
      "before DMA submission or after submitted DMA is quiescent. A slot "
      "version is a wrapping 64-bit tenancy detector, not a lifetime-unique "
      "ID. No binding holds the directory lock across DMA. No raw pointer or "
      "address is exposed. NoSpace requires caller-selected eviction and "
      "retry.")
      .def_static(
          "create_or_attach",
          [](const std::string& name,
             const flex::SharedMetadataConfig& config) {
            spyre::startRuntime();
            const auto config_snapshot = config;
            py::gil_scoped_release release;
            auto metadata = flex::SharedMetadata::CreateOrAttach(
                spyre::GlobalRuntime::get(), name, config_snapshot);
            return std::shared_ptr<flex::SharedMetadata>(std::move(metadata));
          },
          py::arg("name"), py::arg("config"))
      .def_static("unlink_by_name", &flex::SharedMetadata::UnlinkByName,
                  py::arg("name"), py::call_guard<py::gil_scoped_release>())
      .def("pool_count", &flex::SharedMetadata::PoolCount,
           py::call_guard<py::gil_scoped_release>())
      .def("version", &flex::SharedMetadata::Version)
      .def("find_pool", &flex::SharedMetadata::FindPool, py::arg("name"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "register_or_attach_pool",
          [](flex::SharedMetadata& metadata,
             const flex::SharedDataPoolConfig& config) {
            const auto config_snapshot = config;
            py::gil_scoped_release release;
            return metadata.RegisterOrAttachPool(config_snapshot);
          },
          py::arg("config"))
      .def(
          "resolve_pool",
          [](flex::SharedMetadata& metadata, const flex::PoolRef& ref) {
            auto resolved = [&]() {
              py::gil_scoped_release release;
              return metadata.ResolvePool(ref);
            }();
            return std::const_pointer_cast<flex::SharedDataPool>(resolved);
          },
          py::arg("pool_ref"))
      .def("retire_pool", &flex::SharedMetadata::RetirePool,
           py::arg("pool_ref"), py::call_guard<py::gil_scoped_release>(),
           "Retire a pool only after the caller establishes host-wide DMA "
           "quiescence. Retirement does not drain read pins or synchronize "
           "DMA.")
      .def("lookup", &flex::SharedMetadata::Lookup, py::arg("key"),
           py::call_guard<py::gil_scoped_release>())
      .def(
          "pin_read",
          [](flex::SharedMetadata& metadata,
             const flex::LookupEntry& entry) -> py::object {
            auto pin = [&]() {
              py::gil_scoped_release release;
              return metadata.PinRead(entry);
            }();
            if (!pin.has_value()) {
              return py::none();
            }
            return py::cast(std::move(*pin));
          },
          py::arg("entry"), py::keep_alive<0, 1>())
      .def(
          "claim",
          [](flex::SharedMetadata& metadata, const flex::PoolRef& target_pool,
             const flex::CompatibleBlockKey& key) {
            auto result = [&]() {
              py::gil_scoped_release release;
              return metadata.Claim(target_pool, key);
            }();
            return std::visit(
                [](auto&& value) -> py::object {
                  return py::cast(std::forward<decltype(value)>(value));
                },
                std::move(result));
          },
          py::arg("target_pool"), py::arg("key"), py::keep_alive<0, 1>(),
          "Reserve a slot for a key. The caller must publish or abort every "
          "returned Reservation.")
      .def("publish", &flex::SharedMetadata::Publish, py::arg("reservation"),
           py::arg("chunks"), py::call_guard<py::gil_scoped_release>(),
           "Publish a reservation only after its D2H DMA has synchronized.")
      .def("abort", &flex::SharedMetadata::Abort, py::arg("reservation"),
           py::call_guard<py::gil_scoped_release>(),
           "Abort only before DMA submission or after submitted DMA is "
           "quiescent.")
      .def("evict", &flex::SharedMetadata::Evict, py::arg("entry"),
           py::call_guard<py::gil_scoped_release>());
}

}  // namespace torch_spyre::shared_memory
