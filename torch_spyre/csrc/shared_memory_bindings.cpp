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

#include <flex/memory_interface/shared_data_pool.hpp>
#include <flex/memory_interface/shared_host_pool.hpp>
#include <flex/memory_interface/shared_pool.hpp>
#include <memory>
#include <string>
#include <utility>

#include "module.h"

namespace py = pybind11;

namespace torch_spyre::shared_memory {

void init_shared_memory_bindings(py::module_& m) {
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
}

}  // namespace torch_spyre::shared_memory
