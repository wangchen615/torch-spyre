/*
 * Copyright 2025 The Torch-Spyre Authors.
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

#pragma once

#include <pybind11/pybind11.h>
#include <spyrecode-host-functions/sendataconvert/sen_host_ops.h>
#include <torch/csrc/utils/pybind.h>

#include <flex/flex.hpp>
#include <memory>

using DataConversionStrideInfo = data_conversion_stride_info;
using DataConversionInfo = data_conversion_info;

namespace spyre {

class GlobalRuntime {
 public:
  static void set(flex::RuntimeContext* runtime) {
    instance() = runtime;
  }

  static void reset() {
    instance() = nullptr;
  }

  static flex::RuntimeContext* get() {
    return instance();
  }

 private:
  GlobalRuntime() = delete;
  ~GlobalRuntime() = delete;

  static flex::RuntimeContext*& instance() {
    static flex::RuntimeContext* s = nullptr;
    return s;
  }
};

// Borrowed shared_ptr to the global RuntimeContext, for APIs that ask for one.
//
// spyre_comms::initialize_library takes std::shared_ptr<flex::RuntimeContext>
// and stores it for the library's lifetime, but the context is NOT ours to
// hand out ownership of: flex::RuntimeContext::create() documents its result
// as a "Non-owning pointer to the RuntimeContext singleton", flex owns it, and
// GlobalRuntime only borrows it (freeRuntime() nulls the handle, it does not
// delete). Wrapping it in a default-deleter shared_ptr would let the last
// external reference call delete on a flex-owned singleton -- heap corruption
// at teardown, in a path that otherwise looks correct.
//
// Hence the no-op deleter: the callee gets a valid, copyable shared_ptr whose
// destructor does nothing. Returns nullptr when the runtime is not yet started,
// which callers must treat as an error (spyre_comms rejects a null runtime).
inline std::shared_ptr<flex::RuntimeContext> borrowed_runtime_context() {
  flex::RuntimeContext* runtime = GlobalRuntime::get();
  if (runtime == nullptr) return nullptr;
  return std::shared_ptr<flex::RuntimeContext>(runtime,
                                               [](flex::RuntimeContext*) {});
}
bool get_downcast_warn_enabled();
bool is_supported_dtype(c10::ScalarType dtype);
DataFormats get_device_dtype(c10::ScalarType torch_dtype);

int device_count();
void startRuntime();
}  // namespace spyre
