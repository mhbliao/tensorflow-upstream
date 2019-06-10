/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/compiler/xla/service/gpu/empty_thunk.h"

#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace gpu {

EmptyThunk::EmptyThunk(const BufferAllocation::Slice& input,
                       const BufferAllocation::Slice& output,
                       const HloInstruction* hlo)
    : Thunk(Kind::kEmpty, hlo), input_(input), output_(output) {}

Status EmptyThunk::Initialize(const GpuExecutable& executable,
                              se::StreamExecutor* executor) {
  return Status::OK();
}

Status EmptyThunk::ExecuteOnStream(
    const BufferAllocations& buffer_allocations, se::Stream* stream,
    HloExecutionProfiler* profiler) {
  se::DeviceMemoryBase input_data = buffer_allocations.GetDeviceAddress(input_);
  se::DeviceMemoryBase output_data =
      buffer_allocations.GetDeviceAddress(output_);
  auto op_profiler = profiler->MakeScopedInstructionProfiler(hlo_instruction());
  LOG(INFO) << "Execute EmptyThunk\n";
  stream->ThenMemcpy(&output_data, input_data, output_data.size());
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
