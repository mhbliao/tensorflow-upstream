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

EmptyThunk::EmptyThunk(const BufferAllocation::Slice& reduce_output_tensor,
                       const BufferAllocation::Slice& output,
                       const HloInstruction* custom_call_hlo,
                       const HloInstruction* hlo,
                       const BufferAllocation::Slice& reduce_input_tensor,
                       int64 reduce_dimension, float init_value)
    : Thunk(Kind::kEmpty, custom_call_hlo), reduce_input_(reduce_input_tensor),
      reduce_output_(reduce_output_tensor), output_(output), hlo_(hlo),
      reduce_dimension_(reduce_dimension), init_value_(init_value) {}

Status EmptyThunk::Initialize(const GpuExecutable& executable,
                              se::StreamExecutor* executor) {
  return Status::OK();
}

Status EmptyThunk::ExecuteOnStream(
    const BufferAllocations& buffer_allocations, se::Stream* stream,
    HloExecutionProfiler* profiler) {
  se::DeviceMemoryBase input_data = buffer_allocations.GetDeviceAddress(reduce_input_);
  se::DeviceMemoryBase reference_output_data = buffer_allocations.GetDeviceAddress(reduce_output_);
  se::DeviceMemoryBase output_data =
      buffer_allocations.GetDeviceAddress(output_);
  auto op_profiler = profiler->MakeScopedInstructionProfiler(hlo_instruction());
  LOG(INFO) << "EmptyThunk for HLO: " << hlo_->ToString();

  LOG(INFO) << "Execute EmptyThunk\n";
  stream->ThenEmpty(input_data, &output_data, init_value_, reduce_dimension_);
  //stream->ThenMemcpy(&output_data, reference_output_data, output_data.size());
  return Status::OK();
}

}  // namespace gpu
}  // namespace xla
