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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_EMPTY_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_EMPTY_THUNK_H_

#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/hlo_execution_profiler.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

// An empty thunk which does nothing
class EmptyThunk : public Thunk {
 public:
  EmptyThunk(const BufferAllocation::Slice& input,
             const BufferAllocation::Slice& output,
             const HloInstruction* custom_call_hlo,
             const HloInstruction* hlo);
  EmptyThunk(const EmptyThunk&) = delete;
  EmptyThunk& operator=(const EmptyThunk&) = delete;

  Status Initialize(const GpuExecutable& executable,
                    se::StreamExecutor* executor) override;
  Status ExecuteOnStream(const BufferAllocations& buffer_allocations,
                         se::Stream* stream,
                         HloExecutionProfiler* profiler) override;
 private:
  const BufferAllocation::Slice input_;
  const BufferAllocation::Slice output_;
  const HloInstruction* hlo_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_EMPTY_THUNK_H_
