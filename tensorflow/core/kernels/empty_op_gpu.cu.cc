#if TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "external/rocprim_archive/hipcub/include/hipcub/hipcub.hpp"
#include "tensorflow/core/lib/core/bits.h"
#include "tensorflow/core/util/gpu_device_functions.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
namespace gpuprim = ::hipcub;


namespace tensorflow {

__global__ void empty_kernel(int a, int b, int c) {
}

// each block does a grid strided loop and reduces its values locally
// the case of one block is used for low latency small reductions to scalars
__global__ void BlockReduceKernel(
    const float* in, float* out, int num_elems, float initVal) {
  const int bid = blockIdx.x;
  const int tid = threadIdx.x;

  const int gid = bid * blockDim.x + tid;
  const int stride = blockDim.x * gridDim.x;

  float sum = initVal;
  if (gid < num_elems) {
    sum = in[gid];
    for (int pos = gid + stride; pos < num_elems; pos += stride) {
      sum = sum + in[pos];
    }
  }

  typedef gpuprim::BlockReduce<float, 256> BlockReduce;

  __shared__ typename BlockReduce::TempStorage temp_storage;

  // only include input values in the reduction
  //
  // elements: -----------------
  // grid:     |====|====|====|====|====|
  const int num_elements_to_reduce =
      max(min(256 - bid * blockDim.x, 256), 0);

  sum = BlockReduce(temp_storage).Reduce(sum, gpuprim::Sum(), num_elements_to_reduce);

  if (tid == 0) out[bid] = sum;
}

void EmptyKernelLaunch(gpuStream_t gpu_stream,
                       const se::DeviceMemoryBase& input,
                       se::DeviceMemoryBase* output,
                       float init_value,
                       int64 reduction_dimension) {
  LOG(INFO) << "EmptyKernelLaunch()";
  LOG(INFO) << "input device memory size: " << input.size();
  LOG(INFO) << "output device memory size: " << output->size();
  LOG(INFO) << "init value: " << init_value;
  LOG(INFO) << "reudction dimension: " << reduction_dimension;

#if 1
  const int num_blocks = 1;
  const int num_threads = 256;
  for (int i = 0; i < 256; ++i) {
    GPU_LAUNCH_KERNEL(BlockReduceKernel,
        dim3(num_blocks), dim3(num_threads), 0, gpu_stream,
        reinterpret_cast<const float*>(input.opaque()) + (256 * i),
        reinterpret_cast<float*>(output->opaque()) + i,
        256,
        init_value);
  }
#endif 

#if 0
  GpuLaunchConfig config;
  config.virtual_thread_count = 256;
  config.thread_per_block = 256;
  config.block_count = 1;

  GPU_LAUNCH_KERNEL(empty_kernel, dim3(config.block_count), dim3(config.thread_per_block), 0,
                    /* stream */ gpu_stream,
                    /* kernel arguments */ 0, 0, 0);
#endif
}

} // namespace tensorflow

#endif // TENSORFLOW_USE_ROCM
