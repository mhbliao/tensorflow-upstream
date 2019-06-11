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

void EmptyKernelLaunch(gpuStream_t gpu_stream) {
  LOG(INFO) << "EmptyKernelLaunch()";

  GpuLaunchConfig config;
  config.virtual_thread_count = 256;
  config.thread_per_block = 256;
  config.block_count = 1;

  GPU_LAUNCH_KERNEL(empty_kernel, dim3(config.block_count), dim3(config.thread_per_block), 0,
                    /* stream */ gpu_stream,
                    /* kernel arguments */ 0, 0, 0);
}

} // namespace tensorflow

#endif // TENSORFLOW_USE_ROCM
