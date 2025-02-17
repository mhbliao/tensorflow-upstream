/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#endif
#include "tensorflow/core/kernels/fused_batch_norm_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

template struct FusedBatchNormFreezeGrad<Eigen::GpuDevice, float, float>;
template struct FusedBatchNormFreezeGrad<Eigen::GpuDevice, Eigen::half, float>;

template <class T>
__global__ void VarianceToInvVarianceKernel(int nthreads, const T* input,
                                            double epsilon, T* output) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    output[index] = rsqrt(input[index] + T(epsilon));
  }
}

template <class T>
void VarianceToInvVariance<T>::operator()(const Eigen::GpuDevice& d,
                                          const T* variance, double epsilon,
                                          int channels, T* inv_variance) {
  GpuLaunchConfig config = GetGpuLaunchConfig(channels, d);
  TF_CHECK_OK(GpuLaunchKernel(VarianceToInvVarianceKernel<T>,
                               config.block_count, config.thread_per_block, 0,
                               d.stream(), config.virtual_thread_count,
                               variance, epsilon, inv_variance));
}

template <class T>
__global__ void InvVarianceToVarianceKernel(int nthreads, double epsilon,
                                            int sample_size, T* variance) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    T inv_var = variance[index];
#if GOOGLE_CUDA
    T var = __fdividef(1, inv_var * inv_var) - T(epsilon);
#elif TENSORFLOW_USE_ROCM
    // ROCM TODO: fix this in ROCDL or LC
      T var = 1 / (inv_var * inv_var) - T(epsilon);
#endif
    // This is for Bessel's correction
    var *= T(sample_size) / T((sample_size > 1) ? sample_size - 1 : 1);
    variance[index] = (var > 0) ? var : 0;
  }
}

template <class T>
void InvVarianceToVariance<T>::operator()(const Eigen::GpuDevice& d,
                                          double epsilon, int sample_size,
                                          int channels, T* variance) {
  GpuLaunchConfig config = GetGpuLaunchConfig(channels, d);
  TF_CHECK_OK(GpuLaunchKernel(InvVarianceToVarianceKernel<T>,
                               config.block_count, config.thread_per_block, 0,
                               d.stream(), config.virtual_thread_count, epsilon,
                               sample_size, variance));
}

template <class T>
void SetNanFunctor<T>::operator()(const Eigen::GpuDevice& d,
                                  typename TTypes<T>::Flat out) {
  To32Bit(out).device(d) =
      To32Bit(out).constant(Eigen::NumTraits<T>::quiet_NaN());
}

template class VarianceToInvVariance<float>;
template class InvVarianceToVariance<float>;
template class SetNanFunctor<float>;
}  // namespace functor
}  // namespace tensorflow

#else

#include "tensorflow/core/kernels/fused_batch_norm_op.h"

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
