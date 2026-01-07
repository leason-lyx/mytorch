#include "pooling.cuh"
#include "tensor.cuh"

#include <cfloat>
#include <stdexcept>
#include <string>

constexpr int kCudaThreadsNum = 512;

inline int CudaGetBlocks(const int N) {
    return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}

#define CUDA_KERNEL_LOOP(i, n)                                                                     \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define CUDA_CHECK(expr)                                                                           \
    do {                                                                                           \
        cudaError_t _err = (expr);                                                                 \
        if (_err != cudaSuccess) {                                                                 \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err));      \
        }                                                                                          \
    } while (0)


// Forward pass (CPU)
void max_pool2d_fwd_cpu(const float* input,
                        float* output,
                        size_t batch_size,
                        size_t in_channels,
                        size_t in_height,
                        size_t in_width,
                        size_t pool_height,
                        size_t pool_width,
                        size_t stride_height,
                        size_t stride_width,
                        size_t out_height,
                        size_t out_width) {
    for (size_t n = 0; n < batch_size; ++n) {
        for (size_t c = 0; c < in_channels; ++c) {
            for (size_t oh = 0; oh < out_height; ++oh) {
                for (size_t ow = 0; ow < out_width; ++ow) {
                    float max_val = -FLT_MAX;
                    size_t h_start = oh * stride_height;
                    size_t w_start = ow * stride_width;
                    for (size_t ph = 0; ph < pool_height; ++ph) {
                        for (size_t pw = 0; pw < pool_width; ++pw) {
                            size_t ih = h_start + ph;
                            size_t iw = w_start + pw;
                            if (ih < in_height && iw < in_width) {
                                float val = input[n * in_channels * in_height * in_width + c * in_height * in_width + ih * in_width + iw];
                                if (val > max_val) {
                                    max_val = val;
                                }
                            }
                        }
                    }
                    output[n * in_channels * out_height * out_width + c * out_height * out_width + oh * out_width + ow] = max_val;
                }
            }
        }
    }
}

// Backward pass (CPU)
void max_pool2d_bwd_cpu(const float* input,
                        const float* grad_output,
                        size_t batch_size,
                        size_t in_channels,
                        size_t in_height,
                        size_t in_width,
                        size_t pool_height,
                        size_t pool_width,
                        size_t stride_height,
                        size_t stride_width,
                        size_t out_height,
                        size_t out_width,
                        float* grad_input) {
    for (size_t i = 0; i < batch_size * in_channels * in_height * in_width; ++i) {
        grad_input[i] = 0.0f;
    }

    for (size_t n = 0; n < batch_size; ++n) {
        for (size_t c = 0; c < in_channels; ++c) {
            for (size_t oh = 0; oh < out_height; ++oh) {
                for (size_t ow = 0; ow < out_width; ++ow) {
                    float max_val = -FLT_MAX;
                    int max_ih = -1, max_iw = -1;
                    size_t h_start = oh * stride_height;
                    size_t w_start = ow * stride_width;

                    for (size_t ph = 0; ph < pool_height; ++ph) {
                        for (size_t pw = 0; pw < pool_width; ++pw) {
                            size_t ih = h_start + ph;
                            size_t iw = w_start + pw;
                            if (ih < in_height && iw < in_width) {
                                float val = input[n * in_channels * in_height * in_width + c * in_height * in_width + ih * in_width + iw];
                                if (val > max_val) {
                                    max_val = val;
                                    max_ih = ih;
                                    max_iw = iw;
                                }
                            }
                        }
                    }

                    if (max_ih != -1) {
                        grad_input[n * in_channels * in_height * in_width + c * in_height * in_width + max_ih * in_width + max_iw] +=
                            grad_output[n * in_channels * out_height * out_width + c * out_height * out_width + oh * out_width + ow];
                    }
                }
            }
        }
    }
}


// Forward pass (CUDA kernel)
__global__ void max_pool2d_fwd_kernel(const float* input,
                                      float* output,
                                      size_t batch_size,
                                      size_t in_channels,
                                      size_t in_height,
                                      size_t in_width,
                                      size_t pool_height,
                                      size_t pool_width,
                                      size_t stride_height,
                                      size_t stride_width,
                                      size_t out_height,
                                      size_t out_width,
                                      size_t num_elements) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elements; i += blockDim.x * gridDim.x) {
        size_t ow = i % out_width;
        size_t oh = (i / out_width) % out_height;
        size_t c = (i / (out_width * out_height)) % in_channels;
        size_t n = i / (out_width * out_height * in_channels);

        float max_val = -FLT_MAX;
        size_t h_start = oh * stride_height;
        size_t w_start = ow * stride_width;

        for (size_t ph = 0; ph < pool_height; ++ph) {
            for (size_t pw = 0; pw < pool_width; ++pw) {
                size_t ih = h_start + ph;
                size_t iw = w_start + pw;
                if (ih < in_height && iw < in_width) {
                    float val = input[n * in_channels * in_height * in_width + c * in_height * in_width + ih * in_width + iw];
                    if (val > max_val) {
                        max_val = val;
                    }
                }
            }
        }
        output[i] = max_val;
    }
}

// Backward pass (CUDA kernel)
__global__ void max_pool2d_bwd_kernel(const float* input,
                                      const float* grad_output,
                                      size_t batch_size,
                                      size_t in_channels,
                                      size_t in_height,
                                      size_t in_width,
                                      size_t pool_height,
                                      size_t pool_width,
                                      size_t stride_height,
                                      size_t stride_width,
                                      size_t out_height,
                                      size_t out_width,
                                      size_t num_elements,
                                      float* grad_input) {
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_elements; i += blockDim.x * gridDim.x) {
        size_t ow = i % out_width;
        size_t oh = (i / out_width) % out_height;
        size_t c = (i / (out_width * out_height)) % in_channels;
        size_t n = i / (out_width * out_height * in_channels);

        float max_val = -FLT_MAX;
        int max_ih = -1, max_iw = -1;
        size_t h_start = oh * stride_height;
        size_t w_start = ow * stride_width;

        for (size_t ph = 0; ph < pool_height; ++ph) {
            for (size_t pw = 0; pw < pool_width; ++pw) {
                size_t ih = h_start + ph;
                size_t iw = w_start + pw;
                if (ih < in_height && iw < in_width) {
                    float val = input[n * in_channels * in_height * in_width + c * in_height * in_width + ih * in_width + iw];
                    if (val > max_val) {
                        max_val = val;
                        max_ih = ih;
                        max_iw = iw;
                    }
                }
            }
        }

        if (max_ih != -1) {
            atomicAdd(&grad_input[n * in_channels * in_height * in_width + c * in_height * in_width + max_ih * in_width + max_iw],
                      grad_output[i]);
        }
    }
}


// Pointer interface implementations
void max_pool2d_fwd(const float* input,
                    float* output,
                    size_t batch_size,
                    size_t in_channels,
                    size_t in_height,
                    size_t in_width,
                    size_t pool_height,
                    size_t pool_width,
                    size_t stride_height,
                    size_t stride_width,
                    size_t out_height,
                    size_t out_width,
                    Device device) {
    if (device.is_cpu()) {
        max_pool2d_fwd_cpu(input, output, batch_size, in_channels, in_height, in_width, pool_height, pool_width, stride_height, stride_width, out_height, out_width);
    } else {
        size_t num_elements = batch_size * in_channels * out_height * out_width;
        max_pool2d_fwd_kernel<<<CudaGetBlocks(num_elements), kCudaThreadsNum>>>(
            input, output, batch_size, in_channels, in_height, in_width, pool_height, pool_width, stride_height, stride_width, out_height, out_width, num_elements);
        cudaDeviceSynchronize();
    }
}

void max_pool2d_bwd(const float* input,
                    const float* output,
                    const float* grad_output,
                    size_t batch_size,
                    size_t in_channels,
                    size_t in_height,
                    size_t in_width,
                    size_t pool_height,
                    size_t pool_width,
                    size_t stride_height,
                    size_t stride_width,
                    size_t out_height,
                    size_t out_width,
                    Device device,
                    float* grad_input) {
    if (device.is_cpu()) {
        max_pool2d_bwd_cpu(input, grad_output, batch_size, in_channels, in_height, in_width, pool_height, pool_width, stride_height, stride_width, out_height, out_width, grad_input);
    } else {
        cudaMemset(grad_input, 0, batch_size * in_channels * in_height * in_width * sizeof(float));
        size_t num_elements = batch_size * in_channels * out_height * out_width;
        max_pool2d_bwd_kernel<<<CudaGetBlocks(num_elements), kCudaThreadsNum>>>(
            input, grad_output, batch_size, in_channels, in_height, in_width, pool_height, pool_width, stride_height, stride_width, out_height, out_width, num_elements, grad_input);
        cudaDeviceSynchronize();
    }
}

// Tensor interface implementations
Tensor max_pool2d(const Tensor& input,
                  size_t pool_height,
                  size_t pool_width,
                  size_t stride_height,
                  size_t stride_width) {
    if (!input.is_contiguous) {
        throw std::runtime_error("Input tensor must be contiguous");
    }
    if (input.shape().size() != 4) {
        throw std::runtime_error("Input tensor must be 4D (N, C, H, W)");
    }

    size_t batch_size = input.shape()[0];
    size_t in_channels = input.shape()[1];
    size_t in_height = input.shape()[2];
    size_t in_width = input.shape()[3];

    size_t out_height = (in_height - pool_height) / stride_height + 1;
    size_t out_width = (in_width - pool_width) / stride_width + 1;

    Tensor output({batch_size, in_channels, out_height, out_width}, input.device());

    max_pool2d_fwd(input.data(),
                   output.data(),
                   batch_size,
                   in_channels,
                   in_height,
                   in_width,
                   pool_height,
                   pool_width,
                   stride_height,
                   stride_width,
                   out_height,
                   out_width,
                   input.device());

    return output;
}

Tensor max_pool2d_backward(const Tensor& input,
                           const Tensor& output,
                           size_t pool_height,
                           size_t pool_width,
                           size_t stride_height,
                           size_t stride_width,
                           const Tensor& grad_output,
                           Tensor& grad_input) {
    if (!input.is_contiguous || !output.is_contiguous || !grad_output.is_contiguous) {
        throw std::runtime_error("All tensors must be contiguous");
    }
    if (input.shape().size() != 4 || grad_output.shape().size() != 4) {
        throw std::runtime_error("Tensors must be 4D");
    }
    if (input.device().type != output.device().type || input.device().type != grad_output.device().type) {
        throw std::runtime_error("All tensors must be on the same device");
    }

    size_t batch_size = input.shape()[0];
    size_t in_channels = input.shape()[1];
    size_t in_height = input.shape()[2];
    size_t in_width = input.shape()[3];

    size_t out_height = grad_output.shape()[2];
    size_t out_width = grad_output.shape()[3];

    max_pool2d_bwd(input.data(),
                   output.data(),
                   grad_output.data(),
                   batch_size,
                   in_channels,
                   in_height,
                   in_width,
                   pool_height,
                   pool_width,
                   stride_height,
                   stride_width,
                   out_height,
                   out_width,
                   input.device(),
                   grad_input.data());

    return grad_input;
}
