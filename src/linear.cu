#include "linear.cuh"
#include "tensor.cuh"
#include "cuda_utils.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

constexpr int kCudaThreadsNum = 512;

inline int CudaGetBlocks(const int N) {
    return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}

// Define the grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                                                                     \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void
add_bias_kernel(float* output, const float* bias, int out_features, int batch_size) {
    CUDA_KERNEL_LOOP(i, batch_size * out_features) {
        int feature = i % out_features;
        output[i] += bias[feature];
    }
}

// 创建一个全 1 向量的 kernel
__global__ void
fill_ones_kernel(float* data, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        data[i] = 1.0f;
    }
}

#define CUDA_CHECK(expr)                                                                           \
    do {                                                                                           \
        cudaError_t _err = (expr);                                                                 \
        if (_err != cudaSuccess) {                                                                 \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err));      \
        }                                                                                          \
    } while (0)

#define CUBLAS_CHECK(expr)                                                                         \
    do {                                                                                           \
        cublasStatus_t _status = (expr);                                                           \
        if (_status != CUBLAS_STATUS_SUCCESS) {                                                    \
            throw std::runtime_error("cuBLAS error: status code " + std::to_string(_status));      \
        }                                                                                          \
    } while (0)

class CublasHandle {
  public:
    CublasHandle() {
        CUBLAS_CHECK(cublasCreate(&handle_));
    }
    ~CublasHandle() {
        cublasDestroy(handle_);
    }
    cublasHandle_t get() const {
        return handle_;
    }

  private:
    cublasHandle_t handle_{};
};

// output(batch_size*out_features) = input(batch_size*in_features) *
// weights(in_features*out_features) + bias(out_features)
// 所有矩阵都是行主序存储
void fc_fwd(const float* input,
                float* output,
                const float* weights,
                const float* bias,
                int batch_size,
                int in_features,
                int out_features) {
    CublasHandle cublas_handle;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(cublas_handle.get(),
                             CUBLAS_OP_N,
                             CUBLAS_OP_N,
                             out_features,
                             batch_size,
                             in_features,
                             &alpha,
                             weights,
                             out_features,
                             input,
                             in_features,
                             &beta,
                             output,
                             out_features));

    if (bias != nullptr) {
        int total = batch_size * out_features;
        int blocks = CudaGetBlocks(total);
        add_bias_kernel<<<blocks, kCudaThreadsNum>>>(output, bias, out_features, batch_size);
        CUDA_CHECK(cudaGetLastError());
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

// grad_input(batch_size*in_features) =
//      grad_output(batch_size*out_features) * weights.T(out_features*in_features)
// grad_weights(in_features*out_features) =
//      input.T(in_features*batch_size) * grad_output(batch_size*out_features)
// grad_bias(out_features) = sum over batch of grad_output(batch_size*out_features)
void fc_bwd(const float *input, const float *output, const float *weights, const float *bias,
            int batch_size, int in_features, int out_features,
            const float *grad_output, float *grad_input, float *grad_weights,
            float *grad_bias)
            {
    CublasHandle cublas_handle;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // 1. 计算 grad_input = grad_output * weights.T
    // grad_output: batch_size × out_features (row-major)
    // weights: in_features × out_features (row-major)
    // grad_input: batch_size × in_features (row-major)
    // 在 cuBLAS (列主序) 中: grad_input^T = weights^T * grad_output^T
    if (grad_input != nullptr) {
        CUBLAS_CHECK(cublasSgemm(cublas_handle.get(),
                                 CUBLAS_OP_T,
                                 CUBLAS_OP_N,
                                 in_features,
                                 batch_size,
                                 out_features,
                                 &alpha,
                                 weights,
                                 out_features,
                                 grad_output,
                                 out_features,
                                 &beta,
                                 grad_input,
                                 in_features));
    }

    // 2. 计算 grad_weights = input.T * grad_output
    // input: batch_size × in_features (row-major)
    // grad_output: batch_size × out_features (row-major)
    // grad_weights: in_features × out_features (row-major)
    // 在 cuBLAS (列主序) 中: grad_weights^T = grad_output^T * input
    if (grad_weights != nullptr) {
        CUBLAS_CHECK(cublasSgemm(cublas_handle.get(),
                                 CUBLAS_OP_N,
                                 CUBLAS_OP_T,
                                 out_features,
                                 in_features,
                                 batch_size,
                                 &alpha,
                                 grad_output,
                                 out_features,
                                 input,
                                 in_features,
                                 &beta,
                                 grad_weights,
                                 out_features));
    }

    // 3. 计算 grad_bias = sum over batch of grad_output
    // grad_output: batch_size × out_features (row-major)
    // grad_bias: out_features
    // 使用 cuBLAS gemv 计算: grad_bias[j] = sum_i(grad_output[i][j])
    if (grad_bias != nullptr) {
        // 分配并初始化全 1 向量 (长度为 batch_size)
        auto ones_device = make_cuda_unique(batch_size);
        
        int blocks = CudaGetBlocks(batch_size);
        fill_ones_kernel<<<blocks, kCudaThreadsNum>>>(ones_device.get(), batch_size);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 行主序的 batch_size × out_features 矩阵，在 cuBLAS 列主序视角下
        // 等价于 out_features × batch_size 的转置矩阵
        // 我们要对每列（行主序视角）求和，等价于列主序视角下对每行求和
        // y = A * x，其中 A 是 out_features × batch_size (列主序)，x 是全 1 向量 (batch_size)
        // 结果 y 是 out_features
        const float alpha_ones = 1.0f;
        const float beta_ones = 0.0f;
        CUBLAS_CHECK(cublasSgemv(cublas_handle.get(),
                                 CUBLAS_OP_N,                // 不转置
                                 out_features,               // A 的行数（列主序视角）
                                 batch_size,                 // A 的列数（列主序视角）
                                 &alpha_ones,
                                 grad_output,                // A
                                 out_features,               // lda (列主序下的 leading dimension)
                                 ones_device.get(),          // x (长度 batch_size)
                                 1,                          // incx
                                 &beta_ones,
                                 grad_bias,                  // y (长度 out_features)
                                 1));                        // incy
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

Tensor linear(const Tensor &input, const Tensor &weight, const Tensor &bias)
{
    //检查输入都在gpu上
    if (!input.device().is_cuda() || !weight.device().is_cuda() ||
        (!bias.device().is_cuda() && bias.numel() != 0)) {
        throw std::invalid_argument("All input tensors must be on CUDA device");
    }

    // 检查输入维度
    if (input.ndim() != 2 || weight.ndim() != 2) {
        throw std::invalid_argument("Input and weight tensors must be 2-dimensional");
    }
    if (input.shape()[1] != weight.shape()[0]) {
        throw std::invalid_argument("Input feature size must match weight input dimension");
    }
    if (bias.numel() != 0 && bias.numel() != weight.shape()[1]) {
        throw std::invalid_argument("Bias size must match weight output dimension");
    }

    int batch_size = static_cast<int>(input.shape()[0]);
    int in_features = static_cast<int>(input.shape()[1]);
    int out_features = static_cast<int>(weight.shape()[1]);

    Tensor output({static_cast<Tensor::index_t>(batch_size), static_cast<Tensor::index_t>(out_features)}, input.device());

    fc_fwd(input.data(), output.data(), weight.data(),
           bias.numel() == 0 ? nullptr : bias.data(),
           batch_size, in_features, out_features);

    return output;
}

std::tuple<Tensor, Tensor, Tensor> linear_backward(const Tensor &input,
                                                   const Tensor &weight,
                                                   const Tensor &bias,
                                                   const Tensor &grad_output)
{

    //检查输入都在gpu上
    if (!input.device().is_cuda() || !weight.device().is_cuda() ||
        (!bias.device().is_cuda() && bias.numel() != 0) ||
        !grad_output.device().is_cuda()) {
        throw std::invalid_argument("All input tensors must be on CUDA device");
    }
    // 检查输入维度
    if (input.ndim() != 2 || weight.ndim() != 2 || grad_output.ndim() != 2) {
        throw std::invalid_argument("Input, weight, and grad_output tensors must be 2-dimensional");
    }
    if (input.shape()[1] != weight.shape()[0]) {
        throw std::invalid_argument("Input feature size must match weight input dimension");
    }
    if (grad_output.shape()[1] != weight.shape()[1]) {
        throw std::invalid_argument("Grad output feature size must match weight output dimension");
    }

    int batch_size = static_cast<int>(input.shape()[0]);
    int in_features = static_cast<int>(input.shape()[1]);
    int out_features = static_cast<int>(weight.shape()[1]);

    Tensor grad_input({static_cast<Tensor::index_t>(batch_size), static_cast<Tensor::index_t>(in_features)}, input.device());
    Tensor grad_weights({static_cast<Tensor::index_t>(in_features), static_cast<Tensor::index_t>(out_features)}, weight.device());
    Tensor grad_bias({static_cast<Tensor::index_t>(out_features)}, bias.device());

    fc_bwd(input.data(), grad_output.data(), weight.data(), bias.data(),
           batch_size, in_features, out_features,
           grad_output.data(), grad_input.data(), grad_weights.data(), grad_bias.data());

    return {grad_input, grad_weights, grad_bias};
}
