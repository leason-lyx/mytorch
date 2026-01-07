#include "conv.cuh"
#include "tensor.cuh"
#include "cuda_utils.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdexcept>
#include <string>
#include <vector>
#include <cstring>
#include <memory>

constexpr int kCudaThreadsNum = 512;

inline int CudaGetBlocks(const int N) {
    return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}

__device__ __host__ inline int min_int(int a, int b) {
    return (a < b) ? a : b;
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
// ============================================================================
// im2col - 将输入图像转换为列矩阵（CUDA kernel）
// ============================================================================

// im2col kernel: 将一个图像通道展开为列
// input: (in_channels, input_height, input_width)
// output: (in_channels * kernel_height * kernel_width, output_height * output_width)
__global__ void im2col_kernel(
    const float* data_im,
    int in_channels,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    int output_height,
    int output_width,
    float* data_col) {
    
    CUDA_KERNEL_LOOP(index, in_channels * output_height * output_width * kernel_height * kernel_width) {
        // 计算输出位置
        int w_out = index % output_width;
        int idx = index / output_width;
        int h_out = idx % output_height;
        idx /= output_height;
        int kw = idx % kernel_width;
        idx /= kernel_width;
        int kh = idx % kernel_height;
        int c_in = idx / kernel_height;
        
        // 计算输入位置
        int h_in = h_out * stride_height - padding_height + kh;
        int w_in = w_out * stride_width - padding_width + kw;
        
        // 计算 col 的位置
        int col_index = (c_in * kernel_height * kernel_width + kh * kernel_width + kw) * 
                       (output_height * output_width) + h_out * output_width + w_out;
        
        // 如果在边界内，复制数据；否则填充 0
        if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
            int im_index = (c_in * input_height + h_in) * input_width + w_in;
            data_col[col_index] = data_im[im_index];
        } else {
            data_col[col_index] = 0.0f;
        }
    }
}

// CPU 版本的 im2col
void im2col_cpu(
    const float* data_im,
    int in_channels,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    int output_height,
    int output_width,
    float* data_col) {
    
    for (int c = 0; c < in_channels; ++c) {
        for (int kh = 0; kh < kernel_height; ++kh) {
            for (int kw = 0; kw < kernel_width; ++kw) {
                int input_row_start = -padding_height + kh;
                for (int h_out = 0; h_out < output_height; ++h_out) {
                    int h_in = input_row_start + h_out * stride_height;
                    int input_col_start = -padding_width + kw;
                    for (int w_out = 0; w_out < output_width; ++w_out) {
                        int w_in = input_col_start + w_out * stride_width;
                        
                        int col_index = (c * kernel_height * kernel_width + kh * kernel_width + kw) * 
                                       (output_height * output_width) + h_out * output_width + w_out;
                        
                        if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                            int im_index = (c * input_height + h_in) * input_width + w_in;
                            data_col[col_index] = data_im[im_index];
                        } else {
                            data_col[col_index] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// col2im - 将列矩阵转换回图像格式（CUDA kernel）
// ============================================================================

__global__ void col2im_kernel(
    const float* data_col,
    int in_channels,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    int output_height,
    int output_width,
    float* data_im) {
    
    CUDA_KERNEL_LOOP(index, in_channels * input_height * input_width) {
        float val = 0;
        int w_im = index % input_width;
        int idx = index / input_width;
        int h_im = idx % input_height;
        int c_im = idx / input_height;
        
        // 计算哪些卷积核位置会影响这个输入位置
        int kh_start = (h_im + padding_height < kernel_height) ? 0 : 
                       (h_im + padding_height - kernel_height) / stride_height + 1;
        int kh_end = min_int((h_im + padding_height) / stride_height + 1, output_height);
        int kw_start = (w_im + padding_width < kernel_width) ? 0 : 
                       (w_im + padding_width - kernel_width) / stride_width + 1;
        int kw_end = min_int((w_im + padding_width) / stride_width + 1, output_width);
        
        for (int h_out = kh_start; h_out < kh_end; ++h_out) {
            for (int w_out = kw_start; w_out < kw_end; ++w_out) {
                int kh = (h_im + padding_height) - h_out * stride_height;
                int kw = (w_im + padding_width) - w_out * stride_width;
                
                if (kh >= 0 && kh < kernel_height && kw >= 0 && kw < kernel_width) {
                    int col_index = (c_im * kernel_height * kernel_width + kh * kernel_width + kw) * 
                                   (output_height * output_width) + h_out * output_width + w_out;
                    val += data_col[col_index];
                }
            }
        }
        
        data_im[index] = val;
    }
}

// CPU 版本的 col2im
void col2im_cpu(
    const float* data_col,
    int in_channels,
    int input_height,
    int input_width,
    int kernel_height,
    int kernel_width,
    int stride_height,
    int stride_width,
    int padding_height,
    int padding_width,
    int output_height,
    int output_width,
    float* data_im) {
    
    // 初始化为 0
    std::memset(data_im, 0, in_channels * input_height * input_width * sizeof(float));
    
    for (int c = 0; c < in_channels; ++c) {
        for (int kh = 0; kh < kernel_height; ++kh) {
            for (int kw = 0; kw < kernel_width; ++kw) {
                int input_row_start = -padding_height + kh;
                for (int h_out = 0; h_out < output_height; ++h_out) {
                    int h_in = input_row_start + h_out * stride_height;
                    int input_col_start = -padding_width + kw;
                    for (int w_out = 0; w_out < output_width; ++w_out) {
                        int w_in = input_col_start + w_out * stride_width;
                        
                        int col_index = (c * kernel_height * kernel_width + kh * kernel_width + kw) * 
                                       (output_height * output_width) + h_out * output_width + w_out;
                        
                        if (h_in >= 0 && h_in < input_height && w_in >= 0 && w_in < input_width) {
                            int im_index = (c * input_height + h_in) * input_width + w_in;
                            data_im[im_index] += data_col[col_index];
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// 使用 GEMM 添加 bias 的 kernel
// ============================================================================

__global__ void add_bias_kernel(float* output, const float* bias, int out_channels,
                               int output_height, int output_width, int batch_size) {
    CUDA_KERNEL_LOOP(i, batch_size * out_channels * output_height * output_width) {
        int hw = output_height * output_width;
        int c = (i / hw) % out_channels;
        
        output[i] += bias[c];
    }
}

// fill_ones_kernel 在 linear.cu 中已经定义，这里声明一下
__global__ void fill_ones_kernel(float* data, int n);

void add_bias_cpu(float* output, const float* bias, int out_channels,
                  int output_height, int output_width, int batch_size) {
    for (int n = 0; n < batch_size; ++n) {
        for (int c = 0; c < out_channels; ++c) {
            for (int h = 0; h < output_height; ++h) {
                for (int w = 0; w < output_width; ++w) {
                    int idx = ((n * out_channels + c) * output_height + h) * output_width + w;
                    output[idx] += bias[c];
                }
            }
        }
    }
}

// ============================================================================
// Conv2d Forward Pass
// ============================================================================

void conv2d_fwd(const float* input,
                float* output,
                const float* weights,
                const float* bias,
                int batch_size,
                int in_channels,
                int out_channels,
                int input_height,
                int input_width,
                int kernel_height,
                int kernel_width,
                int stride_height,
                int stride_width,
                int padding_height,
                int padding_width) {
    
    // 计算输出尺寸
    int output_height = (input_height + 2 * padding_height - kernel_height) / stride_height + 1;
    int output_width = (input_width + 2 * padding_width - kernel_width) / stride_width + 1;
    
    // col_buffer 大小: (in_channels * kernel_height * kernel_width) × (output_height * output_width)
    int col_buffer_size = in_channels * kernel_height * kernel_width * output_height * output_width;
    auto col_buffer = make_cuda_unique(col_buffer_size);
    
    CublasHandle cublas_handle;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // 对每个 batch 进行处理
    for (int n = 0; n < batch_size; ++n) {
        const float* input_n = input + n * in_channels * input_height * input_width;
        float* output_n = output + n * out_channels * output_height * output_width;
        
        // Step 1: im2col - 将输入图像转换为列矩阵
        int im2col_size = in_channels * output_height * output_width * kernel_height * kernel_width;
        int blocks = CudaGetBlocks(im2col_size);
        im2col_kernel<<<blocks, kCudaThreadsNum>>>(
            input_n, in_channels, input_height, input_width,
            kernel_height, kernel_width, stride_height, stride_width,
            padding_height, padding_width, output_height, output_width,
            col_buffer.get());
        CUDA_CHECK(cudaGetLastError());
        
        // Step 2: GEMM - 计算卷积
        // weights: out_channels × (in_channels * kernel_height * kernel_width) [行主序]
        // col_buffer: (in_channels * kernel_height * kernel_width) × (output_height * output_width) [行主序]
        // output_n: out_channels × (output_height * output_width) [行主序]
        // 
        // 在 cuBLAS（列主序）中: output_n^T = col_buffer^T * weights^T
        int M = output_height * output_width;
        int N = out_channels;
        int K = in_channels * kernel_height * kernel_width;
        
        CUBLAS_CHECK(cublasSgemm(
            cublas_handle.get(),
            CUBLAS_OP_N,           // col_buffer (不转置，在列主序中看是 M×K)
            CUBLAS_OP_N,           // weights (不转置，在列主序中看是 K×N)
            M,                     // 行数（列主序视角）
            N,                     // 列数（列主序视角）
            K,                     // 内积维度
            &alpha,
            col_buffer.get(),      // col_buffer 在行主序中是 K×M，在列主序中看是 M×K
            M,                     // leading dimension (列主序中的行数)
            weights,               // weights 在行主序中是 N×K，在列主序中看是 K×N
            K,                     // leading dimension (列主序中的行数)
            &beta,
            output_n,              // output_n 在行主序中是 N×M，在列主序中看是 M×N
            M                      // leading dimension (列主序中的行数)
        ));
    }
    
    // Step 3: 使用 GEMM 风格添加 bias（实际上是逐通道加）
    if (bias != nullptr) {
        int total = batch_size * out_channels * output_height * output_width;
        int blocks = CudaGetBlocks(total);
        add_bias_kernel<<<blocks, kCudaThreadsNum>>>(
            output, bias, out_channels, output_height, output_width, batch_size);
        CUDA_CHECK(cudaGetLastError());
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Conv2d Backward Pass
// ============================================================================

void conv2d_bwd(const float* input,
                const float* output,
                const float* weights,
                const float* bias,
                int batch_size,
                int in_channels,
                int out_channels,
                int input_height,
                int input_width,
                int kernel_height,
                int kernel_width,
                int stride_height,
                int stride_width,
                int padding_height,
                int padding_width,
                const float* grad_output,
                float* grad_input,
                float* grad_weights,
                float* grad_bias) {
    
    int output_height = (input_height + 2 * padding_height - kernel_height) / stride_height + 1;
    int output_width = (input_width + 2 * padding_width - kernel_width) / stride_width + 1;
    
    int col_buffer_size = in_channels * kernel_height * kernel_width * output_height * output_width;
    auto col_buffer = make_cuda_unique(col_buffer_size);
    
    // 预分配 ones 向量用于 grad_bias 计算
    int HW = output_height * output_width;
    auto ones = make_cuda_unique(HW);
    int blocks = CudaGetBlocks(HW);
    fill_ones_kernel<<<blocks, kCudaThreadsNum>>>(ones.get(), HW);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CublasHandle cublas_handle;
    const float alpha = 1.0f;
    const float beta_zero = 0.0f;
    const float beta_one = 1.0f;
    
    // 初始化梯度为 0
    if (grad_weights != nullptr) {
        CUDA_CHECK(cudaMemset(grad_weights, 0, 
            out_channels * in_channels * kernel_height * kernel_width * sizeof(float)));
    }
    if (grad_bias != nullptr) {
        CUDA_CHECK(cudaMemset(grad_bias, 0, out_channels * sizeof(float)));
    }
    if (grad_input != nullptr) {
        CUDA_CHECK(cudaMemset(grad_input, 0, 
            batch_size * in_channels * input_height * input_width * sizeof(float)));
    }
    
    // 对每个 batch 进行处理
    for (int n = 0; n < batch_size; ++n) {
        const float* input_n = input + n * in_channels * input_height * input_width;
        const float* grad_output_n = grad_output + n * out_channels * output_height * output_width;
        float* grad_input_n = nullptr;
        if (grad_input != nullptr) {
            grad_input_n = grad_input + n * in_channels * input_height * input_width;
        }
        
        // ========================================================================
        // Step 1: 计算 grad_weights - 使用 im2col + GEMM
        // ========================================================================
        if (grad_weights != nullptr) {
            // im2col: input_n -> col_buffer
            int im2col_size = in_channels * output_height * output_width * kernel_height * kernel_width;
            int blocks = CudaGetBlocks(im2col_size);
            im2col_kernel<<<blocks, kCudaThreadsNum>>>(
                input_n, in_channels, input_height, input_width,
                kernel_height, kernel_width, stride_height, stride_width,
                padding_height, padding_width, output_height, output_width,
                col_buffer.get());
            CUDA_CHECK(cudaGetLastError());
            
            // GEMM: grad_weights += grad_output_n * col_buffer^T
            // grad_output_n: out_channels × (output_height * output_width) [行主序]
            // col_buffer: (in_channels * kernel_height * kernel_width) × (output_height * output_width) [行主序]
            // grad_weights: out_channels × (in_channels * kernel_height * kernel_width) [行主序]
            //
            // 在列主序视角：grad_weights^T = col_buffer × grad_output_n^T
            // col_buffer 在行主序中是 M×K，在列主序中看是 K×M
            // grad_output_n 在行主序中是 N×K，在列主序中看是 K×N
            // grad_weights 在行主序中是 N×M，在列主序中看是 M×N
            int M = in_channels * kernel_height * kernel_width;
            int N = out_channels;
            int K = output_height * output_width;
            
            float beta = (n == 0) ? beta_zero : beta_one;
            
            CUBLAS_CHECK(cublasSgemm(
                cublas_handle.get(),
                CUBLAS_OP_T,           // col_buffer^T: 将 M×K (行主序) 转置成 K×M (列主序)
                CUBLAS_OP_N,           // grad_output_n: K×N (列主序)
                M,
                N,
                K,
                &alpha,
                col_buffer.get(),      // M×K 在行主序中
                K,                     // leading dimension (行主序中的列数)
                grad_output_n,         // N×K 在行主序中，等价于 K×N 在列主序中
                K,                     // leading dimension
                &beta,
                grad_weights,          // N×M 在行主序中，等价于 M×N 在列主序中
                M                      // leading dimension
            ));
        }
        
        // ========================================================================
        // Step 2: 计算 grad_input - 使用 GEMM + col2im
        // ========================================================================
        if (grad_input_n != nullptr) {
            // GEMM: col_buffer = weights^T * grad_output_n
            // weights: out_channels × (in_channels * kernel_height * kernel_width) [行主序]
            // grad_output_n: out_channels × (output_height * output_width) [行主序]
            // col_buffer: (in_channels * kernel_height * kernel_width) × (output_height * output_width) [行主序]
            //
            // 在列主序视角计算：col_buffer^T = grad_output_n^T × weights^T
            // 但利用转置技巧，等价于在行主序中直接计算
            int M = output_height * output_width;
            int N = in_channels * kernel_height * kernel_width;
            int K = out_channels;
            
            CUBLAS_CHECK(cublasSgemm(
                cublas_handle.get(),
                CUBLAS_OP_N,           // grad_output_n（当作列主序的 M×K）
                CUBLAS_OP_T,           // weights^T（转置，变成列主序的 K×N）
                M,
                N,
                K,
                &alpha,
                grad_output_n,         // 行主序 K×M，列主序视角 M×K
                M,                     // leading dimension
                weights,               // 行主序 K×N，需要转置
                N,                     // leading dimension
                &beta_zero,
                col_buffer.get(),      // 行主序 N×M，列主序视角 M×N
                M                      // leading dimension
            ));
            
            // col2im: col_buffer -> grad_input_n
            int col2im_size = in_channels * input_height * input_width;
            int blocks = CudaGetBlocks(col2im_size);
            col2im_kernel<<<blocks, kCudaThreadsNum>>>(
                col_buffer.get(), in_channels, input_height, input_width,
                kernel_height, kernel_width, stride_height, stride_width,
                padding_height, padding_width, output_height, output_width,
                grad_input_n);
            CUDA_CHECK(cudaGetLastError());
        }
        
        // ========================================================================
        // Step 3: 计算 grad_bias - 对每个通道求和
        // ========================================================================
        if (grad_bias != nullptr) {
            // grad_bias[c] = sum over (h, w) of grad_output_n[c, h, w]
            // 使用 cuBLAS 的矩阵向量乘法
            // grad_bias[out_channels] = grad_output_n[out_channels × (H*W)] * ones[(H*W)]
            
            int HW = output_height * output_width;
            
            // 使用 cuBLAS gemv: grad_bias += grad_output_n * ones
            // grad_output_n: out_channels × HW [行主序]
            // ones: HW × 1
            // grad_bias: out_channels × 1
            float beta = (n == 0) ? beta_zero : beta_one;
            
            CUBLAS_CHECK(cublasSgemv(
                cublas_handle.get(),
                CUBLAS_OP_N,           // 不转置，直接对行求和
                out_channels,          // 行数
                HW,                    // 列数
                &alpha,
                grad_output_n,         // 矩阵
                out_channels,           // leading dimension
                ones.get(),            // 向量 - 使用预分配的 ones 向量
                1,                     // 增量
                &beta,
                grad_bias,             // 结果
                1                      // 增量
            ));
        }
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Tensor-level wrappers
// ============================================================================

Tensor conv2d(const Tensor& input,
              const Tensor& weights,
              const Tensor& bias,
              int stride_height,
              int stride_width,
              int padding_height,
              int padding_width) {
    
    // 检查设备类型
    if (!input.device().is_cuda() || !weights.device().is_cuda() || 
        (!bias.device().is_cuda() && bias.numel() != 0)) {
        throw std::runtime_error("All input tensors must be on CUDA device");
    }
    
    // 检查输入形状
    if (input.ndim() != 4) {
        throw std::runtime_error("Input must be 4D: (batch_size, in_channels, height, width)");
    }
    if (weights.ndim() != 4) {
        throw std::runtime_error("Weights must be 4D: (out_channels, in_channels, kernel_height, kernel_width)");
    }
    if (bias.ndim() != 1) {
        throw std::runtime_error("Bias must be 1D: (out_channels)");
    }
    
    // 检查参数合理性
    if (stride_height <= 0 || stride_width <= 0) {
        throw std::runtime_error("Stride must be positive");
    }
    if (padding_height < 0 || padding_width < 0) {
        throw std::runtime_error("Padding must be non-negative");
    }
    
    int batch_size = input.shape()[0];
    int in_channels = input.shape()[1];
    int input_height = input.shape()[2];
    int input_width = input.shape()[3];
    
    int out_channels = weights.shape()[0];
    int kernel_height = weights.shape()[2];
    int kernel_width = weights.shape()[3];
    
    int output_height = (input_height + 2 * padding_height - kernel_height) / stride_height + 1;
    int output_width = (input_width + 2 * padding_width - kernel_width) / stride_width + 1;
    
    // 创建输出张量
    Tensor output({static_cast<size_t>(batch_size),
                   static_cast<size_t>(out_channels),
                   static_cast<size_t>(output_height),
                   static_cast<size_t>(output_width)},
                  Device::cuda());
    
    // 调用 pointer interface
    conv2d_fwd(input.data(), output.data(), weights.data(), bias.data(),
               batch_size, in_channels, out_channels,
               input_height, input_width,
               kernel_height, kernel_width,
               stride_height, stride_width,
               padding_height, padding_width);
    
    return output;
}

void conv2d_backward(const Tensor& input,
                     const Tensor& output,
                     const Tensor& weights,
                     const Tensor& bias,
                     int stride_height,
                     int stride_width,
                     int padding_height,
                     int padding_width,
                     const Tensor& grad_output,
                     Tensor& grad_input,
                     Tensor& grad_weights,
                     Tensor& grad_bias) {
    
    // 检查设备类型
    if (!input.device().is_cuda() || !weights.device().is_cuda() || 
        (!bias.device().is_cuda() && bias.numel() != 0) || !grad_output.device().is_cuda() ||
        !grad_input.device().is_cuda() || !grad_weights.device().is_cuda() || 
        !grad_bias.device().is_cuda()) {
        throw std::runtime_error("All tensors must be on CUDA device");
    }
    
    // 检查输入形状
    if (input.ndim() != 4 || weights.ndim() != 4 || grad_output.ndim() != 4 ||
        grad_input.ndim() != 4 || grad_weights.ndim() != 4 || grad_bias.ndim() != 1) {
        throw std::runtime_error("Invalid tensor dimensions for backward pass");
    }
    
    int batch_size = input.shape()[0];
    int in_channels = input.shape()[1];
    int input_height = input.shape()[2];
    int input_width = input.shape()[3];
    
    int out_channels = weights.shape()[0];
    int kernel_height = weights.shape()[2];
    int kernel_width = weights.shape()[3];
    
    conv2d_bwd(input.data(), output.data(), weights.data(), bias.data(),
               batch_size, in_channels, out_channels,
               input_height, input_width,
               kernel_height, kernel_width,
               stride_height, stride_width,
               padding_height, padding_width,
               grad_output.data(),
               grad_input.data(),
               grad_weights.data(),
               grad_bias.data());
}
