#include "functional.cuh"
#include "cuda_utils.cuh"
#include <cmath>
#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/for_each.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>

// Use 512 threads per block
const int kCudaThreadsNum = 512;
inline int CudaGetBlocks(const int N) {
    return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}
// Define the grid stride looping
#define CUDA_KERNEL_LOOP(i, n)                                                                     \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

#define CUDA_CHECK(expr)                                                                           \
    do {                                                                                           \
        cudaError_t _err = (expr);                                                                 \
        if (_err != cudaSuccess) {                                                                 \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err));      \
        }                                                                                          \
    } while (0)

// GPU版本
__global__ void relu_fwd_kernel(const float* x, float* y, size_t n) {
    CUDA_KERNEL_LOOP(i, n) {
        y[i] = x[i] > 0.0f ? x[i] : 0.0f;
    }
}

__global__ void relu_bwd_kernel(const float* x, const float* grad_out, float* grad_in, size_t n) {
    CUDA_KERNEL_LOOP(i, n) {
        grad_in[i] = (x[i] > 0.0f) ? grad_out[i] : 0.0f;
    }
}

__device__ __forceinline__ float sigmoid_scalar(float z) {
    // 数值稳定分段：避免极值下 exp 溢出
    if (z >= 0.0f) {
        float ez = expf(-z);
        return 1.0f / (1.0f + ez);
    } else {
        float ez = expf(z);
        return ez / (1.0f + ez);
    }
}

__global__ void sigmoid_fwd_kernel(const float* x, float* y, size_t n) {
    CUDA_KERNEL_LOOP(i, n) {
        y[i] = sigmoid_scalar(x[i]);
    }
}

__global__ void
sigmoid_bwd_from_x_kernel(const float* x, const float* grad_out, float* grad_in, size_t n) {
    CUDA_KERNEL_LOOP(i, n) {
        float s = sigmoid_scalar(x[i]);
        grad_in[i] = grad_out[i] * s * (1.0f - s);
    }
}

__global__ void
sigmoid_bwd_from_y_kernel(const float* y, const float* grad_out, float* grad_in, size_t n) {
    CUDA_KERNEL_LOOP(i, n) {
        float s = y[i];
        grad_in[i] = grad_out[i] * s * (1.0f - s);
    }
}


// CPU版本
static void relu_fwd_cpu(const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float v = x[i];
        y[i] = v > 0.0f ? v : 0.0f;
    }
}

static void relu_bwd_cpu(const float* x, const float* grad_out, float* grad_in, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        grad_in[i] = (x[i] > 0.0f) ? grad_out[i] : 0.0f;
    }
}

static inline float sigmoid_scalar_cpu(float z) {
    if (z >= 0.0f) {
        float ez = std::exp(-z);
        return 1.0f / (1.0f + ez);
    } else {
        float ez = std::exp(z);
        return ez / (1.0f + ez);
    }
}

static void sigmoid_fwd_cpu(const float* x, float* y, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        y[i] = sigmoid_scalar_cpu(x[i]);
    }
}

static void
sigmoid_bwd_from_x_cpu(const float* x, const float* grad_out, float* grad_in, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float s = sigmoid_scalar_cpu(x[i]);
        grad_in[i] = grad_out[i] * s * (1.0f - s);
    }
}

static void
sigmoid_bwd_from_y_cpu(const float* y, const float* grad_out, float* grad_in, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float s = y[i];
        grad_in[i] = grad_out[i] * s * (1.0f - s);
    }
}

// Softmax CPU实现
static void softmax_fwd_cpu(const float* x, float* y, size_t n, size_t c) {
    for (size_t row = 0; row < n; ++row) {
        const float* row_x = x + row * c;
        float* row_y = y + row * c;
        
        // 1. Compute max elements over input
        float max_val = row_x[0];
        for (size_t i = 1; i < c; ++i) {
            max_val = std::max(max_val, row_x[i]);
        }
        
        // 2. Subtract the max value for each row
        // 3. Compute the exponent for each element
        float sum = 0.0f;
        for (size_t i = 0; i < c; ++i) {
            float exp_val = std::exp(row_x[i] - max_val);
            row_y[i] = exp_val;
            sum += exp_val;
        }
        
        // 4. Sum over each row (already done above)
        // 5. Normalize the results
        for (size_t i = 0; i < c; ++i) {
            row_y[i] /= sum;
        }
    }
}

// 指针接口
void relu_fwd(const float* x, float* y, std::size_t n, const Device& dev) {
    if (dev.is_cpu()) {
        relu_fwd_cpu(x, y, n);
    } else {
        int blocks = CudaGetBlocks(static_cast<int>(n));
        relu_fwd_kernel<<<blocks, kCudaThreadsNum>>>(x, y, n);
        cudaDeviceSynchronize();
    }
}

void relu_bwd(
    const float* x, const float* grad_out, float* grad_in, std::size_t n, const Device& dev) {
    if (dev.is_cpu()) {
        relu_bwd_cpu(x, grad_out, grad_in, n);
    } else {
        int blocks = CudaGetBlocks(static_cast<int>(n));
        relu_bwd_kernel<<<blocks, kCudaThreadsNum>>>(x, grad_out, grad_in, n);
        cudaDeviceSynchronize();
    }
}

void sigmoid_fwd(const float* x, float* y, std::size_t n, const Device& dev) {
    if (dev.is_cpu()) {
        sigmoid_fwd_cpu(x, y, n);
    } else {
        int blocks = CudaGetBlocks(static_cast<int>(n));
        sigmoid_fwd_kernel<<<blocks, kCudaThreadsNum>>>(x, y, n);
        cudaDeviceSynchronize();
    }
}

void sigmoid_bwd(
    const float* x, const float* grad_out, float* grad_in, std::size_t n, const Device& dev) {
    if (dev.is_cpu()) {
        sigmoid_bwd_from_x_cpu(x, grad_out, grad_in, n);
    } else {
        int blocks = CudaGetBlocks(static_cast<int>(n));
        sigmoid_bwd_from_x_kernel<<<blocks, kCudaThreadsNum>>>(x, grad_out, grad_in, n);
        cudaDeviceSynchronize();
    }
}

void sigmoid_bwd_from_output(
    const float* y, const float* grad_out, float* grad_in, std::size_t n, const Device& dev) {
    if (dev.is_cpu()) {
        sigmoid_bwd_from_y_cpu(y, grad_out, grad_in, n);
    } else {
        int blocks = CudaGetBlocks(static_cast<int>(n));
        sigmoid_bwd_from_y_kernel<<<blocks, kCudaThreadsNum>>>(y, grad_out, grad_in, n);
        cudaDeviceSynchronize();
    }
}

// Softmax transform functors
struct exp_transform_functor {
    const float max_val;
    exp_transform_functor(float max) : max_val(max) {}
    __host__ __device__ float operator()(float val) const {
        return expf(val - max_val);
    }
};

struct normalize_transform_functor {
    const float sum_val;
    normalize_transform_functor(float sum) : sum_val(sum) {}
    __host__ __device__ float operator()(float val) const {
        return val / sum_val;
    }
};

// Functors for optimized softmax
struct row_index_functor {
    int c;
    row_index_functor(int cc) : c(cc) {}
    __device__ int operator()(int idx) const {
        return idx / c;
    }
};

struct softmax_exp_functor {
    const float* max_vals;
    const int c;
    softmax_exp_functor(const float* max, int cc) : max_vals(max), c(cc) {}
    __device__ float operator()(const thrust::tuple<float, int>& t) const {
        float val = thrust::get<0>(t);
        int idx = thrust::get<1>(t);
        int row = idx / c;
        return expf(val - max_vals[row]);
    }
};

struct normalize_functor {
    const float* sum_vals;
    const int c;
    normalize_functor(const float* sum, int cc) : sum_vals(sum), c(cc) {}
    __device__ float operator()(const thrust::tuple<float, int>& t) const {
        float val = thrust::get<0>(t);
        int idx = thrust::get<1>(t);
        int row = idx / c;
        return val / sum_vals[row];
    }
};

void softmax_fwd(const float* x, float* y, std::size_t n, std::size_t c, const Device& dev) {
    if (dev.is_cpu()) {
        softmax_fwd_cpu(x, y, n, c);
    } else {
        // GPU实现：使用thrust优化，消除显式for循环
        
        // 将输入数据包装为thrust device vector
        thrust::device_vector<float> x_dv(x, x + n * c);
        thrust::device_vector<float> y_dv(n * c);
        
        // 创建行索引keys，用于segmented operations
        thrust::device_vector<int> keys(n * c);
        thrust::transform(thrust::make_counting_iterator(0), 
                         thrust::make_counting_iterator((int)(n * c)), 
                         keys.begin(), 
                         row_index_functor((int)c));
        
        // 1. Compute max elements over input for each row using reduce_by_key
        thrust::device_vector<int> keys_out_max(n);
        thrust::device_vector<float> max_vals(n);
        thrust::reduce_by_key(keys.begin(), keys.end(), x_dv.begin(), 
                             keys_out_max.begin(), max_vals.begin(), 
                             thrust::equal_to<int>(), thrust::maximum<float>());
        
        // 2. Subtract the max value for each row and compute exponent
        thrust::device_vector<float> exp_vals(n * c);
        thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(x_dv.begin(), thrust::make_counting_iterator(0))),
                         thrust::make_zip_iterator(thrust::make_tuple(x_dv.end(), thrust::make_counting_iterator((int)(n * c)))),
                         exp_vals.begin(), 
                         softmax_exp_functor(thrust::raw_pointer_cast(max_vals.data()), (int)c));
        
        // 3. Sum over each row using reduce_by_key
        thrust::device_vector<int> keys_out_sum(n);
        thrust::device_vector<float> sum_vals(n);
        thrust::reduce_by_key(keys.begin(), keys.end(), exp_vals.begin(), 
                             keys_out_sum.begin(), sum_vals.begin(), 
                             thrust::equal_to<int>(), thrust::plus<float>());
        
        // 4. Normalize the results
        thrust::transform(thrust::make_zip_iterator(thrust::make_tuple(exp_vals.begin(), thrust::make_counting_iterator(0))),
                         thrust::make_zip_iterator(thrust::make_tuple(exp_vals.end(), thrust::make_counting_iterator((int)(n * c)))),
                         y_dv.begin(), 
                         normalize_functor(thrust::raw_pointer_cast(sum_vals.data()), (int)c));
        
        // 将结果复制回输出
        thrust::copy(y_dv.begin(), y_dv.end(), y);
    }
}

// Tensor接口
Tensor relu(const Tensor& x) {
    Tensor y(x.shape(), x.device());
    relu_fwd(x.data(), y.data(), x.numel(), x.device());
    return y;
}

Tensor relu_backward(const Tensor& x, const Tensor& grad_out) {
    Tensor grad_in(x.shape(), x.device());
    relu_bwd(x.data(), grad_out.data(), grad_in.data(), x.numel(), x.device());
    return grad_in;
}

Tensor sigmoid(const Tensor& x) {
    Tensor y(x.shape(), x.device());
    sigmoid_fwd(x.data(), y.data(), x.numel(), x.device());
    return y;
}

Tensor sigmoid_backward(const Tensor& x, const Tensor& grad_out) {
    Tensor grad_in(x.shape(), x.device());
    sigmoid_bwd(x.data(), grad_out.data(), grad_in.data(), x.numel(), x.device());
    return grad_in;
}

Tensor sigmoid_backward_from_output(const Tensor& y, const Tensor& grad_out) {
    Tensor grad_in(y.shape(), y.device());
    sigmoid_bwd_from_output(y.data(), grad_out.data(), grad_in.data(), y.numel(), y.device());
    return grad_in;
}

Tensor softmax(const Tensor& x) {
    // 检查输入shape是否为2维 [n, c]
    if (x.shape().size() != 2) {
        throw std::runtime_error("Softmax requires 2D input tensor [n, c]");
    }
    
    size_t n = x.shape()[0];  // 行数
    size_t c = x.shape()[1];  // 每行元素数
    
    Tensor y(x.shape(), x.device());
    softmax_fwd(x.data(), y.data(), n, c, x.device());
    return y;
}

Tensor cross_entropy_loss(const Tensor& logits, const std::vector<int>& labels) {
    // 检查输入shape
    if (logits.shape().size() != 2) {
        throw std::runtime_error("Cross Entropy Loss requires 2D logits tensor [n, c]");
    }
    if (logits.shape()[0] != static_cast<size_t>(labels.size())) {
        throw std::runtime_error("Logits batch size must match labels count");
    }
    
    size_t n = labels.size();  // batch size
    size_t c = logits.shape()[1];  // number of classes
    
    // 处理labels设备
    const int* labels_ptr = labels.data();
    Storage labels_gpu_storage;
    if (logits.device().is_cuda()) {
        labels_gpu_storage = Storage::empty(sizeof(int) * n, Device::cuda());
        int* labels_gpu_ptr = reinterpret_cast<int*>(labels_gpu_storage.raw_ptr());
        CUDA_CHECK(cudaMemcpy(labels_gpu_ptr, labels.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
        labels_ptr = labels_gpu_ptr;
    }
    
    // 创建输出tensor
    Tensor loss({1}, logits.device());
    
    // 调用指针接口
    cross_entropy_loss_fwd(logits.data(), 
                        labels_ptr, 
                        loss.data(), 
                        n, c, 
                        logits.device());
    
    return loss;
}

Tensor cross_entropy_loss_backward(const Tensor& logits, const std::vector<int>& labels) {
    // 检查输入shape
    if (logits.shape().size() != 2) {
        throw std::runtime_error("Cross Entropy Loss requires 2D logits tensor [n, c]");
    }
    if (logits.shape()[0] != static_cast<size_t>(labels.size())) {
        throw std::runtime_error("Logits batch size must match labels count");
    }
    
    size_t n = labels.size();  // batch size
    size_t c = logits.shape()[1];  // number of classes
    
    // 处理labels设备
    const int* labels_ptr = labels.data();
    Storage labels_gpu_storage;
    if (logits.device().is_cuda()) {
        labels_gpu_storage = Storage::empty(sizeof(int) * n, Device::cuda());
        int* labels_gpu_ptr = reinterpret_cast<int*>(labels_gpu_storage.raw_ptr());
        CUDA_CHECK(cudaMemcpy(labels_gpu_ptr, labels.data(), sizeof(int) * n, cudaMemcpyHostToDevice));
        labels_ptr = labels_gpu_ptr;
    }
    
    // 创建梯度tensor
    Tensor grad_logits(logits.shape(), logits.device());
    
    // 调用指针接口
    cross_entropy_loss_bwd(logits.data(), 
                         labels_ptr, 
                         grad_logits.data(), 
                         n, c, 
                         logits.device());
    
    return grad_logits;
}

// Cross Entropy Loss CPU实现
static void cross_entropy_loss_fwd_cpu(const float* logits, const int* labels, float* loss, size_t n, size_t c) {
    float total_loss = 0.0f;
    
    for (size_t i = 0; i < n; ++i) {
        const float* row_logits = logits + i * c;
        int label = labels[i];
        
        // 数值稳定的Cross Entropy计算
        // 找到最大值用于数值稳定性
        float max_logit = row_logits[0];
        for (size_t j = 1; j < c; ++j) {
            max_logit = std::max(max_logit, row_logits[j]);
        }
        
        // 计算log-sum-exp
        float sum_exp = 0.0f;
        for (size_t j = 0; j < c; ++j) {
            sum_exp += std::exp(row_logits[j] - max_logit);
        }
        float log_sum_exp = std::log(sum_exp);
        
        // 计算loss: -log(softmax[label]) = -logit[label] + log_sum_exp
        float sample_loss = -row_logits[label] + max_logit + log_sum_exp;
        total_loss += sample_loss;
    }
    
    *loss = total_loss / n;
}

static void cross_entropy_loss_bwd_cpu(const float* logits, const int* labels, float* grad_logits, size_t n, size_t c) {
    for (size_t i = 0; i < n; ++i) {
        const float* row_logits = logits + i * c;
        float* row_grad = grad_logits + i * c;
        int label = labels[i];
        
        // 计算softmax概率
        float max_logit = row_logits[0];
        for (size_t j = 1; j < c; ++j) {
            max_logit = std::max(max_logit, row_logits[j]);
        }
        
        float sum_exp = 0.0f;
        for (size_t j = 0; j < c; ++j) {
            sum_exp += std::exp(row_logits[j] - max_logit);
        }
        
        // 计算梯度: (softmax - one_hot) / n
        for (size_t j = 0; j < c; ++j) {
            float softmax_prob = std::exp(row_logits[j] - max_logit) / sum_exp;
            float one_hot = (j == static_cast<size_t>(label)) ? 1.0f : 0.0f;
            row_grad[j] = (softmax_prob - one_hot) / n;
        }
    }
}

// Cross Entropy Loss GPU kernels
__global__ void cross_entropy_loss_fwd_kernel(const float* logits, const int* labels, float* loss_per_sample, size_t n, size_t c) {
    CUDA_KERNEL_LOOP(i, n) {
        const float* row_logits = logits + i * c;
        int label = labels[i];
        
        // 找到最大值用于数值稳定性
        float max_logit = row_logits[0];
        for (size_t j = 1; j < c; ++j) {
            max_logit = fmaxf(max_logit, row_logits[j]);
        }
        
        // 计算log-sum-exp
        float sum_exp = 0.0f;
        for (size_t j = 0; j < c; ++j) {
            sum_exp += expf(row_logits[j] - max_logit);
        }
        float log_sum_exp = logf(sum_exp);
        
        // 计算loss: -log(softmax[label]) = -logit[label] + max_logit + log_sum_exp
        loss_per_sample[i] = -row_logits[label] + max_logit + log_sum_exp;
    }
}

__global__ void cross_entropy_loss_bwd_kernel(const float* logits, const int* labels, float* grad_logits, size_t n, size_t c) {
    CUDA_KERNEL_LOOP(idx, n * c) {
        size_t i = idx / c;  // batch index
        size_t j = idx % c;  // class index
        
        const float* row_logits = logits + i * c;
        int label = labels[i];
        
        // 计算softmax概率
        float max_logit = row_logits[0];
        for (size_t k = 1; k < c; ++k) {
            max_logit = fmaxf(max_logit, row_logits[k]);
        }
        
        float sum_exp = 0.0f;
        for (size_t k = 0; k < c; ++k) {
            sum_exp += expf(row_logits[k] - max_logit);
        }
        
        // 计算梯度: (softmax - one_hot) / n
        float softmax_prob = expf(row_logits[j] - max_logit) / sum_exp;
        float one_hot = (j == static_cast<size_t>(label)) ? 1.0f : 0.0f;
        grad_logits[idx] = (softmax_prob - one_hot) / n;
    }
}

// Cross Entropy Loss 指针接口
void cross_entropy_loss_fwd(const float* logits, const int* labels, float* loss, std::size_t n, std::size_t c, const Device& dev) {
    if (dev.is_cpu()) {
        cross_entropy_loss_fwd_cpu(logits, labels, loss, n, c);
    } else {
        // GPU实现：先计算每个样本的loss，然后求平均
        int blocks = CudaGetBlocks(static_cast<int>(n));
        
        // 分配临时存储
        auto loss_per_sample = make_cuda_unique(n);
        
        // 计算每个样本的loss
        cross_entropy_loss_fwd_kernel<<<blocks, kCudaThreadsNum>>>(logits, labels, loss_per_sample.get(), n, c);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // 使用thrust求和并计算平均值
        thrust::device_ptr<float> dev_ptr(loss_per_sample.get());
        float total_loss = thrust::reduce(dev_ptr, dev_ptr + n, 0.0f, thrust::plus<float>());
        float avg_loss = total_loss / n;
        
        // 复制结果
        CUDA_CHECK(cudaMemcpy(loss, &avg_loss, sizeof(float), cudaMemcpyHostToDevice));
    }
}

void cross_entropy_loss_bwd(const float* logits, const int* labels, float* grad_logits, std::size_t n, std::size_t c, const Device& dev) {
    if (dev.is_cpu()) {
        cross_entropy_loss_bwd_cpu(logits, labels, grad_logits, n, c);
    } else {
        // GPU实现：并行计算所有元素的梯度
        int blocks = CudaGetBlocks(static_cast<int>(n * c));
        cross_entropy_loss_bwd_kernel<<<blocks, kCudaThreadsNum>>>(logits, labels, grad_logits, n, c);
        cudaDeviceSynchronize();
    }
}
