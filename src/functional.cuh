#pragma once
#include "tensor.cuh"
#include <cstddef>

// 本文件实现激活函数 ReLU / Sigmoid 的正向与反向（CPU 与 CUDA 均在 functional.cu 中提供）
// 约定：输入/输出张量需为连续内存（contiguous）

// Tensor接口
Tensor relu(const Tensor& x); // y = ReLU(x) = max(x, 0)

Tensor relu_backward(const Tensor& x, const Tensor& grad_out); // dx = dL/dx = (x > 0) ? dL/dy : 0

Tensor sigmoid(const Tensor& x); // y = sigmoid(x) = 1 / (1 + exp(-x))

Tensor sigmoid_backward(const Tensor& x,
                        const Tensor& grad_out); // dx = dL/dx = dL/dy * y * (1 - y)

Tensor sigmoid_backward_from_output(
    const Tensor& y,
    const Tensor& grad_out); // 若前向已缓存 y = sigmoid(x)，可使用该版本省一次前向计算

Tensor
softmax(const Tensor& x); // y = softmax(x) = exp(x - max(x)) / sum(exp(x - max(x))) for each row

Tensor cross_entropy_loss(const Tensor& logits,
                          const std::vector<int>& labels); // Cross Entropy Loss with Softmax

Tensor
cross_entropy_loss_backward(const Tensor& logits,
                            const std::vector<int>& labels); // Gradient for Cross Entropy Loss with Softmax

// 指针接口
void relu_fwd(const float* x,
              float* y,
              std::size_t n,
              const Device& dev); // y = ReLU(x) = max(x, 0)

void relu_bwd(const float* x,
              const float* grad_out,
              float* grad_in,
              std::size_t n,
              const Device& dev); // dx = dL/dx = (x > 0) ? dL/dy : 0

void sigmoid_fwd(const float* x,
                 float* y,
                 std::size_t n,
                 const Device& dev); // y = sigmoid(x) = 1 / (1 + exp(-x))

void sigmoid_bwd(const float* x,
                 const float* grad_out,
                 float* grad_in,
                 std::size_t n,
                 const Device& dev); // dx = dL/dx = dL/dy * y * (1 - y)

void sigmoid_bwd_from_output(
    const float* y,
    const float* grad_out,
    float* grad_in,
    std::size_t n,
    const Device& dev); // 若前向已缓存 y = sigmoid(x)，可使用该版本省一次前向计算

void softmax_fwd(const float* x,
                 float* y,
                 std::size_t n,
                 std::size_t c,
                 const Device& dev); // y = softmax(x) for each row, n: rows, c: columns per row

void cross_entropy_loss_fwd(const float* logits,
                            const int* labels,
                            float* loss,
                            std::size_t n,
                            std::size_t c,
                            const Device& dev); // Forward pass

void cross_entropy_loss_bwd(const float* logits,
                            const int* labels,
                            float* grad_logits,
                            std::size_t n,
                            std::size_t c,
                            const Device& dev); // Backward pass
