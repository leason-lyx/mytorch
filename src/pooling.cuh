#pragma once

#include "tensor.cuh"

// Pointer interfaces (operate on contiguous row-major buffers)

// input:  (N, C, H_in, W_in)
// output: (N, C, H_out, W_out) where
// H_out = (H_in - pool_height) / stride_height + 1
// W_out = (W_in - pool_width) / stride_width + 1
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
                    Device device);

// input:      (N, C, H_in, W_in)
// output:     (N, C, H_out, W_out)
// grad_output: (N, C, H_out, W_out)
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
                    float* grad_input);

// Tensor interfaces
// input:  (N, C, H_in, W_in)
// output: (N, C, H_out, W_out) where
// H_out = (H_in - pool_height) / stride_height + 1
// W_out = (W_in - pool_width) / stride_width + 1
Tensor max_pool2d(const Tensor& input,
                  size_t pool_height,
                  size_t pool_width,
                  size_t stride_height,
                  size_t stride_width);

Tensor max_pool2d_backward(const Tensor& input,
                           const Tensor& output,
                           size_t pool_height,
                           size_t pool_width,
                           size_t stride_height,
                           size_t stride_width,
                           const Tensor& grad_output,
                           Tensor& grad_input);