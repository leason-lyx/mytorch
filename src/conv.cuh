#pragma once

#include "tensor.cuh"

// Pointer interfaces (operate on contiguous row-major buffers)

// input shape: (batch_size, in_channels, input_height, input_width)
// output shape: (batch_size, out_channels, output_height, output_width)
// weights shape: (out_channels, in_channels, kernel_height, kernel_width)
// bias shape: (out_channels)
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
                int padding_width);

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
                float* grad_bias);

// Tensor-level wrappers

// input shape: (batch_size, in_channels, input_height, input_width)
// output shape: (batch_size, out_channels, output_height, output_width)
// weights shape: (out_channels, in_channels, kernel_height, kernel_width)
// bias shape: (out_channels)
Tensor conv2d(const Tensor& input,
              const Tensor& weights,
              const Tensor& bias,
              int stride_height,
              int stride_width,
              int padding_height,
              int padding_width);

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
                     Tensor& grad_bias);