#pragma once

#include "tensor.cuh"

#include <tuple>

// Pointer interfaces (operate on contiguous row-major buffers)

// output(batch_size*out_features) = input(batch_size*in_features) *
// weights(in_features*out_features) + bias(out_features)
void fc_fwd(const float *input, float *output, const float *weights, const float *bias,
            int batch_size, int in_features, int out_features);

void fc_bwd(const float *input, const float *output, const float *weights, const float *bias,
            int batch_size, int in_features, int out_features,
            const float *grad_output, float *grad_input, float *grad_weights,
            float *grad_bias);

// Tensor-level wrappers

// input.shape = (batch_size, in_features) output.shape = (batch_size, out_features)
// weight.shape = (in_features, out_features) bias.shape = (out_features)
Tensor linear(const Tensor &input, const Tensor &weight, const Tensor &bias);

// Returns (grad_input, grad_weight, grad_bias)
// input.shape = (batch_size, in_features)
// weight.shape = (in_features, out_features)
// bias.shape = (out_features)
std::tuple<Tensor, Tensor, Tensor> linear_backward(const Tensor &input,
                                                   const Tensor &weight,
                                                   const Tensor &bias,
                                                   const Tensor &grad_output);
