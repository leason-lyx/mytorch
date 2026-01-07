#include "src/conv.cuh"
#include "src/functional.cuh"
#include "src/linear.cuh"
#include "src/pooling.cuh"
#include "src/tensor.cuh"

#include <cassert>
#include <cmath>
#include <iostream>
#include <tuple>
#include <vector>

void test_tensor() {
    std::vector<Tensor::index_t> shape = {2, 3};

    // 验证 from_vector 的尺寸校验会抛出异常
    std::vector<float> wrong_payload = {0, 1, 2, 3, 4, 5, 6};
    bool caught = false;
    try {
        Tensor bad(shape, Device::cpu());
        bad.from_vector(wrong_payload);
    } catch (const std::runtime_error& e) {
        caught = true;
        // std::cout << "Caught expected error: " << e.what() << std::endl;
    }
    assert(caught);

    // 使用正确尺寸继续接口测试
    std::vector<float> payload = {0, 1, 2, 3, 4, 5};
    Tensor tensor(shape, Device::cpu());
    tensor.from_vector(payload);

    Tensor c = tensor.cpu();
    Tensor g = tensor.gpu();

    assert(tensor.to_vector() == payload);
    assert(c.device().is_cpu());
    assert(g.device().is_cuda());
    assert(c.to_vector() == payload);
    std::cout << "Tensor from_vector/to_vector interface test passed" << std::endl;

    Tensor gc = g.cpu();
    assert(gc.device().is_cpu());
    assert(gc.to_vector() == payload);

    Tensor cg = c.gpu();
    assert(cg.device().is_cuda());
    assert(cg.cpu().to_vector() == payload);

    std::cout << "Tensor cpu()/gpu() interface test passed" << std::endl << std::endl;
}

void test_relu() {
    std::vector<Tensor::index_t> shape = {2, 3};
    std::vector<float> payload = {-1, -0.5, 0, 0.5, 1, 1.5};
    std::vector<float> expected = {0, 0, 0, 0.5, 1, 1.5};

    Tensor x(shape, Device::cpu());
    x.from_vector(payload);

    Tensor y = relu(x);
    assert(y.to_vector() == expected);

    Tensor grad_out(shape, Device::cpu());
    grad_out.from_vector(std::vector<float>(6, 1.0f)); // 全部梯度为1

    Tensor grad_in = relu_backward(x, grad_out);
    std::vector<float> expected_grad_in = {0, 0, 0, 1, 1, 1};
    assert(grad_in.to_vector() == expected_grad_in);

    std::cout << "ReLU forward/backward test passed" << std::endl;

    // GPU 测试
    Tensor x_gpu(shape, Device::cuda());
    x_gpu.from_vector(payload);

    Tensor y_gpu = relu(x_gpu);
    assert(y_gpu.device().is_cuda());
    auto y_gpu_vec = y_gpu.to_vector();
    for (size_t i = 0; i < y_gpu_vec.size(); ++i) {
        assert(std::abs(y_gpu_vec[i] - expected[i]) < 1e-7f);
    }

    Tensor grad_out_gpu(shape, Device::cuda());
    grad_out_gpu.from_vector(std::vector<float>(6, 1.0f));

    Tensor grad_in_gpu = relu_backward(x_gpu, grad_out_gpu);
    assert(grad_in_gpu.device().is_cuda());
    auto grad_in_gpu_vec = grad_in_gpu.to_vector();
    for (size_t i = 0; i < grad_in_gpu_vec.size(); ++i) {
        assert(std::abs(grad_in_gpu_vec[i] - expected_grad_in[i]) < 1e-7f);
    }

    std::cout << "ReLU GPU forward/backward test passed" << std::endl << std::endl;
}

void test_sigmoid() {
    std::vector<Tensor::index_t> shape = {2, 3};
    std::vector<float> payload = {-1, -0.5, 0, 0.5, 1, 1.5};
    std::vector<float> expected = {
        0.26894142f, 0.37754068f, 0.5f, 0.62245933f, 0.73105858f, 0.81757448f};

    Tensor x(shape, Device::cpu());
    x.from_vector(payload);

    Tensor y = sigmoid(x);
    auto y_vec = y.to_vector();
    for (size_t i = 0; i < y_vec.size(); ++i) {
        assert(std::abs(y_vec[i] - expected[i]) < 1e-6);
    }

    Tensor grad_out(shape, Device::cpu());
    grad_out.from_vector(std::vector<float>(6, 1.0f)); // 全部梯度为1

    Tensor grad_in_from_x = sigmoid_backward(x, grad_out);
    Tensor grad_in_from_y = sigmoid_backward_from_output(y, grad_out);
    auto grad_in_x_vec = grad_in_from_x.to_vector();
    auto grad_in_y_vec = grad_in_from_y.to_vector();
    for (size_t i = 0; i < grad_in_x_vec.size(); ++i) {
        assert(std::abs(grad_in_x_vec[i] - grad_in_y_vec[i]) < 1e-6);
    }

    std::cout << "Sigmoid forward/backward test passed" << std::endl;

    // GPU 测试
    Tensor x_gpu(shape, Device::cuda());
    x_gpu.from_vector(payload);

    Tensor y_gpu = sigmoid(x_gpu);
    assert(y_gpu.device().is_cuda());
    auto y_gpu_vec = y_gpu.to_vector();
    for (size_t i = 0; i < y_gpu_vec.size(); ++i) {
        assert(std::abs(y_gpu_vec[i] - expected[i]) < 1e-6);
    }

    Tensor grad_out_gpu(shape, Device::cuda());
    grad_out_gpu.from_vector(std::vector<float>(6, 1.0f));

    Tensor grad_in_x_gpu = sigmoid_backward(x_gpu, grad_out_gpu);
    Tensor grad_in_y_gpu = sigmoid_backward_from_output(y_gpu, grad_out_gpu);
    assert(grad_in_x_gpu.device().is_cuda());
    assert(grad_in_y_gpu.device().is_cuda());
    auto grad_in_x_gpu_vec = grad_in_x_gpu.to_vector();
    auto grad_in_y_gpu_vec = grad_in_y_gpu.to_vector();
    for (size_t i = 0; i < grad_in_x_gpu_vec.size(); ++i) {
        assert(std::abs(grad_in_x_gpu_vec[i] - grad_in_y_gpu_vec[i]) < 1e-6);
        assert(std::abs(grad_in_x_gpu_vec[i] - grad_in_x_vec[i]) < 1e-6);
    }

    std::cout << "Sigmoid GPU forward/backward test passed" << std::endl << std::endl;
}

void test_softmax() {
    std::cout << "Testing Softmax..." << std::endl;

    // 测试数据：2行3列
    std::vector<Tensor::index_t> shape = {2, 3};
    std::vector<float> input = {1.0f,
                                2.0f,
                                3.0f, // 第一行
                                4.0f,
                                5.0f,
                                6.0f}; // 第二行

    // 手动计算期望输出
    // 第一行: exp(1)/sum, exp(2)/sum, exp(3)/sum
    // sum = exp(1) + exp(2) + exp(3) = 2.718 + 7.389 + 20.086 = 30.193
    // 第一行期望: [0.0900, 0.2447, 0.6652]

    // 第二行: exp(4)/sum, exp(5)/sum, exp(6)/sum
    // sum = exp(4) + exp(5) + exp(6) = 54.598 + 148.413 + 403.429 = 606.440
    // 第二行期望: [0.0900, 0.2447, 0.6652] (与第一行相同，因为每行都减去了最大值)

    std::vector<float> expected = {
        0.09003057f, 0.24472847f, 0.66524096f, 0.09003057f, 0.24472847f, 0.66524096f};

    // CPU测试
    Tensor x_cpu(shape, Device::cpu());
    x_cpu.from_vector(input);

    Tensor y_cpu = softmax(x_cpu);
    auto y_cpu_vec = y_cpu.to_vector();

    // 验证输出shape
    assert(y_cpu.shape() == shape);

    // 验证数值精度
    for (size_t i = 0; i < expected.size(); ++i) {
        assert(std::abs(y_cpu_vec[i] - expected[i]) < 1e-6f);
    }

    // 验证每行和为1
    for (size_t row = 0; row < 2; ++row) {
        float row_sum = 0.0f;
        for (size_t col = 0; col < 3; ++col) {
            row_sum += y_cpu_vec[row * 3 + col];
        }
        assert(std::abs(row_sum - 1.0f) < 1e-6f);
    }

    std::cout << "Softmax CPU forward test passed" << std::endl;

    // GPU测试
    Tensor x_gpu(shape, Device::cuda());
    x_gpu.from_vector(input);

    Tensor y_gpu = softmax(x_gpu);
    auto y_gpu_vec = y_gpu.to_vector();

    // 验证输出shape
    assert(y_gpu.shape() == shape);
    assert(y_gpu.device().is_cuda());

    // 验证数值精度
    for (size_t i = 0; i < expected.size(); ++i) {
        assert(std::abs(y_gpu_vec[i] - expected[i]) < 1e-6f);
    }

    // 验证每行和为1
    for (size_t row = 0; row < 2; ++row) {
        float row_sum = 0.0f;
        for (size_t col = 0; col < 3; ++col) {
            row_sum += y_gpu_vec[row * 3 + col];
        }
        assert(std::abs(row_sum - 1.0f) < 1e-6f);
    }

    // 验证CPU和GPU结果一致
    for (size_t i = 0; i < y_cpu_vec.size(); ++i) {
        assert(std::abs(y_cpu_vec[i] - y_gpu_vec[i]) < 1e-6f);
    }

    std::cout << "Softmax GPU forward test passed" << std::endl;

    // 测试错误情况：非2D输入
    Tensor wrong_shape({2, 3, 4}, Device::cpu());
    bool caught = false;
    try {
        Tensor wrong_result = softmax(wrong_shape);
    } catch (const std::runtime_error& e) {
        caught = true;
    }
    assert(caught);
    std::cout << "Softmax error handling test passed" << std::endl << std::endl;
}

void test_cross_entropy_loss() {
    std::cout << "Testing Cross Entropy Loss..." << std::endl;

    // 测试数据：3个样本，4个类别
    std::vector<Tensor::index_t> logits_shape = {3, 4};
    std::vector<Tensor::index_t> labels_shape = {3};

    // logits: [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [0.1, 0.2, 0.3, 0.4]
    std::vector<float> logits_data = {
        1.0f,
        2.0f,
        3.0f,
        4.0f, // 第一个样本
        5.0f,
        6.0f,
        7.0f,
        8.0f, // 第二个样本
        0.1f,
        0.2f,
        0.3f,
        0.4f // 第三个样本
    };

    // labels: [0, 1, 2] (正确类别)
    std::vector<int> labels_data = {0, 1, 2};

    // 手动计算期望loss
    // 样本0: logits [1,2,3,4], label 0, max=4, sum_exp≈1.553, log_sum_exp≈0.441, loss = -1 + 4 +
    // 0.441 = 3.441 样本1: logits [5,6,7,8], label 1, max=8, sum_exp≈1.553, log_sum_exp≈0.441, loss
    // = -6 + 8 + 0.441 = 2.441 样本2: logits [0.1,0.2,0.3,0.4], label 2, max=0.4, sum_exp≈3.464,
    // log_sum_exp≈1.243, loss = -0.3 + 0.4 + 1.243 = 1.343 平均loss = (3.441 + 2.441 + 1.343) / 3
    // = 7.225 / 3 = 2.408

    float expected_loss = 2.4076383f;

    // CPU测试
    Tensor logits_cpu(logits_shape, Device::cpu());
    logits_cpu.from_vector(logits_data);

    Tensor loss_cpu = cross_entropy_loss(logits_cpu, labels_data);
    auto loss_cpu_val = loss_cpu.to_vector()[0];

    // 验证loss精度
    assert(std::abs(loss_cpu_val - expected_loss) < 1e-6f);

    std::cout << "Cross Entropy Loss CPU forward test passed" << std::endl;

    // GPU测试
    Tensor logits_gpu(logits_shape, Device::cuda());
    logits_gpu.from_vector(logits_data);

    Tensor loss_gpu = cross_entropy_loss(logits_gpu, labels_data);
    auto loss_gpu_val = loss_gpu.to_vector()[0];

    // 验证loss精度
    assert(std::abs(loss_gpu_val - expected_loss) < 1e-6f);

    // 验证CPU和GPU结果一致
    assert(std::abs(loss_cpu_val - loss_gpu_val) < 1e-6f);

    std::cout << "Cross Entropy Loss GPU forward test passed" << std::endl;

    // 测试反向传播
    Tensor grad_logits_cpu = cross_entropy_loss_backward(logits_cpu, labels_data);
    Tensor grad_logits_gpu = cross_entropy_loss_backward(logits_gpu, labels_data);

    // 验证梯度shape
    assert(grad_logits_cpu.shape() == logits_shape);
    assert(grad_logits_gpu.shape() == logits_shape);

    // 验证CPU和GPU梯度一致
    auto grad_cpu_vec = grad_logits_cpu.to_vector();
    auto grad_gpu_vec = grad_logits_gpu.to_vector();

    for (size_t i = 0; i < grad_cpu_vec.size(); ++i) {
        assert(std::abs(grad_cpu_vec[i] - grad_gpu_vec[i]) < 1e-6f);
    }

    std::cout << "Cross Entropy Loss backward test passed" << std::endl;

    // 测试错误情况：shape不匹配
    Tensor wrong_logits_shape({2, 3}, Device::cpu());
    std::vector<int> wrong_labels = {0, 1};

    bool caught = false;
    try {
        cross_entropy_loss(wrong_logits_shape, labels_data);
    } catch (const std::runtime_error& e) {
        caught = true;
    }
    assert(caught);

    caught = false;
    try {
        cross_entropy_loss(logits_cpu, wrong_labels);
    } catch (const std::runtime_error& e) {
        caught = true;
    }
    assert(caught);

    std::cout << "Cross Entropy Loss error handling test passed" << std::endl << std::endl;
}

void test_linear() {
    // 测试配置: batch_size=2, in_features=3, out_features=4
    int batch_size = 2;
    int in_features = 3;
    int out_features = 4;

    // 准备输入数据
    std::vector<float> input_data = {
        1.0f,
        2.0f,
        3.0f, // 第一个样本
        4.0f,
        5.0f,
        6.0f // 第二个样本
    };

    // 准备权重数据 (in_features × out_features)
    std::vector<float> weight_data = {
        0.1f,
        0.2f,
        0.3f,
        0.4f, // 第一个输入特征
        0.5f,
        0.6f,
        0.7f,
        0.8f, // 第二个输入特征
        0.9f,
        1.0f,
        1.1f,
        1.2f // 第三个输入特征
    };

    // 准备偏置数据
    std::vector<float> bias_data = {0.1f, 0.2f, 0.3f, 0.4f};

    // CPU 上直接计算期望输出：output = input @ weight + bias
    // output[i][j] = sum_k(input[i][k] * weight[k][j]) + bias[j]
    std::vector<float> expected_output(batch_size * out_features);
    for (int i = 0; i < batch_size; ++i) {
        for (int j = 0; j < out_features; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < in_features; ++k) {
                sum += input_data[i * in_features + k] * weight_data[k * out_features + j];
            }
            expected_output[i * out_features + j] = sum + bias_data[j];
        }
    }

    // 假设 grad_output 全为 1，CPU 上计算期望梯度
    std::vector<float> grad_output_data(batch_size * out_features, 1.0f);

    // grad_input = grad_output @ weight.T
    // grad_input[i][k] = sum_j(grad_output[i][j] * weight[k][j])
    std::vector<float> expected_grad_input(batch_size * in_features);
    for (int i = 0; i < batch_size; ++i) {
        for (int k = 0; k < in_features; ++k) {
            float sum = 0.0f;
            for (int j = 0; j < out_features; ++j) {
                sum += grad_output_data[i * out_features + j] * weight_data[k * out_features + j];
            }
            expected_grad_input[i * in_features + k] = sum;
        }
    }

    // grad_weight = input.T @ grad_output
    // grad_weight[k][j] = sum_i(input[i][k] * grad_output[i][j])
    std::vector<float> expected_grad_weight(in_features * out_features);
    for (int k = 0; k < in_features; ++k) {
        for (int j = 0; j < out_features; ++j) {
            float sum = 0.0f;
            for (int i = 0; i < batch_size; ++i) {
                sum += input_data[i * in_features + k] * grad_output_data[i * out_features + j];
            }
            expected_grad_weight[k * out_features + j] = sum;
        }
    }

    // grad_bias = sum(grad_output, axis=0)
    // grad_bias[j] = sum_i(grad_output[i][j])
    std::vector<float> expected_grad_bias(out_features);
    for (int j = 0; j < out_features; ++j) {
        float sum = 0.0f;
        for (int i = 0; i < batch_size; ++i) {
            sum += grad_output_data[i * out_features + j];
        }
        expected_grad_bias[j] = sum;
    }

    // GPU 测试
    Tensor input_gpu(
        {static_cast<Tensor::index_t>(batch_size), static_cast<Tensor::index_t>(in_features)},
        Device::cuda());
    input_gpu.from_vector(input_data);

    Tensor weight_gpu(
        {static_cast<Tensor::index_t>(in_features), static_cast<Tensor::index_t>(out_features)},
        Device::cuda());
    weight_gpu.from_vector(weight_data);

    Tensor bias_gpu({static_cast<Tensor::index_t>(out_features)}, Device::cuda());
    bias_gpu.from_vector(bias_data);

    Tensor output_gpu = linear(input_gpu, weight_gpu, bias_gpu);
    assert(output_gpu.device().is_cuda());
    auto output_gpu_vec = output_gpu.to_vector();

    // 验证 GPU 输出与 CPU 输出一致
    for (size_t i = 0; i < expected_output.size(); ++i) {
        assert(std::abs(output_gpu_vec[i] - expected_output[i]) < 1e-4f);
    }

    std::cout << "Linear GPU forward test passed" << std::endl;

    // GPU 反向传播测试
    Tensor grad_output_gpu(
        {static_cast<Tensor::index_t>(batch_size), static_cast<Tensor::index_t>(out_features)},
        Device::cuda());
    grad_output_gpu.from_vector(std::vector<float>(batch_size * out_features, 1.0f));

    auto [grad_input_gpu, grad_weight_gpu, grad_bias_gpu] =
        linear_backward(input_gpu, weight_gpu, bias_gpu, grad_output_gpu);

    assert(grad_input_gpu.device().is_cuda());
    assert(grad_weight_gpu.device().is_cuda());
    assert(grad_bias_gpu.device().is_cuda());

    auto grad_input_gpu_vec = grad_input_gpu.to_vector();
    auto grad_weight_gpu_vec = grad_weight_gpu.to_vector();
    auto grad_bias_gpu_vec = grad_bias_gpu.to_vector();

    // 验证 GPU 梯度与 CPU 梯度一致
    for (size_t i = 0; i < expected_grad_input.size(); ++i) {
        assert(std::abs(grad_input_gpu_vec[i] - expected_grad_input[i]) < 1e-4f);
    }

    for (size_t i = 0; i < expected_grad_weight.size(); ++i) {
        assert(std::abs(grad_weight_gpu_vec[i] - expected_grad_weight[i]) < 1e-4f);
    }

    for (size_t i = 0; i < expected_grad_bias.size(); ++i) {
        assert(std::abs(grad_bias_gpu_vec[i] - expected_grad_bias[i]) < 1e-4f);
    }

    std::cout << "Linear GPU backward test passed" << std::endl << std::endl;
}

void test_max_pool2d() {
    std::cout << "Testing Max-Pooling..." << std::endl;

    // Test parameters
    size_t batch_size = 1;
    size_t in_channels = 1;
    size_t in_height = 4;
    size_t in_width = 4;
    size_t pool_height = 2;
    size_t pool_width = 2;
    size_t stride_height = 2;
    size_t stride_width = 2;

    size_t out_height = (in_height - pool_height) / stride_height + 1;
    size_t out_width = (in_width - pool_width) / stride_width + 1;

    // Input data
    std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};

    // Expected output
    std::vector<float> expected_output = {6, 8, 14, 16};

    // CPU Forward Test
    Tensor input_cpu({batch_size, in_channels, in_height, in_width}, Device::cpu());
    input_cpu.from_vector(input_data);

    Tensor output_cpu = max_pool2d(input_cpu, pool_height, pool_width, stride_height, stride_width);
    auto output_cpu_vec = output_cpu.to_vector();

    assert(output_cpu.shape() ==
           std::vector<Tensor::index_t>({batch_size, in_channels, out_height, out_width}));
    for (size_t i = 0; i < expected_output.size(); ++i) {
        assert(std::abs(output_cpu_vec[i] - expected_output[i]) < 1e-7f);
    }
    std::cout << "Max-Pooling CPU forward test passed" << std::endl;

    // CPU Backward Test
    std::vector<float> grad_output_data = {1, 1, 1, 1};
    Tensor grad_output_cpu({batch_size, in_channels, out_height, out_width}, Device::cpu());
    grad_output_cpu.from_vector(grad_output_data);

    Tensor grad_input_cpu(input_cpu.shape(), Device::cpu());
    max_pool2d_backward(input_cpu,
                        output_cpu,
                        pool_height,
                        pool_width,
                        stride_height,
                        stride_width,
                        grad_output_cpu,
                        grad_input_cpu);

    std::vector<float> expected_grad_input = {0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1};
    auto grad_input_cpu_vec = grad_input_cpu.to_vector();
    for (size_t i = 0; i < expected_grad_input.size(); ++i) {
        assert(std::abs(grad_input_cpu_vec[i] - expected_grad_input[i]) < 1e-7f);
    }
    std::cout << "Max-Pooling CPU backward test passed" << std::endl;

    // GPU Forward Test
    Tensor input_gpu({batch_size, in_channels, in_height, in_width}, Device::cuda());
    input_gpu.from_vector(input_data);

    Tensor output_gpu = max_pool2d(input_gpu, pool_height, pool_width, stride_height, stride_width);
    auto output_gpu_vec = output_gpu.to_vector();

    assert(output_gpu.shape() ==
           std::vector<Tensor::index_t>({batch_size, in_channels, out_height, out_width}));
    for (size_t i = 0; i < expected_output.size(); ++i) {
        assert(std::abs(output_gpu_vec[i] - expected_output[i]) < 1e-7f);
    }
    std::cout << "Max-Pooling GPU forward test passed" << std::endl;

    // GPU Backward Test
    Tensor grad_output_gpu({batch_size, in_channels, out_height, out_width}, Device::cuda());
    grad_output_gpu.from_vector(grad_output_data);

    Tensor grad_input_gpu(input_gpu.shape(), Device::cuda());
    max_pool2d_backward(input_gpu,
                        output_gpu,
                        pool_height,
                        pool_width,
                        stride_height,
                        stride_width,
                        grad_output_gpu,
                        grad_input_gpu);

    auto grad_input_gpu_vec = grad_input_gpu.to_vector();
    for (size_t i = 0; i < expected_grad_input.size(); ++i) {
        assert(std::abs(grad_input_gpu_vec[i] - expected_grad_input[i]) < 1e-7f);
    }
    std::cout << "Max-Pooling GPU backward test passed" << std::endl << std::endl;
}

void test_conv2d() {
    std::cout << "Testing Conv2d..." << std::endl;

    // 简单的卷积测试：
    // Input: 1×1×4×4 (batch_size=1, in_channels=1, height=4, width=4)
    // Weights: 1×1×3×3 (out_channels=1, in_channels=1, kernel_h=3, kernel_w=3)
    // Bias: 1
    // stride=1, padding=0
    // Expected output: 1×1×2×2

    Tensor input({1, 1, 4, 4}, Device::cuda());
    std::vector<float> input_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    input.from_vector(input_data);

    Tensor weights({1, 1, 3, 3}, Device::cuda());
    std::vector<float> weights_data = {1, 0, -1, 1, 0, -1, 1, 0, -1};
    weights.from_vector(weights_data);

    Tensor bias({1}, Device::cuda());
    std::vector<float> bias_data = {0.5f};
    bias.from_vector(bias_data);

    // Forward pass
    Tensor output = conv2d(input, weights, bias, 1, 1, 0, 0);

    assert(output.shape()[0] == 1);
    assert(output.shape()[1] == 1);
    assert(output.shape()[2] == 2);
    assert(output.shape()[3] == 2);

    std::vector<float> output_vec = output.to_vector();

    // 手动计算期望值
    // output[0,0] = (1*1 + 2*0 + 3*(-1) + 5*1 + 6*0 + 7*(-1) + 9*1 + 10*0 + 11*(-1)) + 0.5
    //             = (1 + 0 - 3 + 5 + 0 - 7 + 9 + 0 - 11) + 0.5 = -6 + 0.5 = -5.5
    // output[0,1] = (2*1 + 3*0 + 4*(-1) + 6*1 + 7*0 + 8*(-1) + 10*1 + 11*0 + 12*(-1)) + 0.5
    //             = (2 + 0 - 4 + 6 + 0 - 8 + 10 + 0 - 12) + 0.5 = -6 + 0.5 = -5.5
    // output[1,0] = (5*1 + 6*0 + 7*(-1) + 9*1 + 10*0 + 11*(-1) + 13*1 + 14*0 + 15*(-1)) + 0.5
    //             = (5 + 0 - 7 + 9 + 0 - 11 + 13 + 0 - 15) + 0.5 = -6 + 0.5 = -5.5
    // output[1,1] = (6*1 + 7*0 + 8*(-1) + 10*1 + 11*0 + 12*(-1) + 14*1 + 15*0 + 16*(-1)) + 0.5
    //             = (6 + 0 - 8 + 10 + 0 - 12 + 14 + 0 - 16) + 0.5 = -6 + 0.5 = -5.5

    std::vector<float> expected_output = {-5.5f, -5.5f, -5.5f, -5.5f};

    for (size_t i = 0; i < expected_output.size(); ++i) {
        assert(std::abs(output_vec[i] - expected_output[i]) < 1e-4f);
    }

    std::cout << "Conv2d forward test passed" << std::endl;

    std::cout << "Testing Conv2d backward..." << std::endl;
    // Test backward pass
    Tensor grad_output({1, 1, 2, 2}, Device::cuda());
    std::vector<float> grad_output_data = {1.0f, 2.0f, 3.0f, 4.0f};
    grad_output.from_vector(grad_output_data);

    Tensor grad_input({1, 1, 4, 4}, Device::cuda());
    Tensor grad_weights({1, 1, 3, 3}, Device::cuda());
    Tensor grad_bias({1}, Device::cuda());

    conv2d_backward(
        input, output, weights, bias, 1, 1, 0, 0, grad_output, grad_input, grad_weights, grad_bias);

    // 验证 grad_bias
    std::vector<float> grad_bias_vec = grad_bias.to_vector();
    // grad_bias 应该是所有 grad_output 的和 = 1 + 2 + 3 + 4 = 10.0
    assert(std::abs(grad_bias_vec[0] - 10.0f) < 1e-4f);
    std::cout << "  grad_bias test passed" << std::endl;

    // 验证 grad_weights
    // grad_weights[kh, kw] = sum over all output positions of:
    //   grad_output[h_out, w_out] * input[h_out*stride + kh, w_out*stride + kw]
    //
    // 对于我们的测试：
    // grad_weights[0,0] = 1*1 + 2*2 + 3*5 + 4*6 = 1 + 4 + 15 + 24 = 44
    // grad_weights[0,1] = 1*2 + 2*3 + 3*6 + 4*7 = 2 + 6 + 18 + 28 = 54
    // grad_weights[0,2] = 1*3 + 2*4 + 3*7 + 4*8 = 3 + 8 + 21 + 32 = 64
    // grad_weights[1,0] = 1*5 + 2*6 + 3*9 + 4*10 = 5 + 12 + 27 + 40 = 84
    // grad_weights[1,1] = 1*6 + 2*7 + 3*10 + 4*11 = 6 + 14 + 30 + 44 = 94
    // grad_weights[1,2] = 1*7 + 2*8 + 3*11 + 4*12 = 7 + 16 + 33 + 48 = 104
    // grad_weights[2,0] = 1*9 + 2*10 + 3*13 + 4*14 = 9 + 20 + 39 + 56 = 124
    // grad_weights[2,1] = 1*10 + 2*11 + 3*14 + 4*15 = 10 + 22 + 42 + 60 = 134
    // grad_weights[2,2] = 1*11 + 2*12 + 3*15 + 4*16 = 11 + 24 + 45 + 64 = 144
    std::vector<float> grad_weights_vec = grad_weights.to_vector();
    std::vector<float> expected_grad_weights = {44, 54, 64, 84, 94, 104, 124, 134, 144};

    for (size_t i = 0; i < expected_grad_weights.size(); ++i) {
        assert(std::abs(grad_weights_vec[i] - expected_grad_weights[i]) < 1e-3f);
    }
    std::cout << "  grad_weights test passed" << std::endl;

    // 验证 grad_input
    // grad_input 是通过 col2im 累加得到的
    // 对于每个输入位置 input[h,w]，它会被多个卷积核位置访问到
    // grad_input[h,w] = sum over all (kh, kw, h_out, w_out) where input[h,w] contributes:
    //   grad_output[h_out, w_out] * weights[kh, kw]
    //
    // 让我们手动计算一些关键位置：
    // grad_input[0,0] 只被 (kh=0, kw=0, h_out=0, w_out=0) 访问
    //   = grad_output[0,0] * weights[0,0] = 1 * 1 = 1
    // grad_input[1,1] 被 4 个位置访问：
    //   (kh=0, kw=0, h_out=1, w_out=1): grad_output[1,1] * weights[0,0] = 4 * 1 = 4
    //   (kh=0, kw=1, h_out=1, w_out=0): grad_output[1,0] * weights[0,1] = 3 * 0 = 0
    //   (kh=1, kw=0, h_out=0, w_out=1): grad_output[0,1] * weights[1,0] = 2 * 1 = 2
    //   (kh=1, kw=1, h_out=0, w_out=0): grad_output[0,0] * weights[1,1] = 1 * 0 = 0
    //   总和 = 4 + 0 + 2 + 0 = 6
    std::vector<float> grad_input_vec = grad_input.to_vector();

    // 计算完整的 expected_grad_input
    // 这里用简单的方法：手动计算每个位置
    std::vector<float> expected_grad_input(16, 0.0f);

    // 对于每个输出位置 (h_out, w_out) 和每个卷积核位置 (kh, kw)
    for (int h_out = 0; h_out < 2; ++h_out) {
        for (int w_out = 0; w_out < 2; ++w_out) {
            float grad_out_val = grad_output_data[h_out * 2 + w_out];
            for (int kh = 0; kh < 3; ++kh) {
                for (int kw = 0; kw < 3; ++kw) {
                    int h_in = h_out + kh;
                    int w_in = w_out + kw;
                    float weight_val = weights_data[kh * 3 + kw];
                    expected_grad_input[h_in * 4 + w_in] += grad_out_val * weight_val;
                }
            }
        }
    }

    for (size_t i = 0; i < expected_grad_input.size(); ++i) {
        if (std::abs(grad_input_vec[i] - expected_grad_input[i]) >= 1e-3f) {
            std::cout << "  Mismatch at index " << i << ": actual=" << grad_input_vec[i]
                      << ", expected=" << expected_grad_input[i] << std::endl;
        }
        assert(std::abs(grad_input_vec[i] - expected_grad_input[i]) < 1e-3f);
    }
    std::cout << "  grad_input test passed" << std::endl;

    std::cout << "Conv2d backward test passed" << std::endl << std::endl;
}

int main() {
    try {
        test_tensor();
        test_relu();
        test_sigmoid();
        test_softmax();
        test_cross_entropy_loss();
        test_linear();
        test_conv2d();
        test_max_pool2d();

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Test failed with exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "All tests passed" << std::endl;
    return EXIT_SUCCESS;
}
