import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import torch.nn.functional as F
import numpy as np

import mytensor

# Helper to create random tensors
def get_random_data(shape, dtype=np.float32):
    return np.random.randn(*shape).astype(dtype)

def check_tensors(my_tensor, torch_tensor, atol=1e-4, rtol=1e-3):
    my_np = my_tensor.numpy()
    torch_np = torch_tensor.detach().numpy()
    
    assert my_np.shape == torch_np.shape, f"Shape mismatch: {my_np.shape} vs {torch_np.shape}"
    np.testing.assert_allclose(my_np, torch_np, atol=atol, rtol=rtol)

def check_gradients(my_tensor, torch_tensor, atol=1e-4, rtol=1e-3):
    assert my_tensor.grad is not None, "MyTensor grad is None"
    assert torch_tensor.grad is not None, "Torch grad is None"
    check_tensors(my_tensor.grad, torch_tensor.grad, atol=atol, rtol=rtol)

def test_sigmoid():
    shape = (1024, 1024)
    data = get_random_data(shape)
    
    if not torch.cuda.is_available():
        pytest.fail("CUDA not available, but required for sigmoid test")

    # MyTensor
    t = mytensor.Tensor(data).gpu()
    out_my = mytensor.sigmoid(t).cpu()
    
    # PyTorch
    t_torch = torch.tensor(data)
    out_torch = torch.sigmoid(t_torch)
    
    check_tensors(out_my, out_torch)

def test_relu():
    shape = (1024, 1024)
    data = get_random_data(shape)
    
    if not torch.cuda.is_available():
        pytest.fail("CUDA not available, but required for relu test")

    # MyTensor
    t = mytensor.Tensor(data).gpu()
    out_my = mytensor.relu(t).cpu()
    
    # PyTorch
    t_torch = torch.tensor(data)
    out_torch = torch.relu(t_torch)
    
    check_tensors(out_my, out_torch)

def test_linear():
    batch_size = 128
    in_features = 1024
    out_features = 512
    
    input_data = get_random_data((batch_size, in_features))
    # MyTensor expects (in_features, out_features)
    weight_data = get_random_data((in_features, out_features)) 
    bias_data = get_random_data((out_features,))
    
    # MyTensor
    t_in = mytensor.Tensor(input_data)
    t_w = mytensor.Tensor(weight_data)
    t_b = mytensor.Tensor(bias_data)
    
    if not torch.cuda.is_available():
        pytest.fail("CUDA not available, but required for linear test")

    t_in = t_in.gpu()
    t_w = t_w.gpu()
    t_b = t_b.gpu()
    out_my = mytensor.linear(t_in, t_w, t_b)
    out_my = out_my.cpu() # Move back to CPU for comparison
    
    # PyTorch
    # F.linear expects weight as (out_features, in_features)
    t_in_torch = torch.tensor(input_data)
    t_w_torch = torch.tensor(weight_data.T) # Transpose for PyTorch
    t_b_torch = torch.tensor(bias_data)
    out_torch = F.linear(t_in_torch, t_w_torch, t_b_torch)
    
    check_tensors(out_my, out_torch)

def test_conv2d():
    N, C_in, H, W = 32, 64, 64, 64
    C_out = 128
    K = 3
    stride = 1
    padding = 1
    
    input_data = get_random_data((N, C_in, H, W))
    weight_data = get_random_data((C_out, C_in, K, K))
    bias_data = get_random_data((C_out,))
    
    # MyTensor
    t_in = mytensor.Tensor(input_data)
    t_w = mytensor.Tensor(weight_data)
    t_b = mytensor.Tensor(bias_data)
    
    if not torch.cuda.is_available():
        pytest.fail("CUDA not available, but required for conv2d test")

    t_in = t_in.gpu()
    t_w = t_w.gpu()
    t_b = t_b.gpu()
    out_my = mytensor.conv2d(t_in, t_w, t_b, stride=stride, padding=padding)
    out_my = out_my.cpu()
    
    # PyTorch
    t_in_torch = torch.tensor(input_data)
    t_w_torch = torch.tensor(weight_data)
    t_b_torch = torch.tensor(bias_data)
    out_torch = F.conv2d(t_in_torch, t_w_torch, t_b_torch, stride=stride, padding=padding)
    
    check_tensors(out_my, out_torch, atol=1e-3, rtol=1e-3)

def test_max_pool2d():
    N, C, H, W = 32, 64, 64, 64
    pool_size = 2
    stride = 2
    
    input_data = get_random_data((N, C, H, W))
    
    if not torch.cuda.is_available():
        pytest.fail("CUDA not available, but required for max_pool2d test")

    # MyTensor
    t_in = mytensor.Tensor(input_data).gpu()
    out_my = mytensor.max_pool2d(t_in, pool_size, stride=stride).cpu()
    
    # PyTorch
    t_in_torch = torch.tensor(input_data)
    out_torch = F.max_pool2d(t_in_torch, kernel_size=pool_size, stride=stride)
    
    check_tensors(out_my, out_torch)

def test_softmax():
    shape = (128, 1000)
    data = get_random_data(shape)
    
    if not torch.cuda.is_available():
        pytest.fail("CUDA not available, but required for softmax test")

    # MyTensor
    t = mytensor.Tensor(data).gpu()
    out_my = mytensor.softmax(t).cpu()
    
    # PyTorch
    t_torch = torch.tensor(data)
    out_torch = F.softmax(t_torch, dim=1) # Softmax along the last dimension (features)
    
    check_tensors(out_my, out_torch)

def test_cross_entropy_loss():
    batch_size = 128
    num_classes = 1000
    
    logits_data = get_random_data((batch_size, num_classes))
    # Labels are indices [0, num_classes-1]
    labels_data = np.random.randint(0, num_classes, size=(batch_size,)).astype(np.int32)
    
    if not torch.cuda.is_available():
        pytest.fail("CUDA not available, but required for cross_entropy_loss test")

    # MyTensor
    t_logits = mytensor.Tensor(logits_data).gpu()
    # mytensor.cross_entropy_loss expects std::vector<int> for labels
    # pybind11 automatically converts numpy array of int to vector<int>
    out_my = mytensor.cross_entropy_loss(t_logits, labels_data).cpu()
    
    # PyTorch
    t_logits_torch = torch.tensor(logits_data)
    t_labels_torch = torch.tensor(labels_data, dtype=torch.long)
    out_torch = F.cross_entropy(t_logits_torch, t_labels_torch)
    
    # Note: F.cross_entropy returns a scalar (mean) by default.
    # If mytensor returns a vector of losses, we might need to adjust.
    # Assuming mytensor returns the mean loss as a 1-element tensor or scalar.
    
    # If mytensor returns a scalar tensor, numpy() will return an array of shape (1,) or ().
    # We might need to squeeze it.
    
    my_np = out_my.numpy()
    torch_np = out_torch.detach().numpy()
    
    # Handle potential shape differences (e.g. (1,) vs ())
    if my_np.size == 1 and torch_np.size == 1:
        np.testing.assert_allclose(my_np.item(), torch_np.item(), atol=1e-4, rtol=1e-3)
    else:
        check_tensors(out_my, out_torch)

def test_linear_backward():
    batch_size = 4
    in_features = 8
    out_features = 5
    input_data = get_random_data((batch_size, in_features))
    weight_data = get_random_data((in_features, out_features))
    bias_data = get_random_data((out_features,))
    grad_out = get_random_data((batch_size, out_features))

    if not torch.cuda.is_available():
        pytest.fail("CUDA not available, but required for linear backward test")

    x = mytensor.Tensor(input_data, requires_grad=True, device="cuda")
    w = mytensor.Tensor(weight_data, requires_grad=True, device="cuda")
    b = mytensor.Tensor(bias_data, requires_grad=True, device="cuda")
    out_my = mytensor.linear(x, w, b)
    out_my.backward(grad_out)

    x_torch = torch.tensor(input_data, requires_grad=True)
    w_torch = torch.tensor(weight_data.T, requires_grad=True)
    b_torch = torch.tensor(bias_data, requires_grad=True)
    out_torch = F.linear(x_torch, w_torch, b_torch)
    out_torch.backward(torch.tensor(grad_out))

    check_tensors(x.grad, x_torch.grad)
    check_tensors(w.grad, w_torch.grad.T)
    check_tensors(b.grad, b_torch.grad)

def test_conv2d_backward():
    N, C_in, H, W = 2, 3, 5, 5
    C_out = 4
    K = 3
    stride = 1
    padding = 1

    input_data = get_random_data((N, C_in, H, W))
    weight_data = get_random_data((C_out, C_in, K, K))
    bias_data = get_random_data((C_out,))

    if not torch.cuda.is_available():
        pytest.fail("CUDA not available, but required for conv2d backward test")

    x = mytensor.Tensor(input_data, requires_grad=True, device="cuda")
    w = mytensor.Tensor(weight_data, requires_grad=True, device="cuda")
    b = mytensor.Tensor(bias_data, requires_grad=True, device="cuda")
    out_my = mytensor.conv2d(x, w, b, stride=stride, padding=padding)
    grad_out = get_random_data(out_my.shape)
    out_my.backward(grad_out)

    x_torch = torch.tensor(input_data, requires_grad=True)
    w_torch = torch.tensor(weight_data, requires_grad=True)
    b_torch = torch.tensor(bias_data, requires_grad=True)
    out_torch = F.conv2d(x_torch, w_torch, b_torch, stride=stride, padding=padding)
    out_torch.backward(torch.tensor(grad_out))

    check_tensors(x.grad, x_torch.grad, atol=1e-3, rtol=1e-3)
    check_tensors(w.grad, w_torch.grad, atol=1e-3, rtol=1e-3)
    check_tensors(b.grad, b_torch.grad, atol=1e-3, rtol=1e-3)

def test_relu_backward():
    shape = (8, 16)
    data = get_random_data(shape)
    grad_out = get_random_data(shape)

    if not torch.cuda.is_available():
        pytest.fail("CUDA not available, but required for relu backward test")

    x = mytensor.Tensor(data, requires_grad=True, device="cuda")
    out_my = mytensor.relu(x)
    out_my.backward(grad_out)

    x_torch = torch.tensor(data, requires_grad=True)
    out_torch = torch.relu(x_torch)
    out_torch.backward(torch.tensor(grad_out))

    check_tensors(x.grad, x_torch.grad)

def test_sigmoid_backward():
    shape = (6, 7)
    data = get_random_data(shape)
    grad_out = get_random_data(shape)

    if not torch.cuda.is_available():
        pytest.fail("CUDA not available, but required for sigmoid backward test")

    x = mytensor.Tensor(data, requires_grad=True, device="cuda")
    out_my = mytensor.sigmoid(x)
    out_my.backward(grad_out)

    x_torch = torch.tensor(data, requires_grad=True)
    out_torch = torch.sigmoid(x_torch)
    out_torch.backward(torch.tensor(grad_out))

    check_tensors(x.grad, x_torch.grad)

def test_softmax_backward():
    shape = (5, 4)
    data = get_random_data(shape)
    grad_out = get_random_data(shape)

    if not torch.cuda.is_available():
        pytest.fail("CUDA not available, but required for softmax backward test")

    x = mytensor.Tensor(data, requires_grad=True, device="cuda")
    out_my = mytensor.softmax(x)
    out_my.backward(grad_out)

    x_torch = torch.tensor(data, requires_grad=True)
    out_torch = F.softmax(x_torch, dim=1)
    out_torch.backward(torch.tensor(grad_out))

    check_tensors(x.grad, x_torch.grad, atol=1e-3, rtol=1e-3)

def test_max_pool2d_backward():
    N, C, H, W = 2, 3, 4, 4
    pool_size = 2
    stride = 2
    input_data = get_random_data((N, C, H, W))

    if not torch.cuda.is_available():
        pytest.fail("CUDA not available, but required for max_pool2d backward test")

    x = mytensor.Tensor(input_data, requires_grad=True, device="cuda")
    out_my = mytensor.max_pool2d(x, pool_size, stride=stride)
    grad_out = get_random_data(out_my.shape)
    out_my.backward(grad_out)

    x_torch = torch.tensor(input_data, requires_grad=True)
    out_torch = F.max_pool2d(x_torch, kernel_size=pool_size, stride=stride)
    out_torch.backward(torch.tensor(grad_out))

    check_tensors(x.grad, x_torch.grad)

def test_cross_entropy_loss_backward():
    batch_size = 6
    num_classes = 7
    logits_data = get_random_data((batch_size, num_classes))
    labels_data = np.random.randint(0, num_classes, size=(batch_size,)).astype(np.int32)

    if not torch.cuda.is_available():
        pytest.fail("CUDA not available, but required for cross_entropy_loss backward test")

    logits = mytensor.Tensor(logits_data, requires_grad=True, device="cuda")
    loss_my = mytensor.cross_entropy_loss(logits, labels_data)
    loss_my.backward()

    logits_torch = torch.tensor(logits_data, requires_grad=True)
    labels_torch = torch.tensor(labels_data, dtype=torch.long)
    loss_torch = F.cross_entropy(logits_torch, labels_torch)
    loss_torch.backward()

    check_tensors(logits.grad, logits_torch.grad, atol=1e-3, rtol=1e-3)

if __name__ == "__main__":
    pytest.main([__file__])
