# 第四次作业报告

在本次作业中，我使用了uv管理python依赖，使用pytest作为单元测试框架。这样可以使得复现极为容易。



## 实现方式



### pybind绑定

使用pybind.cpp和setup.py进行绑定。



```c++
//src/pybind.cpp

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "conv.cuh"
#include "functional.cuh"
#include "linear.cuh"
#include "pooling.cuh"
#include "tensor.cuh"

namespace py = pybind11;

PYBIND11_MODULE(mytensor, m) {
    m.doc() = "My Tensor Library";

    py::class_<Device>(m, "Device")
        .def(py::init([](std::string type) {
            if (type == "cpu") return Device::cpu();
            if (type == "cuda") return Device::cuda();
            throw std::invalid_argument("Invalid device type: " + type);
        }))
        .def("is_cpu", &Device::is_cpu)
        .def("is_cuda", &Device::is_cuda)
        .def("__repr__", &Device::str);

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<std::vector<size_t>, Device>(), py::arg("shape"),
             py::arg("device") = Device::cpu())
        .def(py::init([](py::array_t<float> b, Device dev) {
            py::buffer_info info = b.request();
            std::vector<size_t> shape;
            for (auto s : info.shape)
                shape.push_back(s);

            Tensor t(shape, Device::cpu());
            std::memcpy(t.data(), info.ptr, info.size * sizeof(float));

            if (dev.is_cuda()) {
                return t.gpu();
            }
            return t;
        }), py::arg("array"), py::arg("device") = Device::cpu())
        .def("shape", &Tensor::shape)
        .def("cpu", &Tensor::cpu)
        .def("gpu", &Tensor::gpu)
        .def("numpy", [](const Tensor &t) {
            Tensor cpu_t = t.cpu();
            std::vector<size_t> shape = cpu_t.shape();
            py::array_t<float> result(shape);
            py::buffer_info buf = result.request();
            std::memcpy(buf.ptr, cpu_t.data(), cpu_t.numel() * sizeof(float));
            return result;
        })
        .def("__repr__", [](const Tensor &t) {
            std::string s = "Tensor(shape=[";
            for (size_t i = 0; i < t.shape().size(); ++i) {
                s += std::to_string(t.shape()[i]);
                if (i < t.shape().size() - 1) s += ", ";
            }
            s += "], device=" + t.device().str() + ")";
            return s;
        });

    // Functional
    m.def("relu", &relu, "ReLU activation");
    m.def("sigmoid", &sigmoid, "Sigmoid activation");
    m.def("softmax", &softmax, "Softmax activation");
    m.def("cross_entropy_loss", &cross_entropy_loss, "Cross Entropy Loss");

    // Layers
    m.def("linear", &linear, "Linear layer");
    m.def("conv2d", &conv2d, "Conv2d layer", py::arg("input"),
          py::arg("weights"), py::arg("bias"), py::arg("stride_height") = 1,
          py::arg("stride_width") = 1, py::arg("padding_height") = 0,
          py::arg("padding_width") = 0);
    m.def("max_pool2d", &max_pool2d, "MaxPool2d layer", py::arg("input"),
          py::arg("pool_height"), py::arg("pool_width"),
          py::arg("stride_height"), py::arg("stride_width"));
}
```

```python
#setup.py

import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

__version__ = '0.1.0'
sources = ["src/pybind.cpp","src/tensor.cu","src/conv.cu","src/pooling.cu","src/linear.cu","src/functional.cu"]

setup(
    name="mytensor",
    version=__version__,
    # author="LiYixin",
    author_email="2200013104@stu.pku.edu.cn",
    packages=find_packages(exclude=("tests",)),
    zip_safe=False,
    install_requires=[
        "torch",],
    ext_modules=[
        CUDAExtension(name="mytensor",
                      sources=sources,
                      libraries=["cublas"], 
                      )],
    cmdclass={"build_ext": BuildExtension},
)
```

使用pyproject.toml来配置构建系统和依赖。在 `[build-system]` 中声明了编译所需的 `torch` 和 `pybind11`，确保 `pip install .` 时能自动建立正确的编译环境。通过 `[tool.uv]` 指定了 PyTorch 的 CUDA 13.0 索引源，确保在 Windows/Linux 平台上均能下载到支持 GPU 加速的正确版本。

```toml
[build-system]
requires = ["setuptools", "wheel", "torch>=2.7.0", "pybind11>=3.0.1","numpy"]
build-backend = "setuptools.build_meta"

[project]
name = "hw4"
version = "0.1.0"
description = "Add your description here"
readme = "AGENTS.md"
requires-python = ">=3.13"
authors = [
    { name = "Li Yixin", email = "2200013104@stu.pku.edu.cn" }
]
dependencies = [
    "pip>=25.3",
    "pybind11>=3.0.1",
    "pytest>=9.0.1",
    "torch>=2.7.0",
    "torchvision>=0.22.0",
    "web3>=7.14.0",
]



[tool.uv.sources]
torch = [
  { index = "pytorch-cu130", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]
torchvision = [
  { index = "pytorch-cu130", marker = "sys_platform == 'linux' or sys_platform == 'win32'" },
]

[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

```

### 测试MNIST

在MNIST_test.py中，使用torch下载并转换MNIST数据，并测试转成mytensor.Tensor。

```python
# MNIST_test.py
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import mytensor

def main():
    print("Loading MNIST dataset using PyTorch...")
    
    # 定义转换：转换为 Tensor 并归一化
    # 这里使用 PyTorch 的 transforms 将图片转换为 PyTorch Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 下载并加载训练集
    # root='./data' 指定数据集下载/存放的目录
    train_dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )

    # 使用 DataLoader 加载数据
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True
    )

    # 获取一批数据
    data_iter = iter(train_loader)
    images, labels = next(data_iter)

    print(f"PyTorch images shape: {images.shape}")
    print(f"PyTorch labels shape: {labels.shape}")

    # 转换为 numpy 数组
    # mytensor 期望输入为 float32 类型的 numpy 数组
    images_np = images.numpy().astype(np.float32)
    
    print("\nConverting to mytensor.Tensor...")
    
    # 1. 转换为 CPU 上的 mytensor.Tensor
    # 使用 mytensor.Device("cpu") 指定设备
    try:
        device_cpu = mytensor.Device("cpu")
        my_images_cpu = mytensor.Tensor(images_np, device_cpu)
        print(f"Successfully created MyTensor on CPU: {my_images_cpu}")
        
        # 验证数据转换是否正确：转回 numpy 进行比较
        images_back = my_images_cpu.numpy()
        diff = np.abs(images_np - images_back).max()
        print(f"Max difference between original and converted back: {diff}")
        
    except Exception as e:
        print(f"Error creating CPU tensor: {e}")

    # 2. 尝试转换为 GPU 上的 mytensor.Tensor
    # 使用 mytensor.Device("cuda") 指定设备
    try:
        if torch.cuda.is_available():
            print("\nCUDA is available in PyTorch, attempting to create GPU Tensor in mytensor...")
            device_cuda = mytensor.Device("cuda")
            my_images_gpu = mytensor.Tensor(images_np, device_cuda)
            print(f"Successfully created MyTensor on GPU: {my_images_gpu}")
        else:
            print("\nCUDA not available in PyTorch, skipping GPU tensor creation test.")
    except Exception as e:
        print(f"Error creating GPU tensor (make sure CUDA is supported in your C++ build): {e}")

if __name__ == "__main__":
    main()

```

### 算子单元测试



使用pytest框架构建测试。

```python
# tests/test_ops.py

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
    # conv2d(input, weights, bias, stride_h, stride_w, padding_h, padding_w)
    out_my = mytensor.conv2d(t_in, t_w, t_b, stride, stride, padding, padding)
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
    # max_pool2d(input, pool_height, pool_width, stride_height, stride_width)
    out_my = mytensor.max_pool2d(t_in, pool_size, pool_size, stride, stride).cpu()
    
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

if __name__ == "__main__":
    pytest.main([__file__])

```







## 代码运行方式

```powershell
uv sync 				#下载依赖
uv pip install -e . 	#编译自定义的包
uv run MNIST_test.py	#测试读取MNIST数据
uv run pytest			#运行7个算子的单元测试
```



## 单元测试结果

通过全部测试

```powershell
PS D:\code\HW4> uv run pytest -v
================================================= test session starts =================================================
platform win32 -- Python 3.13.3, pytest-9.0.1, pluggy-1.6.0 -- D:\code\HW4\.venv\Scripts\python.exe
cachedir: .pytest_cache
rootdir: D:\code\HW4
configfile: pyproject.toml
collected 7 items

tests/test_ops.py::test_sigmoid PASSED                                                                           [ 14%]
tests/test_ops.py::test_relu PASSED                                                                              [ 28%]
tests/test_ops.py::test_linear PASSED                                                                            [ 42%]
tests/test_ops.py::test_conv2d PASSED                                                                            [ 57%]
tests/test_ops.py::test_max_pool2d PASSED                                                                        [ 71%]
tests/test_ops.py::test_softmax PASSED                                                                           [ 85%]
tests/test_ops.py::test_cross_entropy_loss PASSED                                                                [100%]

================================================== 7 passed in 3.24s ==================================================
```

