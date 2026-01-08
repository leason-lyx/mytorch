# Task3报告

在本task中，我复用了HW4写的算子库，以及HW6的自动微分框架。
我使用了uv管理python依赖，使用pytest作为单元测试框架。这样可以使得复现较为容易。

## 实现方式

在`src/`中实现C++/CUDA实现的算子。在mytensor.py中实现自动微分框架。
使用pybind封装成python库。

### 算子
算子实现主要在 `src/functional.cu`、`src/linear.cu`、`src/conv.cu`、`src/pooling.cu` 等文件中，`src/pybind.cpp` 使用 pybind11 封装为 python 模块，`mytensor.py` 再把底层接口包装成更易用的 python API。

### 自动微分
自动微分框架在 `mytensor.py` 中，包括 `Context`、`Function`、`Tensor`、`Module` 等结构，`Tensor.backward()` 会构建拓扑排序并调用各算子的 `backward` 计算梯度。

### 算子单元测试
测试在 `tests/test_ops.py`，使用 pytest 对前向/反向结果与 PyTorch 进行对比，覆盖主要算子和自动微分接口。

### cifar10任务
训练脚本在 `train_cifar10.py`，使用 mytensor 的自动微分进行训练，并包含数据增强和超参选项。
对比脚本在 `train_cifar10_pytorch.py`，保持相同网络结构、优化器与超参，直接用 PyTorch 实现便于性能对比。

## 代码运行方式

```powershell
uv sync 				    #下载依赖
uv pip install -e . 	    #编译自定义的包
uv run pytest			    #运行单元测试
uv run train_cifar10.py     #进行训练
```



## 运行结果

运行环境：windows11，1x 4080 16g

使用我自己写的库训练cifar10任务
使用Adam优化器，lr为0.001
```
uv run .\train_cifar10.py --optimizer adam --lr 0.001
Epoch 1: train_loss=1.6519, train_acc=0.4183, test_loss=1.2181, test_acc=0.5608, train_time=112.16s, eval_time=7.80s, epoch_time=119.96s
Epoch 2: train_loss=1.1815, train_acc=0.5773, test_loss=1.0520, test_acc=0.6247, train_time=101.24s, eval_time=4.48s, epoch_time=105.72s
Epoch 3: train_loss=0.9834, train_acc=0.6528, test_loss=0.8221, test_acc=0.7189, train_time=138.12s, eval_time=7.34s, epoch_time=145.46s
Epoch 4: train_loss=0.8565, train_acc=0.6995, test_loss=0.7525, test_acc=0.7353, train_time=96.10s, eval_time=4.31s, epoch_time=100.40s
Epoch 5: train_loss=0.7750, train_acc=0.7298, test_loss=0.7092, test_acc=0.7571, train_time=76.50s, eval_time=4.10s, epoch_time=80.61s
Epoch 6: train_loss=0.7104, train_acc=0.7570, test_loss=0.6775, test_acc=0.7681, train_time=75.64s, eval_time=4.11s, epoch_time=79.75s
Epoch 7: train_loss=0.6709, train_acc=0.7672, test_loss=0.6547, test_acc=0.7796, train_time=78.97s, eval_time=4.10s, epoch_time=83.07s
Epoch 8: train_loss=0.6324, train_acc=0.7799, test_loss=0.7904, test_acc=0.7412, train_time=78.13s, eval_time=3.70s, epoch_time=81.83s
Epoch 9: train_loss=0.6081, train_acc=0.7876, test_loss=0.5866, test_acc=0.8031, train_time=82.47s, eval_time=4.14s, epoch_time=86.61s
Epoch 10: train_loss=0.5833, train_acc=0.7981, test_loss=0.5667, test_acc=0.8010, train_time=82.61s, eval_time=4.19s, epoch_time=86.81s
Total training time: 970.21s
```

为了对比我的实现与pytorch库实现的性能差距，我写了train_cifar10_pytorch.py，比较在相同的网络结构、优化器、超参下两种实现的区别。

```
 uv run .\train_cifar10_pytorch.py --optimizer adam --lr 0.001
Epoch 1: train_loss=1.6381, train_acc=0.3897, test_loss=1.2458, test_acc=0.5406, train_time=9.76s, eval_time=1.13s, epoch_time=10.89s
Epoch 2: train_loss=1.2009, train_acc=0.5611, test_loss=1.0316, test_acc=0.6356, train_time=10.73s, eval_time=1.18s, epoch_time=11.91s
Epoch 3: train_loss=0.9826, train_acc=0.6447, test_loss=0.8967, test_acc=0.6846, train_time=11.91s, eval_time=1.13s, epoch_time=13.05s
Epoch 4: train_loss=0.8586, train_acc=0.6956, test_loss=0.7953, test_acc=0.7193, train_time=11.70s, eval_time=1.11s, epoch_time=12.81s
Epoch 5: train_loss=0.7760, train_acc=0.7287, test_loss=0.6832, test_acc=0.7652, train_time=11.66s, eval_time=1.13s, epoch_time=12.79s
Epoch 6: train_loss=0.7264, train_acc=0.7465, test_loss=0.7154, test_acc=0.7520, train_time=11.62s, eval_time=1.12s, epoch_time=12.73s
Epoch 7: train_loss=0.6818, train_acc=0.7640, test_loss=0.7525, test_acc=0.7477, train_time=11.61s, eval_time=1.19s, epoch_time=12.80s
Epoch 8: train_loss=0.6499, train_acc=0.7726, test_loss=0.6481, test_acc=0.7754, train_time=11.82s, eval_time=1.13s, epoch_time=12.95s
Epoch 9: train_loss=0.6262, train_acc=0.7812, test_loss=0.5887, test_acc=0.7945, train_time=11.16s, eval_time=1.05s, epoch_time=12.20s
Epoch 10: train_loss=0.6023, train_acc=0.7902, test_loss=0.5984, test_acc=0.7947, train_time=10.63s, eval_time=1.16s, epoch_time=11.80s
Total training time: 123.93s
```

## 分析

我实现的深度学习框架可以正确完成cifar-10任务。但是性能跟pytorch实现还是有着巨大差距。
