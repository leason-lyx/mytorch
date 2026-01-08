import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
# import mytensor

def main():
    print("Loading CIFAR-10 dataset using PyTorch...")
    
    # 定义转换：转换为 Tensor 并归一化
    # 使用 CIFAR-10 的标准均值和方差进行归一化
    # mean = (0.4914, 0.4822, 0.4465), std = (0.2023, 0.1994, 0.2010)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # 下载并加载 CIFAR-10 训练集
    # root='./data' 指定数据集下载/存放的目录
    train_dataset = torchvision.datasets.CIFAR10(
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

    print(f"PyTorch images shape (CIFAR-10): {images.shape}")
    print(f"PyTorch labels shape: {labels.shape}")

    # 转换为 numpy 数组
    # mytensor 期望输入为 float32 类型的 numpy 数组
    images_np = images.numpy().astype(np.float32)
    
    # print("\nConverting to mytensor.Tensor...")
    
    # # 1. 转换为 CPU 上的 mytensor.Tensor
    # # 使用 mytensor.Device("cpu") 指定设备
    # try:
    #     device_cpu = mytensor.Device("cpu")
    #     my_images_cpu = mytensor.Tensor(images_np, device_cpu)
    #     print(f"Successfully created MyTensor on CPU: {my_images_cpu}")
        
    #     # 验证数据转换是否正确：转回 numpy 进行比较
    #     images_back = my_images_cpu.numpy()
    #     diff = np.abs(images_np - images_back).max()
    #     print(f"Max difference between original and converted back: {diff}")
        
    # except Exception as e:
    #     print(f"Error creating CPU tensor: {e}")

    # 2. 尝试转换为 GPU 上的 mytensor.Tensor
    # 使用 mytensor.Device("cuda") 指定设备
    # try:
    #     if torch.cuda.is_available():
    #         print("\nCUDA is available in PyTorch, attempting to create GPU Tensor in mytensor...")
    #         device_cuda = mytensor.Device("cuda")
    #         my_images_gpu = mytensor.Tensor(images_np, device_cuda)
    #         print(f"Successfully created MyTensor on GPU: {my_images_gpu}")
    #     else:
    #         print("\nCUDA not available in PyTorch, skipping GPU tensor creation test.")
    # except Exception as e:
    #     print(f"Error creating GPU tensor (make sure CUDA is supported in your C++ build): {e}")

if __name__ == "__main__":
    main()
