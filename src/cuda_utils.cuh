#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>

// CUDA error helper
#ifndef CUDA_CHECK
#define CUDA_CHECK(expr)                                                                           \
    do {                                                                                           \
        cudaError_t _err = (expr);                                                                 \
        if (_err != cudaSuccess) {                                                                 \
            throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_err));      \
        }                                                                                          \
    } while (0)
#endif

// CUDA 内存管理的智能指针 deleter
struct CudaDeleter {
    void operator()(void* ptr) const {
        if (ptr) {
            cudaFree(ptr);
        }
    }
};

// CUDA 智能指针类型别名
using CudaUniquePtr = std::unique_ptr<float, CudaDeleter>;

// 辅助函数：创建 CUDA 智能指针
inline CudaUniquePtr make_cuda_unique(size_t size) {
    float* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, size * sizeof(float)));
    return CudaUniquePtr(ptr);
}
