/*
tensor.cu

实现一个用于在CPU和GPU之间管理数据的Tensor类，支持基本的张量操作
*/

#pragma once
#include "tensor.cuh"

#include <cuda_runtime.h>

#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>

// CUDA error helper
#define CUDA_CHECK(expr) do { \
  cudaError_t _e = (expr); \
  if (_e != cudaSuccess) { \
    throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(_e)); \
  } \
} while(0)

Device Device::cpu() { return Device{Type::CPU}; }
Device Device::cuda() { return Device{Type::CUDA}; }

bool Device::is_cpu() const { return type == Type::CPU; }
bool Device::is_cuda() const { return type == Type::CUDA; }
std::string Device::str() const { return is_cpu() ? "cpu" : "cuda"; }

void copy_bytes(void* dst, const Device& dst_dev,
                const void* src, const Device& src_dev,
                std::size_t nbytes) {
  if (nbytes == 0) return;
  if (src_dev.is_cuda() && dst_dev.is_cuda()) {
    CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToDevice));
  } else if (src_dev.is_cpu() && dst_dev.is_cuda()) {
    CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyHostToDevice));
  } else if (src_dev.is_cuda() && dst_dev.is_cpu()) {
    CUDA_CHECK(cudaMemcpy(dst, src, nbytes, cudaMemcpyDeviceToHost));
  } else {
    std::memcpy(dst, src, nbytes);
  }
}

Storage Storage::empty(std::size_t n, Device dev) {
  Storage s;
  s.nbytes = n;
  s.device = dev;
  if (n == 0) return s;

  if (n % sizeof(float) != 0) {
    throw std::runtime_error("Storage::empty: size must be a multiple of sizeof(float)");
  }
  std::size_t n_floats = n / sizeof(float);

  if (dev.is_cpu()) {
    float* p = new float[n_floats];
    s.host_data.reset(p, [](float* q) { delete[] q; });
    s.device_data.reset();
  } else {
    void* raw = nullptr;
    CUDA_CHECK(cudaMalloc(&raw, n));
    s.device_data.reset(static_cast<float*>(raw), [](float* q) { cudaFree(q); });
    s.host_data.reset();
  }

  return s;
}

float* Storage::raw_ptr() {
  return device.is_cuda() ? device_data.get() : host_data.get();
}

const float* Storage::raw_ptr() const {
  return device.is_cuda() ? device_data.get() : host_data.get();
}

Tensor::Tensor(std::vector<index_t> shape, Device dev)
    : shape_(std::move(shape)), device_(dev) {
  compute_default_strides_();
  storage_ = Storage::empty(numel() * sizeof(float), dev);
  is_contiguous = true;
}

const std::vector<Tensor::index_t>& Tensor::shape() const { return shape_; }
const std::vector<Tensor::index_t>& Tensor::strides() const { return strides_; }

Tensor::index_t Tensor::ndim() const { return shape_.size(); }

Tensor::index_t Tensor::numel() const {
  if (shape_.empty()) {
    return index_t{0};
  }
  index_t total = 1;
  for (index_t dim : shape_) {
    total *= dim;
  }
  return total;
}

Device Tensor::device() const { return device_; }

float* Tensor::data() {
  float* base = storage_.raw_ptr();
  if (!base) return nullptr;
  return base + offset_;
}

const float* Tensor::data() const {
  const float* base = storage_.raw_ptr();
  if (!base) return nullptr;
  return base + offset_;
}

void Tensor::from_vector(const std::vector<float>& host) {
  if (host.size() != numel()) {
    throw std::runtime_error("from_vector: data size mismatch");
  }
  copy_bytes(data(), device_, host.data(), Device::cpu(), host.size() * sizeof(float));
}

std::vector<float> Tensor::to_vector() const {
  std::vector<float> host(numel());
  if (numel()) {
    copy_bytes(host.data(), Device::cpu(), data(), device_, numel() * sizeof(float));
  }
  return host;
}

Tensor Tensor::cpu() const { return transfer_to(Device::cpu()); }

Tensor Tensor::gpu() const { return transfer_to(Device::cuda()); }

Tensor Tensor::transfer_to(const Device& target) const {
  if (device_.type == target.type) {
    return *this;
  }

  if (!is_contiguous) {
    throw std::runtime_error("transfer_to: only contiguous tensors are supported");
  }
  if (offset_ != 0) {
    throw std::runtime_error("transfer_to: tensors with non-zero offset are not supported");
  }

  Tensor out;
  out.shape_ = shape_;
  out.strides_ = strides_;
  out.offset_ = offset_;
  out.is_contiguous = is_contiguous;
  out.device_ = target;
  out.storage_ = Storage::empty(storage_.nbytes, target);

  if (numel()) {
    copy_bytes(out.data(), target, data(), device_, numel() * sizeof(float));
  }

  return out;
}

void Tensor::compute_default_strides_() {
  strides_.resize(shape_.size());
  index_t stride = 1;
  for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
    strides_[i] = stride;
    stride *= shape_[static_cast<std::size_t>(i)];
  }
}
