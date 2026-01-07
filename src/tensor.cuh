#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

struct Device
{
    enum class Type
    {
        CPU,
        CUDA
    };

    Type type{Type::CPU};

    static Device cpu();
    static Device cuda();

    bool is_cpu() const;
    bool is_cuda() const;
    std::string str() const;
};

void copy_bytes(void *dst, const Device &dst_dev,
                const void *src, const Device &src_dev,
                std::size_t nbytes);

struct Storage
{
    std::shared_ptr<float> host_data{};
    std::shared_ptr<float> device_data{};
    std::size_t nbytes{0};
    Device device{Device::cpu()};

    static Storage empty(std::size_t n, Device dev);

    float *raw_ptr();
    const float *raw_ptr() const;
};

class Tensor
{
public:
    using index_t = std::size_t;

    Tensor() = default;
    ~Tensor() = default;

    explicit Tensor(std::vector<index_t> shape, Device dev = Device::cpu());

    const std::vector<index_t> &shape() const;
    const std::vector<index_t> &strides() const;
    index_t ndim() const;
    index_t numel() const;
    Device device() const;

    float *data();
    const float *data() const;

    void from_vector(const std::vector<float> &host);
    std::vector<float> to_vector() const;

    Tensor cpu() const;
    Tensor gpu() const;

    bool is_contiguous{true};

private:
    std::vector<index_t> shape_{};
    std::vector<index_t> strides_{};
    index_t offset_{0};
    Storage storage_{};
    Device device_{Device::cpu()};

    Tensor transfer_to(const Device &target) const;
    void compute_default_strides_();
};
