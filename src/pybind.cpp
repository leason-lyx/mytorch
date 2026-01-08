#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>

#include "conv.cuh"
#include "functional.cuh"
#include "linear.cuh"
#include "pooling.cuh"
#include "tensor.cuh"

namespace py = pybind11;

PYBIND11_MODULE(_core, m) {
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
        .def("reshape", &Tensor::reshape)
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
    m.def("relu_backward", &relu_backward, "ReLU backward");
    m.def("sigmoid", &sigmoid, "Sigmoid activation");
    m.def("sigmoid_backward", &sigmoid_backward, "Sigmoid backward");
    m.def("softmax", &softmax, "Softmax activation");
    m.def("cross_entropy_loss", &cross_entropy_loss, "Cross Entropy Loss");
    m.def("cross_entropy_loss_backward", &cross_entropy_loss_backward,
          "Cross Entropy Loss backward");

    // Layers
    m.def("linear", &linear, "Linear layer");
    m.def("linear_backward", &linear_backward, "Linear backward");
    m.def("conv2d", &conv2d, "Conv2d layer", py::arg("input"),
          py::arg("weights"), py::arg("bias"), py::arg("stride_height") = 1,
          py::arg("stride_width") = 1, py::arg("padding_height") = 0,
          py::arg("padding_width") = 0);
    m.def("conv2d_backward",
          [](const Tensor &input, const Tensor &output, const Tensor &weights,
             const Tensor &bias, int stride_height, int stride_width,
             int padding_height, int padding_width, const Tensor &grad_output) {
              Tensor grad_input(input.shape(), input.device());
              Tensor grad_weights(weights.shape(), weights.device());
              Tensor grad_bias(bias.shape(), bias.device());
              conv2d_backward(input, output, weights, bias, stride_height,
                              stride_width, padding_height, padding_width,
                              grad_output, grad_input, grad_weights, grad_bias);
              return py::make_tuple(grad_input, grad_weights, grad_bias);
          },
          "Conv2d backward", py::arg("input"), py::arg("output"),
          py::arg("weights"), py::arg("bias"), py::arg("stride_height") = 1,
          py::arg("stride_width") = 1, py::arg("padding_height") = 0,
          py::arg("padding_width") = 0, py::arg("grad_output"));
    m.def("max_pool2d", &max_pool2d, "MaxPool2d layer", py::arg("input"),
          py::arg("pool_height"), py::arg("pool_width"),
          py::arg("stride_height"), py::arg("stride_width"));
    m.def("max_pool2d_backward",
          [](const Tensor &input, const Tensor &output, std::size_t pool_height,
             std::size_t pool_width, std::size_t stride_height,
             std::size_t stride_width,
             const Tensor &grad_output) {
              Tensor grad_input(input.shape(), input.device());
              return max_pool2d_backward(input, output, pool_height, pool_width,
                                         stride_height, stride_width,
                                         grad_output, grad_input);
          },
          "MaxPool2d backward", py::arg("input"), py::arg("output"),
          py::arg("pool_height"), py::arg("pool_width"),
          py::arg("stride_height"), py::arg("stride_width"),
          py::arg("grad_output"));
}
