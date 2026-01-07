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