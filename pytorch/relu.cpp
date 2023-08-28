/**
 * relu.cpp
 * Used to register the relu function to the torch library
*/
#include <torch/extension.h>
#include "../include/utils.h"

torch::Tensor relu_forward(torch::Tensor input) {
    CHECK_INPUT(input);
    return relu_forward_cuda(input);
}

torch::Tensor relu_backward(torch::Tensor grad_output, torch::Tensor input) {
    CHECK_INPUT(grad_output);
    CHECK_INPUT(input);
    return relu_backward_cuda(grad_output, input);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &relu_forward, "relu forward (CUDA)", py::arg("input"));
    m.def("backward", &relu_backward, "relu backward (CUDA)", py::arg("grad_output"), py::arg("input"));
}

TORCH_LIBRARY(TORCH_MyReLU, m) {
    m.def("forward", &relu_forward);
    m.def("backward", &relu_backward);
}