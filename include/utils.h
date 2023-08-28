/**
 * @file utils.h
 * @brief Some utility functions for PyTorch C++ Extension
 * @date 2023/08/28
*/
#ifndef _UTILS_H_
#define _UTILS_H_

#include <torch/extension.h>
#include <vector>

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor relu_forward_cuda(torch::Tensor input);
torch::Tensor relu_backward_cuda(torch::Tensor grad_output, torch::Tensor input);

#endif // _UTILS_H_