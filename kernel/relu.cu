/**
 * @file relu.cu
 * @brief cuda implementation of relu
 * @date 2023/08/28
*/

#include <cuda_runtime.h>
#include <torch/extension.h>

template <typename scalar_t>
/// @brief CUDA Kernel: forward ReLU
/// @param input the input features data pointer
/// @param output the output features data pointer
/// @param size the size of input and output, the output's size must be the same as input
/// @return 
__global__ void relu_forward_cuda_kernel(
    const scalar_t *input,
    scalar_t *output,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = input[idx] > 0 ? input[idx] : 0;
    }
}


template <typename scalar_t>
/// @brief CUDA Kernel: backward ReLU
/// @param grad_output the grad output features data pointer
/// @param input the input features data pointer, used to judge whether the input is greater than 0
/// @param grad_input the grad input features data pointer
/// @param size the size of input and output
/// @return 
__global__ void relu_backward_cuda_kernel(
    const scalar_t *grad_output,
    const scalar_t *input,
    scalar_t *grad_input,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        grad_input[idx] = input[idx] > 0 ? grad_output[idx] : 0;
    }
}

// ==================== launch kernel ====================
torch::Tensor relu_forward_cuda(
    torch::Tensor input)
{
    auto output = torch::zeros_like(input);
    int size = input.numel();
    const int threads = 1024;
    const int blocks = (size + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_forward_cuda", ([&] {
        relu_forward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(),
            size);
    }));
    return output;
}

torch::Tensor relu_backward_cuda(
    torch::Tensor grad_output,
    torch::Tensor input)
{
    auto grad_input = torch::zeros_like(input);
    int size = input.numel();
    const int threads = 1024;
    const int blocks = (size + threads - 1) / threads;
    AT_DISPATCH_FLOATING_TYPES(input.type(), "relu_backward_cuda", ([&] {
        relu_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            grad_input.data_ptr<scalar_t>(),
            size);
    }));
    return grad_input;
}

