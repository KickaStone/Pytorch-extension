import torch
import torch.nn as nn
import MyReLU
from torch.autograd import gradcheck


class MyReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return MyReLU.forward(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = MyReLU.backward(grad_output, input)
        return grad_input
    
class MyReLUModel(nn.Module):
    def __init__(self):
        super(MyReLUModel, self).__init__()
        self.myrelu = MyReLUFunction()

    def forward(self, input):
        return self.myrelu(input)
    

test_input = torch.randn(20, 20, dtype=torch.double, requires_grad=True, device='cuda')
test = gradcheck(MyReLUFunction.apply, test_input)
print(test)