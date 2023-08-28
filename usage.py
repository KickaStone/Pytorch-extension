import torch
from ctypes import *

test_input = torch.randn(4, 4, dtype=torch.double, device='cuda')

# using torch load_library
torch.ops.load_library("build/libMyReLU.so")
test_output = torch.ops.TORCH_MyReLU.forward(test_input)
print(test_output)

# using setup script
import MyReLU
test_output = MyReLU.forward(test_input)
print(test_output)

# using jit
from torch.utils.cpp_extension import load
MyReLU = load(name="MyReLU",
                    extra_include_paths=["include"],
                    sources=["pytorch/relu.cpp", "kernel/relu.cu"],
                    verbose=True)
help(MyReLU)
test_output = MyReLU.forward(test_input)
print(test_output)