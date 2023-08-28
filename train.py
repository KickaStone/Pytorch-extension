import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# jit
from torch.utils.cpp_extension import load
MyReLU = load(name="MyReLU",
                    extra_include_paths=["include"],
                    sources=["pytorch/relu.cpp", "kernel/relu.cu"],
                    verbose=True)

# Define Functions
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
    
# gradcheck
test_input = torch.randn(20, 20, dtype=torch.double, requires_grad=True, device='cuda')
test = torch.autograd.gradcheck(MyReLUFunction.apply, test_input)
print("Gradcheck Passed" if test else "Gradcheck Failed")
 
# Define DIY Model
class MyReLUModule(nn.Module):
    def __init__(self):
        super(MyReLUModule, self).__init__()
    

    def forward(self, input):
        return MyReLUFunction.apply(input)
    
    
# Define training Model
class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 30),
            # nn.ReLU(),
            MyReLUModule(),
            nn.Linear(30, 10)
        )
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
    
    

# parameters
learning_rate = 0.3
batch_size = 64
epochs = 10

# load data
training_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor())

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor())

# create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# create model
if torch.cuda.is_available():
    device = "cuda"
else:
    EOFError("No GPU found")
    
model = NN().to(device)
print(model)

# optimizer and loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# train model
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        
        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch*len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")
            
def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0,0
    
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
    
print("Done!")