import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR

# Define the total number of iterations (90,000)
iterations = 90000

# Define the initial learning rate
initial_lr = 0.01

# Define the minimum learning rate
eta_min = 1e-4

# Power for PolyLR
power = 0.9

# Create an optimizer (dummy optimizer as a placeholder)
optimizer = SGD([torch.tensor([0.0])], lr=initial_lr)

# Define the PolyLR scheduler
def poly_lr_scheduler(epoch, max_epochs=iterations, power=0.9):
    return (1 - epoch / max_epochs) ** power

poly_lr = LambdaLR(optimizer, lr_lambda=poly_lr_scheduler)

# Track the learning rate over iterations
poly_lr_lrs = []

# Simulate the learning rate changes for each iteration
for iteration in range(iterations):
    poly_lr.step()
    current_lr = eta_min + (initial_lr - eta_min) * (1 - iteration / iterations) ** power
    poly_lr_lrs.append(current_lr)

# Plotting the result
plt.figure(figsize=(10, 6))
plt.plot(poly_lr_lrs, label='PolyLR', color='g')

plt.title('PolyLR Learning Rate Decay (90k Iterations)')
plt.xlabel('Iterations')
plt.ylabel('Learning Rate')
plt.legend()
plt.grid(True)
plt.show()
