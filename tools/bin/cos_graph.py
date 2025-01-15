import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR

# Define the total number of iterations (90,000)
iterations = 90000

# Define the initial learning rate
initial_lr = 0.1

# Create an optimizer (dummy optimizer as a placeholder)
optimizer = SGD([torch.tensor([0.0])], lr=initial_lr)

# CosineAnnealingLR Scheduler with T_max=90,000
cosine_lr = CosineAnnealingLR(optimizer, T_max=iterations, eta_min=1e-6)

# Track the learning rate over iterations
cosine_lr_lrs = []

# Simulate the learning rate changes for each iteration
for iteration in range(iterations):
    cosine_lr.step()
    cosine_lr_lrs.append(optimizer.param_groups[0]['lr'])

# Plotting the result
plt.figure(figsize=(10, 6))
plt.plot(cosine_lr_lrs, label='CosineAnnealingLR', color='r')

plt.title('CosineAnnealingLR Learning Rate Decay (90k Iterations)')
plt.xlabel('Iterations')
plt.ylabel('Learning Rate')
plt.legend()
plt.grid(True)
plt.show()



