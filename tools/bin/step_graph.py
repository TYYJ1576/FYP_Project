import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR

# Define the total number of iterations (90,000)
iterations = 90000

# Define the initial learning rate
initial_lr = 0.01

# Create an optimizer (dummy optimizer as a placeholder)
optimizer = SGD([torch.tensor([0.0])], lr=initial_lr)

# MultiStepLR Scheduler with milestones at 30k and 60k, gamma = 0.1
scheduler = MultiStepLR(optimizer, milestones=[30000, 60000], gamma=0.1)

# Track the learning rate over iterations
multi_step_lr_lrs = []

# Simulate the learning rate changes for each iteration
for iteration in range(iterations):
    scheduler.step()
    multi_step_lr_lrs.append(optimizer.param_groups[0]['lr'])

# Plotting the result
plt.figure(figsize=(10, 6))
plt.plot(multi_step_lr_lrs, label='MultiStepLR', color='b')

plt.title('MultiStepLR Learning Rate Decay (90k Iterations)')
plt.xlabel('Iterations')
plt.ylabel('Learning Rate')
plt.legend()
plt.grid(True)
plt.show()

