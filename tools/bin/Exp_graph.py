import torch
import numpy as np
import matplotlib.pyplot as plt

# Parameters
initial_lr = 0.1  # Initial learning rate (adjust this based on your setup)
gamma = 0.9999
max_iterations = 90000

# Calculate the learning rate for each iteration
iterations = np.arange(0, max_iterations + 1)
learning_rates = initial_lr * gamma**iterations

# Plotting the graph
plt.figure(figsize=(8, 6))
plt.plot(iterations, learning_rates, label="Exponential Decay", color='b')
plt.title("Exponential Learning Rate Decay (ExponentialLR)", fontsize=14)
plt.xlabel("Iterations", fontsize=12)
plt.ylabel("Learning Rate", fontsize=12)
plt.grid(True)
plt.legend()
plt.tight_layout()

# Display the plot
plt.show()
