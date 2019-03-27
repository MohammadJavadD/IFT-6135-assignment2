import matplotlib.pyplot as plt
import numpy as np
import os

grads_RNN = np.load('grads_RNN.npy')[()]
grads_GRU = np.load('grads_GRU.npy')[()]
grads_RNN = (grads_RNN - np.min(grads_RNN)) / (np.max(grads_RNN) - np.min(grads_RNN))
grads_GRU = (grads_GRU - np.min(grads_GRU)) / (np.max(grads_GRU) - np.min(grads_GRU))

plt.figure(figsize=(12,10))
plt.plot(grads_GRU,label='GRU')
plt.plot(grads_RNN,label='RNN')
plt.xlabel('Timestep')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norm with respect to Timestep')
plt.legend()
plt.savefig('figures/grad_timestep.png')