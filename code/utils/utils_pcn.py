import numpy as np
import matplotlib.pyplot as plt 

import torch 
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

import MotionClouds as mc 
from tqdm import tqdm

# ---------------------------------------------------------------
# ACTIVATION FUNCTIONS
# ---------------------------------------------------------------
# Hyperbolic tangents ----------------------------------------------
# Define the hyperbolic tangent activation function
def tanh(xs):
    return torch.tanh(xs)

# Define the derivative of the hyperbolic tangent activation function
def tanh_deriv(xs):
    return 1.0 - torch.tanh(xs) ** 2.0


# Linear ----------------------------------------------------------
# Define the linear activation function (identity function)
def linear(x):
    return x

# Define the derivative of the linear activation function (always 1)
def linear_deriv(x):
    return set_tensor(torch.ones((1,)))


# Rectified Linear Unit --------------------------------------------
# Define the ReLU activation function
def relu(xs):
    return torch.clamp(xs,min=0)

# Define the derivative of the ReLU activation function
def relu_deriv(xs):
    rel = relu(xs)
    rel[rel>0] = 1
    return rel 


# Sigmoid ---------------------------------------------------------
# Define the sigmoid activation function
def sigmoid(xs):
    return F.sigmoid(xs)

# Define the derivative of the sigmoid activation function
def sigmoid_deriv(xs):
    return F.sigmoid(xs) * (torch.ones_like(xs) - F.sigmoid(xs))



# ---------------------------------------------------------
# TORCH UTILS 
# ---------------------------------------------------------------
# Define the accuracy function to compute the accuracy of a model's predictions
def accuracy(out, L):
    B, l = out.shape  # B is the batch size, l is the number of output classes
    total = 0  # Initialize the count of correct predictions

    # Iterate through the batch
    for i in range(B):
        # Compare the index of the maximum value in the output with the index of the maximum value in the labels
        if torch.argmax(out[i, :]) == torch.argmax(L[i, :]):
            total += 1  # Increment the count of correct predictions if they match

    return total / B  # Return the accuracy as the ratio of correct predictions to the total number of samples

# Define the onehot function to convert a list of class indices to one-hot encoded tensors
def onehot(x, device, n_classes=10):
    xs = x.clone() - 1  # Clone the input tensor and subtract 1 from each element (assuming class indices start from 1)
    z = torch.zeros([len(xs), n_classes])  # Initialize a zero tensor of shape (number of samples, number of classes)

    # Iterate through the samples
    for i in range(len(xs)):
        z[i, xs[i]] = 1  # Set the value at the corresponding class index to 1 (one-hot encoding)

    return z.float().to(device) # Convert the tensor to float and move it to the specified device (CPU or GPU)

# Define the set_tensor function to convert a tensor to a float tensor and move it to the specified device (CPU or GPU)
def set_tensor(xs, device):
    return xs.float().to(device)



# ---------------------------------------------------------------
# NUMPY UTILS 
# ---------------------------------------------------------------
def norm_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def neg_norm_data(data):
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data)) # Normalizing between 0 and 1
    normalized_data = (normalized_data * 2) - 1 # Shifting the range to -1 and 1
    return normalized_data