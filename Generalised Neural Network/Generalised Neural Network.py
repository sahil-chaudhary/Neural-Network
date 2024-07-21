import numpy as np
from matplotlib import pyplot as plt
import torch
from keras.datasets import mnist
from keras.utils import to_categorical

# Load and preprocess the MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], -1).T / 255.0
X_test = X_test.reshape(X_test.shape[0], -1).T / 255.0
Y_train = to_categorical(Y_train).T
Y_test = to_categorical(Y_test).T

# Initialize parameters for neural network
def initialize_parameters(layer_dims):
    np.random.seed(1)
    parameters = {}
    for l in range(1, len(layer_dims)):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) * np.sqrt(2 / layer_dims[l - 1])  # He initialization
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

# Activation functions and their derivatives
def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def relu(Z):
    return np.maximum(0, Z)

def tanh(Z):
    return np.tanh(Z)

def softmax(Z):
    expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return expZ / expZ.sum(axis=0, keepdims=True)

def sigmoid_backward(dA, Z):
    sig = sigmoid(Z)
    return dA * sig * (1 - sig)

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def tanh_backward(dA, Z):
    return dA * (1 - np.tanh(Z)**2)

# Mean Squared Error Loss and its derivative
def mean_squared_error(Y_hat, Y):
    m = Y.shape[1]
    return np.sum((Y_hat - Y)**2) / (2 * m)

def mean_squared_error_backward(Y_hat, Y):
    m = Y.shape[1]
    return (Y_hat - Y) / m

# Linear backward step
def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db

# Activation backward
def activate_backward(dA, cache, activation):
    linear_cache, activation_cache = cache
    
    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif activation == "tanh":
        dZ = tanh_backward(dA, activation_cache)
    
    return linear_backward(dZ, linear_cache)

# Full backward propagation
def backward_propagation(AL, Y, caches, activations):
    grads = {}
    L = len(caches)  # number of layers
    m = Y.shape[1]
    Y = Y.reshape(AL.shape)  # after this line, Y is the same shape as AL

    # Initializing the backpropagation
    dAL = mean_squared_error_backward(AL, Y)

    # Lth layer (SOFTMAX -> LINEAR) gradients
    current_cache = caches[L - 1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = activate_backward(dAL, current_cache, activations[L - 1])

    # Loop from l=L-2 to l=0
    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = activate_backward(grads["dA" + str(l + 2)], current_cache, activations[l])
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

# Training loop
def model(X, Y, layer_dims, activations, optimizer, learning_rate=0.01, beta=0.9, beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=1000, print_cost=False):
    costs = []
    parameters = initialize_parameters(layer_dims)
    
    if optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "rmsprop":
        s = initialize_rmsprop(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
    
    for t in range(1, num_epochs + 1):
        # Forward propagation
        A = X
        caches = []
        for l in range(1, len(layer_dims)):
            A_prev = A
            W = parameters['W' + str(l)]
            b = parameters['b' + str(l)]
            A, cache = activate_forward(A_prev, W, b, activations[l-1])
            caches.append(cache)
        
        # Compute loss
        loss = mean_squared_error(A, Y)
        
        # Backward propagation
        grads = backward_propagation(A, Y, caches, activations)
        
        # Update parameters
        if optimizer == "sgd":
            parameters = update_parameters_sgd(parameters, grads, learning_rate)
        elif optimizer == "momentum":
            parameters, v = update_parameters_momentum(parameters, grads, v, beta, learning_rate)
        elif optimizer == "rmsprop":
            parameters, s = update_parameters_rmsprop(parameters, grads, s, beta, learning_rate, epsilon)
        elif optimizer == "adam":
            parameters, v, s = update_parameters_adam(parameters, grads, v, s, t, learning_rate, beta1, beta2, epsilon)
        
        # Print the cost every 100 iterations
        if print_cost and t % 100 == 0:
            print(f"Cost after epoch {t}: {loss}")
            costs.append(loss)
    
    return parameters, costs

# Define the model architecture
layer_dims = [784, 128, 64, 10]  # Input layer, two hidden layers, output layer
activations = ["relu", "relu", "softmax"]

# Train the model with Adam optimizer
parameters, costs = model(X_train, Y_train, layer_dims, activations, optimizer="adam", learning_rate=0.001, num_epochs=2000, print_cost=True)

# Plotting the cost function
plt.plot(costs)
plt.ylabel('Cost')
plt.xlabel('Iterations (per hundreds)')
plt.title('Cost reduction over time')
plt.show()

# Predict on the test set
def predict(X, parameters, activations):
    A = X
    L = len(parameters) // 2  # number of layers in the neural network
    caches = []
    
    for l in range(1, L + 1):
        A_prev = A
        W = parameters['W' + str(l)]
        b = parameters['b' + str(l)]
        A, cache = activate_forward(A_prev, W, b, activations[l-1])
        caches.append(cache)
    
    return A

Y_pred = predict(X_test, parameters, activations)

# Compute accuracy
def compute_accuracy(Y_pred, Y_true):
    predictions = np.argmax(Y_pred, axis=0)
    labels = np.argmax(Y_true, axis=0)
    accuracy = np.mean(predictions == labels)
    return accuracy

accuracy = compute_accuracy(Y_pred, Y_test)
print(f"Test set accuracy: {accuracy * 100:.2f}%")
