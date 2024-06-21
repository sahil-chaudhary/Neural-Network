import numpy as np
from matplotlib import pyplot as plt
import torch


class forward_Prop:
    #The idea is to generalised the idea of forward propogation for any number of layers and nodes for the neural network

class back_Prop:
    #The idea is to generalised the idea of back propogation for any number of layers and nodes for the neural network. This class will take care of finding gradients of the loss function with respect to the weights and biases of the neural network using the chain rule of differentiation

class optimiser:
    #This will take care of the collection of optimisation algorithms that can be used to update the weights and biases of the neural network. Later, we will compare the performance of different optimisation algorithms on the same neural network and make some conclusions.
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def sgd(self, parameters, gradients):

    def momentum(self, parameters, gradients):

    def adam(self, parameters, gradients):

class error:
    #This class will take care of the different error functions that can be used to evaluate the performance of the neural network. Later, we will compare the performance of different error functions on the same neural network and make some conclusions.
    def __init__(self) -> None:
        
        
    def mean_squared_error(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)
    
    def cross_entropy_error(self, y_pred, y_true):
        return -np.sum(y_true*np.log(y_pred))
    

class predict:
    #This class will take care of the prediction of the neural network. This class will have the following methods:

class plot:
    #This class will take care of the plotting of the neural network. This class will have the following methods:
    #1. plot_loss : This will plot the loss function of the neural network
    #2. plot_accuracy : This will plot the accuracy of the neural network
    #3. plot : This will plot both the loss function and the accuracy of the neural network
    #4. plot_decision_boundary : This will plot the decision boundary of the neural network. This will be useful for the classification problems. This will help us to visualise how the neural network is making the decision on the basis of the input features.
    #5. plot_confusion_matrix : This will plot the confusion matrix of the neural network. This will help us to visualise how the neural network is performing on the basis of the true labels and the predicted labels.


class accuracy:
    #This class will take care of the accuracy of the neural network.


class Generalised_Neural_Network:
    #Ofcourse, the main class that will take care of the neural network. This is the class where we will call the above classes to train the model and make predictions.
