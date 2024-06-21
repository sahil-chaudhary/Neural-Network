import numpy as np
from matplotlib import pyplot as plt
import torch


class forward_Prop:
    #The idea is to generalised the idea of forward propogation for any number of layers and nodes for the neural network
    def __init__(self, x, y, parameters):
        self.x = x
        self.y = y
        self.parameters = parameters

    def linear(self, x, w, b):
        return np.dot(x, w) + b
    
class activation:
    #The idea is to generalised the idea of activation functions for any number of layers and nodes for the neural network
    def __init__(self, x):
        self.x = x

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def relu(self, x):
        return np.maximum(0, x)

    def tanh(self, x):
        return np.tanh(x)

    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x), axis=0)
    

class back_Prop:
    #The idea is to generalised the idea of back propogation for any number of layers and nodes for the neural network. This class will take care of finding gradients of the loss function with respect to the weights and biases of the neural network using the chain rule of differentiation
    def __init__(self, x, y, parameters):
        self.x = x
        self.y = y
        self.parameters = parameters

    def linear_backward(self, dz, x, w, b):
        dw = np.dot(x.T, dz)
        db = np.sum(dz, axis=0)
        dx = np.dot(dz, w.T)
        return dx, dw, db
    
    def activation_backward(self, dz, x, activation):
        if activation == 'sigmoid':
            return dz*activation.sigmoid(x)*(1-activation.sigmoid(x))
        elif activation == 'relu':
            return np.where(x>0, dz, 0)
        elif activation == 'tanh':
            return 1 - np.tanh(x)**2
        elif activation == 'softmax':
            return dz*activation.softmax(x)*(1-activation.softmax(x))


class optimiser:
    #This will take care of the collection of optimisation algorithms that can be used to update the weights and biases of the neural network. Later, we will compare the performance of different optimisation algorithms on the same neural network and make some conclusions.
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def sgd(self, parameters, gradients):
        for param, grad in zip(parameters, gradients):
            param -= self.learning_rate*grad

    def momentum(self, parameters, gradients):
        for param, grad in zip(parameters, gradients):
            param -= self.learning_rate*grad

    def adam(self, parameters, gradients):
        for param, grad in zip(parameters, gradients):
            param -= self.learning_rate*grad

    def rmsprop(self, parameters, gradients):
        for param, grad in zip(parameters, gradients):
            param -= self.learning_rate*grad


class error:
    #This class will take care of the different error functions that can be used to evaluate the performance of the neural network. Later, we will compare the performance of different error functions on the same neural network and make some conclusions.
    def __init__(self) -> None:
        
        
    def mean_squared_error(self, y_pred, y_true):
        return np.mean((y_pred - y_true)**2)
    
    def cross_entropy_error(self, y_pred, y_true):
        return -np.sum(y_true*np.log(y_pred))
    

class predict:
    #This class will take care of the prediction of the neural network. 
    def __init__(self, x, parameters):
        self.x = x
        self.parameters = parameters

    def predict(self, x):
        return np.argmax(x, axis=1)
    

class plot:
    #This class will take care of the plotting of the neural network. This class will have the following methods:
    #1. plot_loss : This will plot the loss function of the neural network
    #2. plot_accuracy : This will plot the accuracy of the neural network
    #3. plot : This will plot both the loss function and the accuracy of the neural network
    #4. plot_decision_boundary : This will plot the decision boundary of the neural network. This will be useful for the classification problems. This will help us to visualise how the neural network is making the decision on the basis of the input features.
    #5. plot_confusion_matrix : This will plot the confusion matrix of the neural network. This will help us to visualise how the neural network is performing on the basis of the true labels and the predicted labels.
    def __init__(self, x, y, parameters):
        self.x = x
        self.y = y
        self.parameters = parameters

    def plot_loss(self, loss):
        plt.plot(loss)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Function')
        plt.show()

    def plot_accuracy(self, accuracy):
        plt.plot(accuracy)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.show()

    def plot(self, loss, accuracy):
        plt.plot(loss, label='Loss')
        plt.plot(accuracy, label='Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Loss/Accuracy')
        plt.title('Loss/Accuracy')
        plt.legend()
        plt.show()

    def plot_decision_boundary(self, x, y, parameters):
        pass

    def plot_confusion_matrix(self, y_true, y_pred):
        pass




class accuracy:
    #This class will take care of the accuracy of the neural network.
    def __init__(self, y_pred, y_true):
        self.y_pred = y_pred
        self.y_true = y_true

    def accuracy(self, y_pred, y_true):
        return np.mean(y_pred == y_true)
    



class Generalised_Neural_Network:
    #Ofcourse, the main class that will take care of the neural network. This is the class where we will call the above classes to train the model and make predictions.
    def __init__(self, x, y, parameters, learning_rate, epochs):
        self.x = x
        self.y = y
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.epochs = epochs

    def train(self, x, y, parameters, learning_rate, epochs):
        pass

    def predict(self, x, parameters):
        pass

    def evaluate(self, y_pred, y_true):
        pass

    def plot(self, loss, accuracy):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def summary(self):
        pass



    

