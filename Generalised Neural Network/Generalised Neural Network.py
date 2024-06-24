import numpy as np
from matplotlib import pyplot as plt
import torch


class forward_Prop:
    #The idea is to generalised the idea of forward propogation for any number of layers and nodes for the neural network
    def __init__(self, x, y, b):
        self.x = x
        self.y = y
        self.b = b

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
        #dz is the gradient of the loss function with respect to the output of the linear layer which happens to be same as the input of the activation function
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
    def __init__(self):
        pass

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
    # This should also take care of number of layers and nodes in each layer
    def __init__(self, x, y, learning_rate, epochs, nodes_num, layers_num, activations, optimiser, error):
        self.x = x
        self.y = y
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.nodes_num = nodes_num
        self.layers_num = layers_num
        self.activations = activations
        self.optimiser = optimiser
        self.error = error

class parameter_initialisation:
    def __init__(self, nodes_num, layers_num, x, y):
        self.nodes_num = nodes_num
        self.layers_num = layers_num
        self.x = x

    #The idea is to create 'n+2' layers with 'd' nodes in each layer. For this, w_0 \in R^{m x d}[## I am expecting the input to be m x 1] and b_0 \in R^{d x 1} and w_1 \in R^{d x d} and b_1 \in R^{d x 1} and so on till w_{n} \in R^{d x d} and b_{n} \in R^{d x 1} and w_{n+1} \in R^{d x k} and b_{n+1} \in R^{k x 1} where k is the number of classes in the classification problem

    # 1. w={w_0,w_1,...,w_{n},w_{n+1}} and b={b_0,b_1,...,b_{n},b_{n+1}}
    # 2. a={a_0,a_1,...,a_{n},a_{n+1}} , a_0 till a_{n} \in R^{d x 1} and a_{n+1} \in R^{k x 1}
    # 3. z={z_0,z_1,...,z_{n},z_{n+1},z_{n+2}} , z_0 till z_{n+1} \in R^{d x 1} and z_{n+2} \in R^{k x 1}
    # 4. where z_0=x and a_0=w_0^T.x+b_0. Similarly, z_1=h(a_0) and a_1=w_1^T.z_1+b_1 and so on till z_{n+2}=h(a_{n+1}) where h is the activation function


    def weights_and_biases_initialisation(self, x, y, nodes_num, layers_num):
        weights = {}
        biases = {}
        for i in range(layers_num+1):
            if i == 0:
                weights['w_0'] = np.random.randn(x.shape[1], nodes_num)
                biases['b_0'] = np.random.randn(1, nodes_num)
            elif i == layers_num:
                weights['w_'+str(i)] = np.random.randn(nodes_num, y.shape[1])
                biases['b_'+str(i)] = np.random.randn(1, y.shape[1])
            else:
                weights['w_'+str(i)] = np.random.randn(nodes_num, nodes_num)
                biases['b_'+str(i)] = np.random.randn(1, nodes_num)

        return weights, biases
    
    def activation_initialisation(self, x, y, nodes_num, layers_num):
        z={}
        for i in range(layers_num+2):
            if i == 0:
                z['z_0'] = x
            else:
                z['z_'+str(i)] = np.zeros((x.shape[0], nodes_num))
        return z

    
    def layer_forward_linear_initialisation(self, x, y, nodes_num, layers_num):
        a={}
        for i in range(layers_num+1):
                a['a_'+str(i)] = np.zeros((x.shape[0], nodes_num)) #we will evaluate in network class using forward class
        return a

    

    




