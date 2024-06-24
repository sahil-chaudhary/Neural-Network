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
    def __init__(self, x, y, b):
        self.x = x
        self.y = y
        self.b = b

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

    def gradient_descent(self, weights, biases, dw, db):
        weights -= self.learning_rate*dw
        biases -= self.learning_rate*db
        return weights, biases
    
    def momentum(self, weights, biases, dw, db, beta):
        v_w = beta*v_w + (1-beta)*dw
        v_b = beta*v_b + (1-beta)*db
        weights -= self.learning_rate*v_w
        biases -= self.learning_rate*v_b
        return weights, biases
    
    def rmsprop(self, weights, biases, dw, db, beta):
        s_w = beta*s_w + (1-beta)*dw**2
        s_b = beta*s_b + (1-beta)*db**2
        weights -= self.learning_rate*dw/np.sqrt(s_w + 1e-8)
        biases -= self.learning_rate*db/np.sqrt(s_b + 1e-8)
        return weights, biases
    
    def adam(self, weights, biases, dw, db, beta1, beta2):
        v_w = beta1*v_w + (1-beta1)*dw
        v_b = beta1*v_b + (1-beta1)*db
        s_w = beta2*s_w + (1-beta2)*dw**2
        s_b = beta2*s_b + (1-beta2)*db**2
        v_w_corrected = v_w/(1-beta1)
        v_b_corrected = v_b/(1-beta1)
        s_w_corrected = s_w/(1-beta2)
        s_b_corrected = s_b/(1-beta2)
        weights -= self.learning_rate*v_w_corrected/np.sqrt(s_w_corrected + 1e-8)
        biases -= self.learning_rate*v_b_corrected/np.sqrt(s_b_corrected + 1e-8)
        return weights, biases
    

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
    def __init__(self, x, y, learning_rate, epochs, nodes_num, layers_num, activations,  optimiser, error, weights, biases, a, z):
        self.x = x
        self.y = y
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.nodes_num = nodes_num
        self.layers_num = layers_num
        self.activations = activations
        self.optimiser = optimiser
        self.error = error
        self.weights = weights
        self.biases = biases
        self.a = a
        self.z = z

    def update_parameters(self, x, y, weights, biases, a, z, nodes_num, layers_num, activations, learning_rate):
        for i in range(layers_num+1):
            a['a_'+str(i)] = forward_Prop.linear(z['z_'+str(i)], weights['w_'+str(i)], biases['b_'+str(i)])
            if activations[i] == 'sigmoid':
                z['z_'+str(i+1)] = activation.sigmoid(a['a_'+str(i)])
            elif activations[i] == 'relu':
                z['z_'+str(i+1)] = activation.relu(a['a_'+str(i)])
            elif activations[i] == 'tanh':
                z['z_'+str(i+1)] = activation.tanh(a['a_'+str(i)])
            elif activations[i] == 'softmax':
                z['z_'+str(i+1)] = activation.softmax(a['a_'+str(i)])
        print('done with update parameters')
        return a, z
    
    def train(self, x, y, weights, biases, a, z, nodes_num, layers_num, activations, learning_rate, epochs):
        loss = []
        accuracy = []
        for i in range(epochs):
            a, z = Generalised_Neural_Network.update_parameters(x, y, weights, biases, a, z, nodes_num, layers_num, activations, learning_rate)
            loss.append(error.cross_entropy_error(z['z_'+str(layers_num+1)], y))
            print('Epoch:', i, 'Loss:', loss[-1])
            accuracy.append(accuracy.accuracy(predict.predict(z['z_'+str(layers_num+1)]), y))
            print('Accuracy:', accuracy[-1])
            dz = back_Prop.activation_backward(z['z_'+str(layers_num+1)] - y, a['a_'+str(layers_num)], activations[layers_num])
            for i in range(layers_num, -1, -1):
                dx, dw, db = back_Prop.linear_backward(dz, z['z_'+str(i)], weights['w_'+str(i)], biases['b_'+str(i)])
                weights['w_'+str(i)] -= learning_rate*dw
                biases['b_'+str(i)] -= learning_rate*db
                dz = back_Prop.activation_backward(dx, a['a_'+str(i)], activations[i])
        return weights, biases, loss, accuracy
    
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
        print('done with weights and biases initialisation')
        return weights, biases
    
    def activation_initialisation(self, x, y, nodes_num, layers_num):
        z={}
        for i in range(layers_num+2):
            if i == 0:
                z['z_0'] = x
            else:
                z['z_'+str(i)] = np.zeros((x.shape[0], nodes_num))
        print('done with activation initialisation')
        return z

    
    def layer_forward_linear_initialisation(self, x, y, nodes_num, layers_num):
        a={}
        for i in range(layers_num+1):
                a['a_'+str(i)] = np.zeros((x.shape[0], nodes_num)) #we will evaluate in network class using forward class
        print('done with layer forward linear initialisation')
        return a
    
    def activations_initialisation(self, x, y, nodes_num, layers_num):
        activations = []
        for i in range(layers_num):
            activations.append('sigmoid')
        activations.append('softmax')
        print('done with activations initialisation')
        return activations
    

#Get the MNIST DATASET
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)
x, y = mnist['data'], mnist['target']
x = x/255
y = np.array(y, dtype='int')
y = np.eye(10)[y]
x_train, x_test, y_train, y_test = x[:60000], x[60000:], y[:60000], y[60000:]

#Initialise the parameters
nodes_num = 10
layers_num = 2
learning_rate = 0.01
epochs = 100
parameter_initialisation = parameter_initialisation(nodes_num, layers_num, x_train, y_train)
weights, biases = parameter_initialisation.weights_and_biases_initialisation(x_train, y_train, nodes_num, layers_num)
a = parameter_initialisation.layer_forward_linear_initialisation(x_train, y_train, nodes_num, layers_num)
z = parameter_initialisation.activation_initialisation(x_train, y_train, nodes_num, layers_num)
activations = parameter_initialisation.activations_initialisation(x_train, y_train, nodes_num, layers_num)


#Train the model
generalised_neural_network = Generalised_Neural_Network(x_train, y_train, learning_rate, epochs, nodes_num, layers_num, activations, optimiser, error, weights, biases, a, z)
weights, biases, loss, accuracy1 = generalised_neural_network.train(x_train, y_train, weights, biases, a, z, nodes_num, layers_num, activations, learning_rate, epochs)

#Plot the loss and accuracy
plot = plot(x_train, y_train, weights)
plot.plot_loss(loss)
plot.plot_accuracy(accuracy1)
plot.plot(loss, accuracy1)

#Make predictions
predict = predict(x_test, weights)
y_pred = predict.predict(x_test)
accuracy2 = accuracy(y_pred, y_test)
print('Accuracy:', accuracy2)


    

    




