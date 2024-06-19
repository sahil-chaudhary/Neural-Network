#Neural Network

I wanted to learn neural network and I amd doing the same!

Neural Network model can be described as a series of functional transformation.
First, consider M linear combinations of the input x.

Let $x \in R^n, w^{(1)} \in R^{n \times m}$ and $b^{(1)} \in R^m$
\begin{align}
    a^{(1)}=w^{(1)}^Tx+b^{(1)}
\end{align}
\begin{align*}
    \begin{bmatrix}
        a_1^{(1)}\\
        a_2^{(1)}\\
        \vdots
        \\
        a_m^{(1)}
    \end{bmatrix}&=\begin{bmatrix}
        w_{11} & w_{12} & \cdots & w_{1m}\\
        w_{21} & w_{22} & \cdots & w_{2n}\\
        \vdots & \vdots & \ddots & \vdots\\
        w_{n1} & w_{n2} & \cdots & w_{nm}
    \end{bmatrix}^T
    \begin{bmatrix}
        x_1\\
        x_2\\
        \vdots\\
        x_n
    \end{bmatrix}+\begin{bmatrix}
        b_1^{(1)}\\
        b_2^{(1)}\\
        \vdots\\
        b_m^{(1)}
    \end{bmatrix}\\
    &=\begin{bmatrix}
        w_{11} & w_{21} & \cdots & w_{n1}\\
        w_{12} & w_{22} & \cdots & w_{n2}\\
        \vdots & \vdots & \ddots & \vdots\\
        w_{1m} & w_{2n} & \cdots & w_{nm}
    \end{bmatrix}\begin{bmatrix}
        x_1\\
        x_2\\
        \vdots\\
        x_n
    \end{bmatrix}+\begin{bmatrix}
        b_1^{(1)}\\
        b_2^{(1)}\\
        \vdots\\
        b_m^{(1)}
    \end{bmatrix}\\
\end{align*}
The evaluated matrix '$a^{(1)}$' are known as activations. This is then transformed using a \textbf{differentiable, nonlinear} activation function $h(\cdot)$ to give
\begin{align}
    z^{(2)}=h(a^{(1)})
\end{align}
In particular,
\begin{align*}
    \begin{bmatrix}
        z_1^{(2)}\\
        z_2^{(2)}\\
        \vdots\\
        z_m^{(2)}
    \end{bmatrix}=\begin{bmatrix}
        h(a_1^{(1)})\\
        h(a_2^{(1)})\\
        \vdots\\
        h(a_m^{(1)})
    \end{bmatrix}
\end{align*}
These quantities in context of neural networks are called hidden units. The nonlinear functions $h(\cdot)$ are generally chosen to be sigmoidal functions such as the logistic sigmoid or the tanh function.
Following, these values are again linearly combined to give another unit activations of k nodes.\\

Let $w^{(2)} \in R^{m \times k}, b^{(2)} \in R^k$
\begin{align}
    a^{(2)}=w^{(2)}^Tz^{(2)}+b^{(2)}
\end{align}
This transformations corresponds to the second layer of the network and one can choose on keep on going for as many layers as they want.

For the sake of notes, let us say this is our output unit activations. The output unit activations are transformed using an appropriate activation function to give a set of network output $y$.
The choice of activation function is determined by the nature of the data and the assumed distribution of target variables.
For standard regression problems, the activation function is the identity so that $y=a$. Similarly, for multiple binary classification problems, each output unit activation is tranformed using a logistic sigmoid function so that
\begin{align}
    y&=\sigma\left(a^{(2)}\right)\\
    &=\sigma\left(w^{(2)}^Tz^{(2)}+b^{(2)}\right)\\
    &=\sigma\left(w^{(2)}^Th\left(a^{(1)}\right)+b^{(2)}\right)\\
    &=\sigma\left(w^{(2)}^Th\left(w^{(1)}^Tx+b^{(1)}\right)+b^{(2)}\right)
\end{align}

The process of evaluating $eq^n-10$ can be interpreted as a forward propagation of information through the network. As can be seen from derivation, the neural network model comprises of multiple stages of processing, each of which resembles the perceptron model and for this reason the neural network is known as the multilayer perceptron or MLP.

A key difference compared to the perceptron, however is that the neural network uses continuous sigmoidal nonlinearities in the hidden units, whereas the perceptron uses step-function nonlinearities.

\subsection{Network Training}
So far, we have viewed neural networks as a general class of parametric non-linear functions from a vector x of input variables to a vector y of output variables.
Let's call these set of weight matrix for each layers as $w=\left\{w^{(1)},w^{(2)},\cdots,w^{(k)}\right\}$.
Given a training set comprising a set of input vectors $\{x_n\}$, together with a corresponding set of target vectors $\{t_n\}$, we minimize the error function
\begin{align}
    E(w)=\frac{1}{2}\sum_{n=1}^{N}||y(x_n,w)-t_n||^2
\end{align}

Like every other machine learning algorithm, now our goal is to find a vector w such that $E(w)$ takes it smallest value.
\begin{align}
    \nabla E(w)=0
\end{align}
Please note that, one can choose to consider any other error functions. Its easier to do analysis in square loss function and hence I have chosen this. However, the error function typically has a highly nonlinear dependence on the weights and bias parameters, and so there will be many points in weight space at which the gradient vanishes.

Also, I would like to point out there are different methods to determine these weights and every other methods has its own merit. The only problem is, the determined weights might not be most optimal weight we might want!

There is clearly no hope of finding an analytical solution to the $eq^n-12$, we resort to iterative numerical procedures. Most of the techniques involve choosing some initial value $w^{(0)}$ for the weight vector and then moving through weight space in a succession of steps of the form
\begin{align}
    w^{(t+1)}=w^{(t)}+ \Delta w^{(t)}
\end{align}

\subsubsection{Gradient Descent Optimization}
The simplest approach to using gradient information is to choose the weight update in $eq^n-13$ to comprise a small step in the direction of the negative gradient so that,
\begin{align}
    w^{(t+1)}=w^{(t)}-\eta\nabla E(w^{(t)})
\end{align}
where the parameter $\eta>0$ is known as learning rate.

After each such update, the gradient is re-evaluated for the new weight vector and the process is repeated.

\subsubsection{Back Propagation}
Our goal in this section is to find an efficient technique for evaluating the gradient of an error function $E(w)$ for a feed forward neural network. We will see that this can be achieved using a local message passing scheme in which information is sent alternatively forwards and backwards through the network and is known as back propagation. 

Consider our famous square loss function.
\begin{align*}
    E(w)&=\frac{1}{2}\sum_{n=1}^{N}||y(x_n,w)-t_n||^2\\
    &=\sum_{n=1}^{N}E_n(w)
\end{align*}
where $E_n(w)=\frac{1}{2}||y(x_n,w)-t_n||^2$.

The gradient of this error function with respect to a weight vector w is,
\begin{align}
    \nabla E_n(w)&=(y_n-t_n)\nabla y_n
\end{align}
In general feed forward network, each unit computes a weighted sum of its input variables of the form,
\begin{align*}
    a_j^{(k)}=\sum_{i}w_{ij}^{(k)}z_i^{^{(k)}}+b_j^{(k)}
\end{align*}

This is then transformed using a nonlinear activation function $h(\cdot)$ to give the activation $z^{(k+1)}$ in the form
\begin{align*}
    z^{(k+1)}=h\left(a^{(k)}\right)
\end{align*}

To generalize the functions,
\begin{align*}
    y &= h^{(k)}\left(a^{(k)}\right)\\
    &=h^{(k)}\left(w^{(k)}^Tz^{(k)}+b^{(k)}\right)\\
    &=h^{(k)}\left(w^{(k)}h^{(k-1)}\left(w^{(k-1)}z^{(k-1)}+b^{(k-1)}\right)+b^{(k)}\right)
\end{align*}
Now, differentiating using first princple of derivative is very time consuming!
\begin{align}
    \nabla y &= \begin{bmatrix}
        \frac{\partial y}{\partial w_{11}^{(1)}} & \frac{\partial y}{\partial w_{12}^{(1)}} & \cdots &\frac{\partial y}{\partial w_{ij}^{(k)}} & \cdots & \frac{\partial y}{\partial b_{1}^{(1)}} & \cdots & \frac{\partial y}{\partial b_{n}^{(k)}} 
    \end{bmatrix}
\end{align}
Using chain rule,
\begin{align}
    \frac{\partial y}{\partial w_{ij}^{(k)}}&=\frac{\partial y}{\partial a_j^{(k)}}\frac{\partial a_j^{(k)}}{\partial w_{ij}^{(k)}}\\
    &=\delta_j^{(k)}\frac{\partial a_j^{(k)}}{\partial w_{ij}^{(k)}}
\end{align}
where the $\delta's$ are often reffered as errors. Also,
\begin{align}
    \frac{\partial a_j^{(k)}}{\partial w_{ij}^{(k)}}=z_i^{(k)}
\end{align}
So, finding required derivative is simply multiplying the value of $\delta$ for the unit at the output end of the weight by the value of z for the unit at the input end of the weight.
To evaluate $\delta's$ for hidden units, we again make use of the chain rule for partial derivatives,
\begin{align}
    \delta_j^{(k)}\equiv \frac{\partial y}{\partial a_j^{(k)}}=\sum_{i}\frac{\partial y}{\partial a_i^{(k+1)}}\frac{\partial a_i^{(k+1)}}{\partial a_j^{(k)}}
\end{align}
Basically, the sums run over all the units i and x(layer) to which j sends connections.
If we use the derived formula of y for feed-forward network from last derivation, we have
\begin{align}
    \delta_{j}^{(k)}=h'(a^{(K)})\sum_{i}w_{ij}^{(k+1)}\delta_{i}^{(k+1)}
\end{align}

Which tells us that the value of $\delta$ for a particular hidden unit can be obtained by propagating the $\delta's $ backwards from units higher up in the network.

This is all just fancy maths to describe the method, If I be real and just tell you the method, the idea is to use chain rule in the error function with respect to connected weights and keep on going until you reach your desired weights.
