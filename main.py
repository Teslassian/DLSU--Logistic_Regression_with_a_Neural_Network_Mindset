#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# Load the datasets
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()
# #Test the images and their respective labels
# index = 1
# plt.imshow(train_set_x_orig[index])
# print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
# plt.show()

# Create variables of dataset sizes
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

# Reshape datasets into vectors
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# Normalize the data
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255

# Function to calculate the sigmoid function
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Return:
    s -- sigmoid(z)
    """

    s = 1/(1+np.exp(-z))
    return s
# #Test the sigmoid function
# print("sigmoid([0,2]) = " + str(sigmoid(np.array([0,2]))))

# Function to initialize parameters
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    w = np.zeros((dim,1))
    b = 0

    assert(w.shape == (dim,1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w,b
# Initialize w, b.
dim = 2
w,b = initialize_with_zeros(dim)
print("w = " + str(w))
print("b = " + str(b))

# Function for forward and backward propagation
def propagate(w, b, X, Y):
    """
    Implement the cost function and its gradient for the propagation explained above

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b

    Tips:
    - Write your code step by step for the propagation. np.log(), np.dot()
    """

    m = X.shape[1]

    # Forward Propagation (X to Cost)
    A = sigmoid(w.T @ X + b) #1x209
    cost = -1/m * (Y @ (np.log(A).T) + (1-Y) @ (np.log(1-A).T))

    # Backward Propagation (find Grad)
    dw = 1/m * X @ ((A-Y).T)
    db = 1/m * sum(sum(A - Y))

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw":dw,
             "db":db}

    return grads, cost
#Initialize w, b, X, Y. Calculate grads and cost.
w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
grads, cost = propagate(w, b, X, Y)
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))
print("cost = " + str(cost))

# Function for optimization
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.

    Tips:
    You basically need to write down two steps and iterate through them:
        1) Calculate the cost and the gradient for the current parameters. Use propagate().
        2) Update the parameters using gradient descent rule for w and b.
    """

    costs = []

    for i in range(num_iterations):
        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)

        # Derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # Update rule
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print every 100th cost
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs
#Optimize w, b, dw, db.
params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate = 0.009, print_cost=False)
print("w = " + str(params["w"]))
print("b = " + str(params["b"]))
print("dw = " + str(grads["dw"]))
print("db = " + str(grads["db"]))













# Variables and their sizes/values
print("Matrices:\n\n")
print("train_set_x_orig: " + str(train_set_x_orig.shape) + "\n")
print("train_set_x: " + str(train_set_x.shape) + "\n")
print("train_set_x_flatten: " + str(train_set_x_flatten.shape) + "\n")
print("train_set_y: " + str(train_set_y.shape) + "\n")
print("\n")

print("test_set_x_orig: " + str(test_set_x_orig.shape) + "\n")
print("test_set_x: " + str(test_set_x.shape) + "\n")
print("test_set_x_flatten: " + str(test_set_x_flatten.shape) + "\n")
print("test_set_y: " + str(test_set_y.shape) + "\n")
print("\n")

print("Variables:\n\n")
print("m_train: " + str(m_train) + "\n")
print("m_test: " + str(m_test) + "\n")
print("num_px: " + str(num_px) + "\n")
print("\n")
