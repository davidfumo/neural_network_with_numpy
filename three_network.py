# https://iamtrask.github.io/2015/07/12/basic-python-network/

# 2 layer neural network

import numpy as np

# define the non-linearity using sigmoid function
def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

# input data (4, 3)
X = np.array([ [0,0,1],
               [0,1,1],
               [1,0,1],
               [1,1,1] ])


# output data (4,1)
# y = np.array([[0,0,1,1]]).T
y = np.array([[0,1,1,0]]).T


# seed random numbers to make calculation
# seed helps for results reproducibility
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,4)) - 1
syn1 = 2*np.random.random((4,1)) - 1


for i in range(60000):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))

    # how much did we miss?
    # l1_error = y - l1
    l2_error = y - l2

    if(i % 10000) == 0:
        print("error: ", str(np.mean(np.abs(l2_error))))

    # In what direction is the target value?
    # were we really sure? if so, don't change too much.
    l2_delta = l2_error * nonlin(l2, deriv=True)

    # how much did each l1 value contribute to
    # the l2 error (according to the weights)?
    l1_error = l2_delta.dot(syn1.T)

    # in what direction is the target l1?
    # were we really sure? if so, don't change too much.
    l1_delta = l1_error * nonlin(l1,deriv=True)
        
    # update weights
    #print(l0.T.shape)
    #print(l1_delta.shape)
    #print(syn0.shape)
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
