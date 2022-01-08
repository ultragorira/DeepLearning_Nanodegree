import numpy as np

# Write a function that takes as input a list of numbers, and returns
# the list of values given by the softmax function.
def softmax(L):
    expL = np.exp(L) #=> this will do e^x for each item in the list
    return np.divide(expL, expL.sum()) #=> this will divide each item in expL by the sum of all items in expL
