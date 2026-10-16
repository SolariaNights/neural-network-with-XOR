import numpy as np

#sigmoid function
def nonlin(x, deriv=False):
    if deriv == True:
        return x * (1-x)
    return 1/(1+np.exp(-x))

# input dataset
x = np.array([[0,0,1],  
              [0,1,1],
              [1,0,1],
              [1,1,1]])

#output dataset
y = np.array([[0],[1],[1],[0]])

#seed
#deterministic
np.random.seed(1)

#initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,2)) - 1
syn1 = 2*np.random.random((2,1)) - 1
bias_output = 0.0

for iter in range(10000):
    # forward propagation
    l0 = x
    l1 = nonlin(np.dot(l0, syn0))
    l2 = nonlin(np.dot(l1, syn1) + bias_output)

    # how much did we miss?
    l2_error = y - l2

    # multiply how much we missed by the slope of the sigmoid at the values in l1
    l2_delta = l2_error * nonlin(l2, True)
    bias_output += np.sum(l2_delta)
    
    if iter == 0:
        print("l0 shape:", l0.shape)
        print("l2_delta shape:", l2_delta.shape)
        print("syn0 shape:", syn0.shape)
        print("syn1 shape:", syn1.shape)

    # update weights
    l1_error = np.dot(l2_delta, syn1.T)
    l1_delta = l1_error * nonlin(l1, True)
    syn1 += np.dot(l1.T, l2_delta)
    syn0 += np.dot(l0.T, l1_delta)

print("Output After Training:")
print(l2)
print(l1)
print(syn0)
print(syn1)
print("Expected:")
print(y)
