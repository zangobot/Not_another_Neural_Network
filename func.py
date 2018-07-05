import numpy as np

def relu(x):
    if not np.isscalar(x):
        return np.array([ np.max(_x, 0) for _x in x] )
    return np.max(x,0)

def relu_drv(x):
    if not np.isscalar(x):
        return np.array( [1 if _x >0 else 0 for _x in x] )
    return 1 if x >0 else 0

def square_loss(y1, y2):
    return (y1 - y2)**2

def sq_loss_dr(y1,y2):
    return y1 - y2

def sigm(x):
    return 1 / (np.exp(-x)+1)

def sigm_drv(x):
    return sigm(x)*(1-sigm(x))

class Function:
    def __init__(self, func, derivative):
        self.func = func
        self.derivative = derivative

relu_func = Function(relu, relu_drv)
sigmoid_func = Function(sigm, sigm_drv)
sq_loss_func = Function(square_loss, sq_loss_dr)