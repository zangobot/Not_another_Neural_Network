import numpy as np
import matplotlib.pyplot as plt
from func import relu_func, sq_loss_func

class ShallowNN:
    def __init__(self, input_dimension, loss_function = None, act_func = None, weights = None, gamma = 1e-2):
        self.i_d = input_dimension
        self.loss_function = loss_function if loss_function is not None else sq_loss_func
        self.activation = act_func if act_func is not None else relu_func
        self.gamma = gamma
        self.bias = np.random.randn()
        if weights is None:
            self.weights = np.random.randn(input_dimension)
        
    def feed(self, x):
        return self.activation.func( self.linear(x) )

    def matrix_feed(self, X):
        return self.activation.func(self.matrix_linear(X))

    def linear(self, x):
        return np.dot( self.weights.T , x ) + self.bias

    def matrix_linear(self, X):
        result =  np.dot( X , self.weights ) + self.bias 
        return result

    def back_propagate(self,x, delta):
        dl = delta
        do = self.activation.derivative(self.linear(x)) 
        dw = x
        new_w = self.weights - self.gamma * dw * do * dl
        new_b = self.bias - self.gamma * do * dl
        self.weights = new_w
        self.bias = new_b


    def matrix_back_propagation(self, X, delta):
        #TODO: complete GD 
        dl = delta
        do = self.activation.derivative(self.matrix_linear(X)) * dl
        dw = X
        n = len(X)
        new_w = self.weights - self.gamma *  np.dot(dw.T, do) / n
        new_b = self.bias - self.gamma * np.sum(do * dl) / n
        self.weights = new_w
        self.bias = new_b

    def fit(self, X, Y, animate=False, matrix = False):
        if matrix:
            self.__matrix_fit__(X,Y)
        else: 
            self.__fit__(X,Y,animate=animate)

    def __matrix_fit__(self, X, Y):
        Y_pred = self.matrix_feed(X)
        delta = self.loss_function.derivative( np.sign(Y_pred), Y )
        self.matrix_back_propagation(X, delta)

    def __fit__(self, X, Y, animate=False):
        if animate: 
            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.scatter(X[:,0], X[:,1], c=Y)
            x_axis = np.linspace(np.min(X[:,0]), np.max(X[:,1]), 100)
            line = -(self.weights[0] / self.weights[1]) * x_axis - self.bias / self.weights[1]
            boundary, = ax.plot(x_axis, line )
            plt.show(block=False)

        for (x, y) in zip(X,Y):
            y_p = self.feed(x)
            delta = self.loss_function.derivative( np.sign(y_p), y ) 
            self.back_propagate(x, delta)

            if animate:
                line = -(self.weights[1] / self.weights[0]) * x_axis - self.bias / self.weights[1]
                boundary.set_ydata(line)
                fig.canvas.draw()
                fig.canvas.flush_events()
    
    def predict(self, X):
        y_pred = []
        print(self.weights, self.bias)
        for x in X:
            y = self.feed(x)
            y_pred.append( np.sign(y) )
        return np.array(y_pred)

        
                
        




    