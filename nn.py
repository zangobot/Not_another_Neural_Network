import numpy as np
import matplotlib.pyplot as plt
from func  import relu_func, sq_loss_func

class SingleNeuronNN:
    """
    A single neuron.
    """
    def __init__(self, input_dimension, loss_function = None, act_func = None, weights = None, gamma = 1e-2):
        """
        Creates a single neuron. The dimension is given by the input_dimension parameter.
        Initial weights can be assigned or randomly initialized. 
        Loss and activation functions needs to be Function parameters. If None, the square loss and relu are used.
        Gamma is gradient descent parameter.
        """
        self.i_d = input_dimension
        self.loss_function = loss_function if loss_function is not None else sq_loss_func
        self.activation = act_func if act_func is not None else relu_func
        self.gamma = gamma
        self.bias = np.random.randn()
        self.__last_input__ = None
        if weights is None:
            self.weights = np.random.randn(input_dimension)
        
    def feed(self, x):
        """
        Returns the output for a single point.
        """
        self.__last_input__ = x
        return self.activation.func( self.linear(x) )

    def matrix_feed(self, X):
        """
        Returns the output for a set of points.
        """
        self.__last_input__ = X
        return self.activation.func(self.matrix_linear(X))

    def linear(self, x):
        """
        The inner linear function, which is f(x) = w.t * x + b
        """
        return np.dot( self.weights.T , x ) + self.bias

    def matrix_linear(self, X):
        """
        The inner linear function, which is f(x) = w.t * x + b, applied to a set of points.
        """
        result =  np.dot( X , self.weights ) + self.bias 
        return result

    def back_propagate(self,x, delta):
        """
        Back propagation algorithm for a single point.
        """
        dl = delta
        do = self.activation.derivative(self.linear(x)) 
        dw = x
        new_w = self.weights - self.gamma * dw * do * dl
        new_b = self.bias - self.gamma * do * dl
        self.weights = new_w
        self.bias = new_b
        return dw * do * dl

    def matrix_back_propagation(self, X, Y, delta, iterations = 100, animate = False):
        """
        Back propagation algorithm for a set of points.
        """
        if animate: 
            fig, boundary,x_axis = self.__init__animation__(X,Y)

        #TODO: fix GD computation, beacuse it is not working now
        dl = delta
        dw = X
        n = len(X)
        for _ in range(iterations):
            Z = self.matrix_linear(X)
            do = self.activation.derivative(Z)
            new_w = self.weights - self.gamma *  np.dot(dw.T, do * dl) / n
            new_b = self.bias - self.gamma * np.sum(do * dl) / n
            self.weights = new_w
            self.bias = new_b
            if animate:
                self.__animate__(fig, boundary,x_axis)

    def fit(self, X, Y, animate=False, matrix = False):
        """
        Model fitting. 
        If animate is set to true, then it will render there results on a plot (if 2D).
        If matrix is set to true, then it will use the matrix formula.
        """
        if matrix:
            self.__matrix_fit__(X,Y, animate=animate)
        else: 
            self.__fit__(X,Y,animate=animate)

    def __matrix_fit__(self, X, Y, animate = False):
        """
        Matrix fit.
        """
        Y_pred = self.matrix_feed(X)
        delta = self.loss_function.derivative( np.sign(Y_pred), Y )
        self.matrix_back_propagation(X, Y, delta, animate=animate)

    def __fit__(self, X, Y, animate=False):
        """
        Fit point per point.
        """
        if animate: 
            fig,boundary, x_axis = self.__init__animation__(X, Y)

        for (x, y) in zip(X,Y):
            y_p = self.feed(x)
            delta = self.loss_function.derivative( np.sign(y_p), y ) 
            self.back_propagate(x, delta)

            if animate:
                self.__animate__(fig,boundary,x_axis)
    
    def __init__animation__(self,X,Y):
        """
        Setup for animating the fit.
        """
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.scatter(X[:,0], X[:,1], c=Y)
        x_axis = np.linspace(np.min(X[:,0]), np.max(X[:,1]), 100)
        line = -(self.weights[0] / self.weights[1]) * x_axis - self.bias / self.weights[1]
        boundary, = ax.plot(x_axis, line )
        plt.show(block=False)
        return fig, boundary, x_axis

    def __animate__(self, fig, boundary, x_axis):
        """
        Draw the new decision boundary of this single neuron.
        """
        line = -(self.weights[1] / self.weights[0]) * x_axis - self.bias / self.weights[1]
        boundary.set_ydata(line)
        fig.canvas.draw()
        fig.canvas.flush_events()

    def predict(self, X):
        """
        Given a set of points, it returns the prediction for that points.
        The model needs to be fit first.
        """
        y_pred = []
        print(self.weights, self.bias)
        for x in X:
            y = self.feed(x)
            y_pred.append( np.sign(y) )
        return np.array(y_pred)

        
                
        




    