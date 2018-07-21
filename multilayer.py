from func import relu_func, sq_loss_func, sigmoid_func
from nn import SingleNeuronNN
import numpy as np

EPSILON = 1e-6

class MultilayerNN():
    def __init__(self, input_dim, n_layers, neurons_per_layers, act_func = relu_func, out_func=sigmoid_func, loss_func=sq_loss_func, gamma=1e-3):
        self.input_layer    = [ SingleNeuronNN(input_dim, loss_func, act_func, gamma=gamma) for _ in range(neurons_per_layers) ]
        self.hidden_layers  = [ [ SingleNeuronNN(neurons_per_layers, loss_func, act_func, gamma=gamma) for _ in range(neurons_per_layers) ] for _ in range(n_layers)]
        self.output_layer   = SingleNeuronNN(neurons_per_layers, loss_func, out_func)
        self.activation = act_func
        self.output = out_func
        self.loss_func = loss_func
        self.iterations = 1000
        self.condition = lambda w1,w2 : np.abs(np.linalg.norm(w1) - np.linalg.norm(w2)) < EPSILON

    def fit(self, X, Y):
        self.back_propagate(X,Y)


    def feed(self, X):
        phiX = np.array( l.matrix_feed(X) for l in self.input_layer )
        for layer in self.hidden_layers:
            phiX = np.array( l.matrix_feed(phiX) for l in layer )
        output = self.output_layer.matrix_feed(phiX)
        return output

    def predict(self, X):
        output = self.feed(X)
        return np.sign(output)

    def back_propagate(self, X, Y):
        y_pred = self.predict(X)
        delta_0 = self.loss_func.derivative(y_pred, Y)
        self.output_layer.matrix_back_propagation(self.output_layer.__last_input__,None, delta_0, iterations=self.iterations,animate=False)
        delta_i = delta_0
        for layer in self.hidden_layers[::-1]:
            deltas = []
            for l in layer:
                deltas.append(l.delta_matrix(delta_i))
                l.matrix_back_propagation(l.__last_input__, None, delta_i, iterations=self.iterations, animate=False)



