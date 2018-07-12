from func import relu_func, sq_loss_func, sigmoid_func
from nn import SingleNeuronNN
import numpy as np

class MultilayerNN():
    def __init__(self, input_dim, n_layers, neurons_per_layers, act_func = relu_func, out_func=sigmoid_func, loss_func=sq_loss_func, gamma=1e-3):
        self.input_layer = np.random.randn(input_dim, neurons_per_layers)
        self.hidden_layer = [np.random.randn(neurons_per_layers, neurons_per_layers) for _ in range(n_layers)]
        self.output_layer = np.random.randn(neurons_per_layers,1)
        self.gamma = gamma
        self.activation = act_func
        self.output = out_func
        self.loss = loss_func

    def fit(self, X, y):
        pass

    def feed(self, X):
        pass

    def __feed_forward__(self, X):
        calc = np.dot(X, self.input_layer)
        for l in self.hidden_layer:
            calc = np.dot(l.T, calc)
            print(calc.shape)
        calc = np.dot(calc, self.output_layer)