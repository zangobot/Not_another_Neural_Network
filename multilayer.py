from func import relu_func, sq_loss_func, sigmoid_func
from nn import SingleNeuronNN
import numpy as np

class MultilayerNN():
    def __init__(self, input_dim, n_layers, neurons_per_layers, act_func = relu_func, out_func=sigmoid_func, loss_func=sq_loss_func, gamma=1e-3):
        self.input_layer    = [ SingleNeuronNN(input_dim, loss_func, act_func, gamma=gamma) for _ in range(neurons_per_layers) ]
        self.hidden_layers   = [ [ SingleNeuronNN(neurons_per_layers, loss_func, act_func, gamma=gamma) for _ in range(neurons_per_layers) ] for _ in range(n_layers)]
        self.output_layer   = SingleNeuronNN(neurons_per_layers, loss_func, out_func)

    def fit(self, X, y):
        pass

    def feed(self, X):
        phiX = np.array( l.matrix_feed(X) for l in self.input_layer )
        for layer in self.hidden_layers:
            phiX = np.array( l.matrix_feed(phiX) for l in layer )
        output = self.output_layer.matrix_feed(phiX)
        return output

    def predict(self, X):
        output = self.feed(X)
        return np.sign(output)

    def __feed_forward__(self, X):
       pass