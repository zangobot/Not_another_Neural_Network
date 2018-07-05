import nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import StratifiedShuffleSplit

features = 5
# X,y = make_classification(n_samples=500, n_classes=2, n_features=2, n_informative=2, n_redundant=0, n_repeated=0)
X,y = make_blobs(n_samples=3000, centers=2, n_features=features)
y[y==0] = -1
s = StratifiedShuffleSplit(test_size=0.05)
for tr, ts in s.split(X,y):
    X_tr, X_ts = X[tr], X[ts]
    y_tr, y_ts = y[tr], y[ts]

net = nn.ShallowNN(input_dimension=features, gamma=1e-2)
net.fit(X_tr, y_tr, animate=False, matrix=True)
y_pred = net.predict(X_ts).ravel()
print(y_pred,'\n',y_ts)
print( 'Accuracy of the model is: ', np.sum(y_pred == y_ts) / len(y_ts) * 100 )
