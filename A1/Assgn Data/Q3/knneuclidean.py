import sklearn, sys, time
from sklearn.neighbors import KNeighborsClassifier
import numpy as numpy
from sklearn.datasets.mldata import fetch_mldata
t0 = time.clock()
print("Fetching dataset")
mnist = fetch_mldata('MNIST original')
X, y = mnist.data / 255., mnist.target
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]
print("Dataset fetch completed!")
print(X[0])
time_dataset= time.clock() - t0
print("Dataset created in time "+str(time_dataset))

# t0 = time.clock()
# # print(mnist.data.shape)
# neigh = KNeighborsClassifier(n_neighbors=3, metric="euclidean", n_jobs=2)
# neigh.fit(X_train, y_train)
# time_train= time.clock() - t0
# print("Dataset trained in time "+str(time_train))

# t0 = time.clock()
# y_pred = neigh.predict(X_test)
# time_pred = time.clock() - t0
# print("Predicted in time "+ str(time_pred))
# print(y_pred.tolist())
# score = 0
# for x in range(len(y_pred)):
# 	if y_pred[x] == y_test[x]:
# 		score = score + 1
# score = float(score)/len(y_pred)
# print("score = "+ str(score))