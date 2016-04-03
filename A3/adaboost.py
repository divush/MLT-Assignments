import sklearn, sys, time, skimage, os, struct
from sklearn import tree
from skimage import data, color, exposure
import numpy as np
from sklearn.datasets.mldata import fetch_mldata
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
from skimage.feature import hog
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def load_mnist(dataset="training", digits=np.arange(10), path="."):
    """
    Loads MNIST files into 3D numpy arrays

    Adapted from: http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
    """

    if dataset == "training":
        fname_img = os.path.join(path, 'mnist-train-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'mnist-train-labels-idx1-ubyte')
    elif dataset == "testing":
        fname_img = os.path.join(path, 'mnist-t10k-images-idx3-ubyte')
        fname_lbl = os.path.join(path, 'mnist-t10k-labels-idx1-ubyte')
    else:
        raise ValueError("dataset must be 'testing' or 'training'")

    flbl = open(fname_lbl, 'rb')
    magic_nr, size = struct.unpack(">II", flbl.read(8))
    lbl = pyarray("b", flbl.read())
    flbl.close()

    fimg = open(fname_img, 'rb')
    magic_nr, size, rows, cols = struct.unpack(">IIII", fimg.read(16))
    img = pyarray("B", fimg.read())
    fimg.close()
    if dataset== "training":
        size = 10000
    else:
        size = 1000
    ind = [ k for k in range(size) if lbl[k] in digits ]
    N = len(ind)

    images = zeros((N, rows, cols), dtype=uint8)
    labels = zeros((N, 1), dtype=int8)
    for i in range(len(ind)):
        images[i] = array(img[ ind[i]*rows*cols : (ind[i]+1)*rows*cols ]).reshape((rows, cols))
        labels[i] = lbl[ind[i]]

    return images, labels

digs= [0,1,2,3,4,5,6,7,8,9]
X_tr, y_train = load_mnist('training', digits=digs)
X_te, y_test = load_mnist('testing', digits=digs)
X_train=[]
X_test=[]
for img in X_tr:
	# print(len(img))
	# print(len(img[0]))
	temp2 = hog(img)
	X_train.append(temp2)
for img in X_te:
	temp2 = hog(img)
	X_test.append(temp2)
nt=500
clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=4),
                         algorithm="SAMME",
                         n_estimators=nt)

clf.fit(X_train, y_train)
test_out=clf.predict(X_test)
scor=clf.score(X_test, y_test)
print("number of n_estimators = "+str(nt)+ "\tscore = "+str(scor))