import sklearn, nltk
from sklearn import svm
import numpy as np
import sys

fp=open('connect-4.data', 'r')
data=fp.split()
fp.close()

#Vectorize data.
Game=[]
Outcome=[]
for x in data:
	y = x.split(',')
	if y[-1] == 'win':
		Outcome.append(1)
	elif y[-1] == 'loss':
		Outcome.append(-1)
	elif y[-1] == 'draw':
		Outcome.append(0)
	y = y[:-1]
	temp=[]
	for stuff in y:
		if stuff == 'x':
			temp.append(1)
		elif stuff == 'o':
			temp.append(-1)
		elif stuff == 'b':
			temp.append(0)
	Game.append(temp)

#Vectorization of data complete! Game contains feature vectors and Outcome contains the outcomes

trainwin=[]
trainloss=[]
traindraw=[]
