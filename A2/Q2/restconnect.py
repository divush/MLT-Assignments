import sklearn, nltk
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import sys
import copy
fp=open('connect-4.data', 'r')
data=fp.read().strip().split()
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
		elif stuff == 'b':
			temp.append(0)
		elif stuff == 'o':
			temp.append(-1)
	Game.append(temp)

#Vectorization of data complete! Game contains feature vectors and Outcome contains the outcomes
VGame=[[],[],[],[],[],[]]
VOutcome=[[],[],[],[],[],[]]
l = len(Game)
vl=int(l/5)+1
for i in range(l):
	t=int(i/vl)
	x=Game[i]
	y=Outcome[i]
	VGame[t].append(x)
	VOutcome[t].append(y)

clfw = svm.LinearSVC(loss='hinge')
clfd = svm.LinearSVC(loss='hinge')
clfl = svm.LinearSVC(loss='hinge')

#create learning sets for 3 classifiers
for vpart in range(5):
	# print("Validating part "+str(vpart))
	#win classifier
	gw=[]
	ow=[]
	#draw classifier
	gd=[]
	od=[]
	#loss classifier
	gl=[]
	ol=[]

	#Extract Test data
	Pred=[]
	OTest=copy.deepcopy(VOutcome[vpart])
	GTest=copy.deepcopy(VGame[vpart])
	#Create Training sets for 3 classifiers!
	for z in range(5):
		# print("z = "+str(z))
		if z == vpart:
			continue
		else:
			# print(len(VGame[z]))
			O=copy.deepcopy(VOutcome[z])
			G=copy.deepcopy(VGame[z])
			# print(str(len(VGame[z]))+" "+str(len(G)))
			for i in range(len(O)):
				if O[i] == 1:
					gw.append(G[i])
					gd.append(G[i])
					gl.append(G[i])
					ow.append(1)
					od.append(0)
					ol.append(0)
				if O[i] == 0:
					gw.append(G[i])
					gd.append(G[i])
					gl.append(G[i])
					ow.append(0)
					od.append(1)
					ol.append(0)
				if O[i] == -1:
					gw.append(G[i])
					gd.append(G[i])
					gl.append(G[i])
					ow.append(0)
					od.append(0)
					ol.append(1)
	clfw.fit(gw,ow)
	clfd.fit(gd,od)
	clfl.fit(gl,ol)
	Pred1 = clfw.decision_function(GTest)
	Pred2 = clfl.decision_function(GTest)
	Pred3 = clfd.decision_function(GTest)
	temparr.append(Pred1)
	for x in range(len(Pred1)):
		temparr=[]
		temparr.append(Pred1[x])
		temparr.append(Pred2[x])
		temparr.append(Pred3[x])
		# print(temparr)
		maxarg = np.argmax(temparr)
		if maxarg==0:
			Pred.append(1)
		if maxarg==1:
			Pred.append(0)
		if maxarg==2:
			Pred.append(-1)

	accuracy = accuracy_score(OTest, Pred)
	print(accuracy)
