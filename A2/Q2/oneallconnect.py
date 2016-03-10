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


clf1 = svm.LinearSVC(loss='hinge')
clf2 = svm.LinearSVC(loss='hinge')
clf3 = svm.LinearSVC(loss='hinge')
#create learning sets for 3 classifiers
for vpart in range(5):
	# print("Validating part "+str(vpart))
	#win, loss classifier
	g1=[]
	o1=[]
	#loss, draw classifier
	g2=[]
	o2=[]
	#win, draw classifier
	g3=[]
	o3=[]

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
				if O[i] == 1 or O[i] == -1:
					g1.append(G[i])
					o1.append(O[i])
				if O[i] == -1 or O[i] == 0:
					g2.append(G[i])
					o2.append(O[i])
				if O[i] == 1 or O[i] == 0:
					g3.append(G[i])
					o3.append(O[i])
	# print(len(g1))
	clf1.fit(g1,o1)
	clf2.fit(g2,o2)
	clf3.fit(g3,o3)
	Pred1 = clf1.predict(GTest)
	Pred2 = clf2.predict(GTest)
	Pred3 = clf3.predict(GTest)
	for k in range(len(Pred1)):
		cw,cl,cd=0,0,0
		l=-2
		if Pred1[k]==1:
			cw = cw + 1
		if Pred1[k]==-1:
			cl = cl + 1
		if Pred2[k]==0:
			cd = cd + 1
		if Pred2[k]==-1:
			cl = cl + 1
		if Pred3[k]==1:
			cw = cw + 1
		if Pred3[k]==0:
			cd = cd + 1
		m=max(cw, cd, cl)
		if m==cw:
			Pred.append(1)
		elif m==cd:
			Pred.append(0)
		elif m==cl:
			Pred.append(-1)

	accuracy = accuracy_score(OTest, Pred)
	print(accuracy)
