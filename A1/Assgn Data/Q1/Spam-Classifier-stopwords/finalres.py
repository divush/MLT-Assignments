import sys

root='res'
flist=[]
score=0
prec=0
recall=0
for x in range(1, 11):
	flist.append(root+str(x))
for x in flist:
	f=open(x, 'r')
	data=f.read().strip().split('\n')
	score = score + float((data[-3].split('='))[-1])
	prec = prec + float((data[-2].split('='))[-1])
	recall = recall + float((data[-1].split('='))[-1])
	f.close()
score=score/10
prec=prec/10
recall=recall/10
print(score)
print(prec)
print(recall)