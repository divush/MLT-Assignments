import sklearn
from sklearn import feature_extraction
from sklearn.naive_bayes import MultinomialNB

import sys
partlist=['part1', 'part2', 'part3', 'part4', 'part5', 'part6', 'part7', 'part8', 'part9', 'part10']
test=str(sys.argv[1])
folder_validate=int(test[-1])
if folder_validate == 0:
	folder_validate=10
rmstr='part'+str(folder_validate)
#plist contains the folders for testing!
plist=partlist
plist.remove(rmstr)
data=[]
spam=[]

for foldername in plist:
	folder_number=int(foldername[4:])
	namefile='p'+str(folder_number)+'flist'
	listfile=open(namefile, 'r')
	flist=listfile.read().split()	#list of file names
	listfile.close()

	for fl in flist:
		# print(fl)
		if 'spm' in fl:
			spam.append(1)
		else:
			spam.append(0)
		fname=foldername+'/'+fl
		# print(fname)
		f=open(fname, 'r')
		data.append(f.read().strip())
		f.close()

CV=feature_extraction.text.CountVectorizer(binary=True)
vec=CV.fit_transform(data)
clf=MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
clf.fit(vec, spam)

testfolder='part'+str(folder_validate)
tflist='p'+str(folder_validate)+'flist'
tf=open(tflist, 'r')
testflist=tf.read().split()
tf.close()
testdata=[]
testspam=[]
actualspam=[]
# print(testflist)
for fl in testflist:
	# print(fl)
	if 'spm' in fl:
		actualspam.append(1)
	else:
		actualspam.append(0)
	fname=testfolder+'/'+fl
	# print(fname)
	f=open(fname, 'r')
	tempstring=f.read().strip()
	testdata.append(tempstring)
	
testvec=CV.transform(testdata)
testspam=clf.predict(testvec)
scor=clf.score(testvec, actualspam)
print(testspam)
print(scor)
# count=0
# for x in range(len(testspam)):
# 	if testspam[x] == actualspam[x]:
# 		count = count + 1

# print("accuracy = "+str(float(count)/len(testspam)))