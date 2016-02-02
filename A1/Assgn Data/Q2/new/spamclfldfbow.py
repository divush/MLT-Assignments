import sklearn, nltk
from sklearn import feature_extraction
from sklearn.naive_bayes import MultinomialNB
from nltk import stem
from nltk.stem import WordNetLemmatizer
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
fp=open('stopwords_list', 'r')
stopwords=fp.read().strip().split()
fp.close()

lmtzr=WordNetLemmatizer()
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
		#Lemmatizing..........................................................
		filetext=f.read().strip()
		templist=filetext.split()
		for x in range(len(templist)):
			templist[x] = lmtzr.lemmatize(templist[x])
		filetext=' '.join(templist)
		#End Lemmatizing.......................................................
		data.append(filetext)
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
	#Lemmatizing..........................................................
	filetext=f.read().strip()
	templist=filetext.split()
	for x in range(len(templist)):
		templist[x] = lmtzr.lemmatize(templist[x])
	filetext=' '.join(templist)
	#End Lemmatizing........
	testdata.append(filetext)
	f.close()
	
testvec=CV.transform(testdata)
testspam=clf.predict(testvec)
scor=clf.score(testvec, actualspam)
print(testspam)
print("score = "+str(scor))	#printing correct/total
fn=0
fp=0
tp=0
tn=0
for x in range(len(testspam)):
	if actualspam[x]==1 and testspam[x]==1:			#spam marked as spam
		tn=tn+1
	if actualspam[x]==0 and testspam[x]==0:			#non-spam marked as non-spam
		tp=tp+1
	if actualspam[x]==1 and testspam[x]==0:			#spam marked as non-spam. False Positive.
		fp=fp+1
	if actualspam[x]==0 and testspam[x]==1:			#non-spam marked as spam. False Negative.
		fn=fn+1

precision=tp/(tp+fp)
recall = tp/(tp+fn)
print("Precision = "+str(precision))
print("Recall = "+str(recall))