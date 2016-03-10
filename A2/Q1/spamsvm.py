import sklearn, nltk
from sklearn import feature_extraction
from sklearn import svm, cross_validation
from nltk import stem
from nltk.stem import WordNetLemmatizer
import numpy as np
import sys
partlist=['part1', 'part2', 'part3', 'part4', 'part5', 'part6', 'part7', 'part8', 'part9', 'part10']
#plist contains the folders for testing!
plist=partlist
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
			spam.append(-1)
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
#At this point, data[] contains the vector of all messages , i.e., it is a list of message strings.
#Also, spam[] stores the true labels. This corresponds to Y in the training set.


TFV=feature_extraction.text.TfidfVectorizer(ngram_range=(1,1), stop_words=stopwords) #------------------------------------------------>The only difference!
#Note that binary is not True in prev!
tfidf_mat=TFV.fit_transform(data)
#Here tfidf_mat stores a vector corresponding to each message. Essentially the X in our learning set.


#Build and train classifier. Also, take prior probabilities into account while making calculations.
clf = svm.LinearSVC(loss='hinge')

accuracy = cross_validation.cross_val_score(clf, tfidf_mat, spam,cv=5,scoring='accuracy')
precision = cross_validation.cross_val_score(clf, tfidf_mat, spam,cv=5,scoring='precision')
recall = cross_validation.cross_val_score(clf, tfidf_mat, spam,cv=5,scoring='recall')

print("Accuracy = "+str(accuracy))
print("Precision = "+str(precision))
print("Recall = "+str(recall))

print("Average Accuracy = "+str(np.mean(accuracy)))
print("Average Precision = "+str(np.mean(precision)))
print("Average Recall = "+str(np.mean(recall)))