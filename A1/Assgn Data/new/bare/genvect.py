import sklearn
from sklearn import feature_extraction

import sys
folder=str(sys.argv[1])
folder_number=folder[-1]
if folder[-1] == '0':
	folder_number='10'
namefile='p'+folder_number+'flist'
flist=open(namefile, 'r')
files=flist.read().split()
flist.close()
spam=[]		#Array stores 1 for spam and 0 for non-spam

filedata=[]
for fl in files:
	if 'spm' in fl:
		spam.append(1)
	else:
		spam.append(0)
	f=open(folder+'/'+fl, 'r')
	filedata.append(f.read().strip())
	f.close()

CV=feature_extraction.text.CountVectorizer()
vec=CV.fit_transform(filedata).toarray()
LS=[]
for x in range(len(vec)):
	temp=[vec[x], spam[x]]
	LS.append(temp)

ofname='part'+folder_number+'vec'
of=open(ofname, 'w')
for x in range(len(vec)):
	of.write(str(vec[x].tolist())+'\n')
of.close()

ofname='part'+folder_number+'labels'
of=open(ofname, 'w')
for x in range(len(spam)):
	of.write(str(spam[x])+'\n')
of.close()