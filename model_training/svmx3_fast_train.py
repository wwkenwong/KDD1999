import pandas as pd  
import numpy as np 
from matplotlib import pyplot as plt
import pylab
from sklearn import svm
from sklearn.metrics import  zero_one_loss
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
import time
start_time = time.time()
f = open('result_summary.txt', 'w', encoding = 'UTF-8')
#from sklearn.decomposition import PCA
from sklearn.decomposition import PCA

def loader_train_1_5(z):
    x=z
    x[1], uniques=pd.factorize(x[1])
    x[2], uniques=pd.factorize(x[2])
    x[3], uniques=pd.factorize(x[3])
    x[41] = x[41].map( {'normal.': 0, 'snmpgetattack.': 2,'mailbomb.':1,'snmpguess.':2,'mscan.':3,'apache2.':1,'processtable.':1,'saint.':3,'httptunnel.':4,'sendmail.':2,'named.':2,'ps.':4,'xterm.':4,'xlock.':2,'xsnoop.':2,'worm.':2,'sqlattack.':4,'udpstorm.':1,
               'back.':1,'buffer_overflow.':4,'ftp_write.':2,'guess_passwd.':2,'imap.':2,'ipsweep.':3,'land.':1,'loadmodule.':4,'multihop.':2,'neptune.':1,'nmap.':3,'perl.':4,'phf.':2,'pod.':1,'portsweep.':3,'rootkit.':4,'satan.':3,'smurf.':1,'spy.':2,'teardrop.':1,'warezclient.':2,'warezmaster.':2} ).astype(int)
    x=x[x[41]!=0]
    x=x[x[41]!=3]
    x[41]=x[41].map({1:1,2:5,4:5})
    rety=np.array(x[41])
    x=x.drop(41,1)
    retx=np.array(x) 
    return retx,rety   
def loader_train_0_5(z):
    x=z
    x[1], uniques=pd.factorize(x[1])
    x[2], uniques=pd.factorize(x[2])
    x[3], uniques=pd.factorize(x[3])
    x[41] = x[41].map( {'normal.': 0, 'snmpgetattack.': 2,'mailbomb.':1,'snmpguess.':2,'mscan.':3,'apache2.':1,'processtable.':1,'saint.':3,'httptunnel.':4,'sendmail.':2,'named.':2,'ps.':4,'xterm.':4,'xlock.':2,'xsnoop.':2,'worm.':2,'sqlattack.':4,'udpstorm.':1,
               'back.':1,'buffer_overflow.':4,'ftp_write.':2,'guess_passwd.':2,'imap.':2,'ipsweep.':3,'land.':1,'loadmodule.':4,'multihop.':2,'neptune.':1,'nmap.':3,'perl.':4,'phf.':2,'pod.':1,'portsweep.':3,'rootkit.':4,'satan.':3,'smurf.':1,'spy.':2,'teardrop.':1,'warezclient.':2,'warezmaster.':2} ).astype(int)
    x[41]=x[41].map({0:0,1:0,2:5,3:0,4:5})
    rety=np.array(x[41])
    x=x.drop(41,1)
    retx=np.array(x) 
    return retx,rety   
def loader_train_3_5(z):
    x=z
    x[1], uniques=pd.factorize(x[1])
    x[2], uniques=pd.factorize(x[2])
    x[3], uniques=pd.factorize(x[3])
    x[41] = x[41].map( {'normal.': 0, 'snmpgetattack.': 2,'mailbomb.':1,'snmpguess.':2,'mscan.':3,'apache2.':1,'processtable.':1,'saint.':3,'httptunnel.':4,'sendmail.':2,'named.':2,'ps.':4,'xterm.':4,'xlock.':2,'xsnoop.':2,'worm.':2,'sqlattack.':4,'udpstorm.':1,
               'back.':1,'buffer_overflow.':4,'ftp_write.':2,'guess_passwd.':2,'imap.':2,'ipsweep.':3,'land.':1,'loadmodule.':4,'multihop.':2,'neptune.':1,'nmap.':3,'perl.':4,'phf.':2,'pod.':1,'portsweep.':3,'rootkit.':4,'satan.':3,'smurf.':1,'spy.':2,'teardrop.':1,'warezclient.':2,'warezmaster.':2} ).astype(int)
    x=x[x[41]!=0]
    x=x[x[41]!=1]
    x[41]=x[41].map({3:3,2:5,4:5})
    rety=np.array(x[41])
    x=x.drop(41,1)
    retx=np.array(x) 
    return retx,rety   
def processdata(z):
    x=z
    x[1], uniques=pd.factorize(x[1])
    x[2], uniques=pd.factorize(x[2])
    x[3], uniques=pd.factorize(x[3])
    x[41] = x[41].map( {'normal.': 0, 'snmpgetattack.': 2,'mailbomb.':1,'snmpguess.':2,'mscan.':3,'apache2.':1,'processtable.':1,'saint.':3,'httptunnel.':4,'sendmail.':2,'named.':2,'ps.':4,'xterm.':4,'xlock.':2,'xsnoop.':2,'worm.':2,'sqlattack.':4,'udpstorm.':1,
               'back.':1,'buffer_overflow.':4,'ftp_write.':2,'guess_passwd.':2,'imap.':2,'ipsweep.':3,'land.':1,'loadmodule.':4,'multihop.':2,'neptune.':1,'nmap.':3,'perl.':4,'phf.':2,'pod.':1,'portsweep.':3,'rootkit.':4,'satan.':3,'smurf.':1,'spy.':2,'teardrop.':1,'warezclient.':2,'warezmaster.':2} ).astype(int)
    x[41]=x[41].map({0:0,1:1,2:5,3:3,4:5})
    rety=np.array(x[41])
    x=x.drop(41,1)
    retx=np.array(x)
    return retx,rety
print('loading')
print('PCA LOADED')
pca=PCA(n_components=1)  



#=======KNN 1 3 5=================
x = pd.read_csv("kddcup.data_10_percent_corrected", header=None)
test_x = pd.read_csv("corrected", header=None)

train_x,train_y=processdata(x)

testx,testy=processdata(test_x)
##===============================================
train_x=train_x.astype(np.int32)
testx=testx.astype(np.int32)

save_train_x=train_x
save_testx=testx

pca=PCA(n_components=9)  
train_x=pca.fit_transform(save_train_x)
testx=pca.transform(save_testx)
print("predicting")
f.write("predicting\n")
alg=KNeighborsClassifier(n_jobs =-1)
alg.fit(train_x,train_y)
pred_y=alg.predict(testx)
joblib.dump(alg,'alg.pkl',compress=3)
error_function=zero_one_loss(testy, pred_y)
print('knn error= ',error_function)
f.write('knn error= :'+str(error_function)+'\n')

#=======Train 0 to 5 svm=================
x = pd.read_csv("kddcup.data_10_percent_corrected", header=None)
test_x_ = pd.read_csv("corrected", header=None)
print('loaded')

train_x,train_y=loader_train_0_5(x)
test_x,test_y=loader_train_0_5(test_x_)
save_train_x=pca.fit_transform(train_x)
test__x=pca.transform(test_x)
print('training')
clf05 = svm.SVC()
clf05.fit(save_train_x,train_y)
pred_y=clf05.predict(test__x)
joblib.dump(clf05,'clf05.pkl',compress=3)
error_function=zero_one_loss(test_y, pred_y)
print('clf05 error= ',error_function)
f.write('clf05 error= :'+str(error_function)+'\n')

#=======Train 1 to 5 svm=================
x = pd.read_csv("kddcup.data_10_percent_corrected", header=None)
test_x_ = pd.read_csv("corrected", header=None)
train_x,train_y=loader_train_1_5(x)
test_x,test_y=loader_train_1_5(test_x_)
save_train_x=pca.fit_transform(train_x)
test__x=pca.transform(test_x)
print('training')
clf15 = svm.SVC()
clf15.fit(save_train_x,train_y)
pred_y=clf15.predict(test__x)
joblib.dump(clf15,'clf15.pkl',compress=3)
error_function=zero_one_loss(test_y, pred_y)
print('clf15 error= ',error_function)
f.write('clf15 error= :'+str(error_function)+'\n')

#=======Train 3 to 5 svm=================
x = pd.read_csv("kddcup.data_10_percent_corrected", header=None)
test_x_ = pd.read_csv("corrected", header=None)
train_x,train_y=loader_train_3_5(x)
test_x,test_y=loader_train_3_5(test_x_)
save_train_x=pca.fit_transform(train_x)
test__x=pca.transform(test_x)
clf35 = svm.SVC()
clf35.fit(save_train_x,train_y)
pred_y=clf35.predict(test__x)
joblib.dump(clf35,'clf35.pkl',compress=3)
error_function=zero_one_loss(test_y, pred_y)
print('clf35 error= ',error_function)
f.write('clf35 error= :'+str(error_function)+'\n')

f.write('--- seconds --- :'+str(time.time() - start_time)+'\n')
f.close()
