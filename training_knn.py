import pandas as pd  
import numpy as np  
import pylab
from matplotlib import pyplot as plt
#from array import array
#from sklearn.linear_model import LogisticRegression
#from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
#from sklearn import svm
#from sklearn.svm import SVC 
#from sklearn.cluster import KMeans
#from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
#41 is the ans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  zero_one_loss
import time
from sklearn.externals import joblib
f = open('result_summary.txt', 'w', encoding = 'UTF-8')
knn_result_txt = open('knnresult.txt', 'w', encoding = 'UTF-8')
rf_result_txt = open('rfresult.txt', 'w', encoding = 'UTF-8')

from sklearn.ensemble import IsolationForest
ilf = IsolationForest(n_estimators=100,
                      n_jobs=-1,          
                      verbose=2,)

#========================================================================
start_time = time.time()


train_X = pd.read_csv("kddcup.data_10_percent_corrected", header=None)
test_x = pd.read_csv("test_data+ans", header=None)


def processdata(x):
    
    x=x.sample(n=10000)
    #num_examples=100000
    #x = x[0:num_examples]
    #Encode input values as an enumerated type or categorical variable
    
    x[1], uniques=pd.factorize(x[1])
    x[2], uniques=pd.factorize(x[2])
    x[3], uniques=pd.factorize(x[3])



    x[41] = x[41].map( {'normal.': 0, 'snmpgetattack.': 2,'mailbomb.':1,'snmpguess.':2,'mscan.':3,'apache2.':1,'processtable.':1,'saint.':3,'httptunnel.':4,'sendmail.':2,'named.':2,'ps.':4,'xterm.':4,'xlock.':2,'xsnoop.':2,'worm.':2,'sqlattack.':4,'udpstorm.':1,
               'back.':1,'buffer_overflow.':4,'ftp_write.':2,'guess_passwd.':2,'imap.':2,'ipsweep.':3,'land.':1,'loadmodule.':4,'multihop.':2,'neptune.':1,'nmap.':3,'perl.':4,'phf.':2,'pod.':1,'portsweep.':3,'rootkit.':4,'satan.':3,'smurf.':1,'spy.':2,'teardrop.':1,'warezclient.':2,'warezmaster.':2} ).astype(int)
    
##0	1  2   3     4	
#normal dos r2l probe u2r     
    
    rety=np.array(x[41])
    x=x.drop(41,1)
    retx=np.array(x)
    
    return retx,rety
print("loading")
#f.write("loading\n")
train_x,train_y=processdata(train_X)

testx,testy=processdata(test_x)
##===============================================
train_x=train_x.astype(np.int32)
testx=testx.astype(np.int32)

save_train_x=train_x
save_testx=testx

for i in range(1):
    pca=PCA(n_components=9)  
    train_x=pca.fit_transform(save_train_x)
    testx=pca.transform(save_testx)

    #print("using ",i," PCA")
    #f.write('using '+str(i)+' PCA\n')    
    
    print("predicting")
    f.write("predicting\n")
    alg=KNeighborsClassifier(n_jobs =-1)
    alg.fit(train_x,train_y)
    pred_y=alg.predict(testx)
    
    
    plt.figure()
  
    
    # Plot also the training points
    plt.scatter(testx[:,0], testx[:,1],c=testy,cmap=pylab.cm.hot )
   
    plt.colorbar()
   
    plt.show()    
    
    
    error = zero_one_loss(testy, pred_y)
    kn_result=np.zeros((5,5))
    # i is testy j is predy
    for i in range(len(testy)):
        kn_result[testy[i],pred_y[i]]+=1
        
    for i in range(0,5):
        for j in range(0,5):
            knn_result_txt.write(str(kn_result[i,j])+' ')
        knn_result_txt.write('\n')
        
    knn_result_txt.close()
        
    joblib.dump(alg,'knn_model.pkl',compress=3)
    
    print("error of Kneighbour ",error)
    print("stat of knn :")
    print(kn_result)
    
    
    f.write('error of Kneighbour :'+str(error)+'\n')
    #========================================================================
 
    rf = RandomForestClassifier()
    rf.fit(train_x,train_y)
    pred_y=rf.predict(testx)
    
    #pred_y[pred_y==5]=4
    #pred_y[pred_y==17]=12
    
    error = zero_one_loss(testy, pred_y)
    rf_result=np.zeros((5,5))
    # i is testy j is predy
    for i in range(len(testy)):
        rf_result[testy[i],pred_y[i]]+=1   
    for i in range(0,5):
        for j in range(0,5):
            rf_result_txt.write(str(rf_result[i,j])+' ')
        rf_result_txt.write('\n')
        
    rf_result_txt.close()
    joblib.dump(rf,'randomforest.pkl',compress=3) 
    print("rf ",error)
    
    print("stat of rf :")
    print(rf_result)
    
    f.write('rf :'+str(error)+'\n')
    
    ##clf = joblib.load('../../data/model/randomforest.pkl')
    timetime=(time.time() - start_time)
    
    print("--- %s seconds ---" % (time.time() - start_time))

f.write('--- seconds --- :'+str(time.time() - start_time)+'\n')
f.close()
