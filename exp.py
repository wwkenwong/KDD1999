import pandas as pd  
import numpy as np  
from matplotlib import pyplot as plt
#from sklearn.linear_model import LogisticRegression
#from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.svm import SVC 
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
#from sklearn.decomposition import PCA
#pca=PCA(n_components=2)  
#41 is the ans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import  zero_one_loss
from datetime import datetime
import time

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
    x[41], uniques=pd.factorize(x[41])
    
    
    retx=np.array(x)
    rety=np.array(x[41])
    
    return retx,rety
print("loading")
train_x,train_y=processdata(train_X)

testx,testy=processdata(test_x)

print("predicting")
alg=KNeighborsClassifier(n_jobs =-1)
alg.fit(train_x,train_y)
pred_y=alg.predict(testx)

#pred_y[pred_y==5]=4
#pred_y[pred_y==17]=12

error = zero_one_loss(testy, pred_y)



print("error of Kneighbour ",error)
#========================================================================



rf = RandomForestClassifier()
rf.fit(train_x,train_y)
pred_y=rf.predict(testx)

#pred_y[pred_y==5]=4
#pred_y[pred_y==17]=12

error = zero_one_loss(testy, pred_y)


print("rf ",error)

timetime=(time.time() - start_time)

print("--- %s seconds ---" % (time.time() - start_time))


