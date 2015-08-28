from sklearn.datasets import fetch_mldata
from sklearn import svm
from sklearn.cross_validation import train_test_split
import numpy as np
from collections import Counter
import cPickle
import gzip
def oneclass_svm_baseline():
  #mnist = fetch_mldata('MNIST original', data_home='../ScikitData')
  #Xtr,Xts,Ytr,Yts = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=42)
  f = gzip.open('../ScikitData/mnist.pkl.gz','rb')
  training_data,validation_data,test_data=cPickle.load(f)
  f.close()
  # build a one class svm model
  model = svm.OneClassSVM(nu=0.1,kernel='rbf',gamma=0.1)
  model.fit( training_data[0][training_data[1]==0] )    
  # make prediction
  for i in range(10):
    predictions = [int(a) for a in model.predict(test_data[0][test_data[1]==i])]
    num_corr = sum(int(a==1) for a in predictions)
    print "One class SVM trained on digit 0 to predict %d   " % i,
    if i==0:
      print "%d of %d values correct." % (num_corr, len(predictions))
    else:
      print "%d of %d values correct." % (len(predictions)-num_corr, len(predictions))
  pass
if __name__ == '__main__':
  oneclass_svm_baseline()
