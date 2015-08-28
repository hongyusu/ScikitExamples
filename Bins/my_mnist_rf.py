




from sklearn.datasets import fetch_mldata
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import numpy as np
from collections import Counter
import cPickle
import gzip

def rf_baseline():
  #mnist = fetch_mldata('MNIST original', data_home='../ScikitData')
  #Xtr,Xts,Ytr,Yts = train_test_split(mnist.data, mnist.target, test_size=10000, random_state=42)
  f = gzip.open('../ScikitData/mnist.pkl.gz','rb')
  training_data,validation_data,test_data=cPickle.load(f)
  f.close()
  model = RandomForestClassifier(n_estimators=1000, max_depth=None, min_samples_split=1, random_state=0)
  model.fit(training_data[0],training_data[1])
  predictions = [int(a) for a in model.predict(test_data[0])]
  num_corr = sum(int(a==y) for a,y in zip(predictions,test_data[1]))
  print "Baseline classifier using an Random Forest."
  print "%s of %s values correct." % (num_corr, len(test_data[1]))
  pass

if __name__ == '__main__':
  rf_baseline()
