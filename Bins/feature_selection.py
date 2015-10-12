



"""
feature selection with scikit learn
copyright @ Hongyu Su (hongyu.su@me.com)
"""

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.ensemble import ExtraTreesClassifier
import itertools


# collect data for training and test
categories = ['alt.atheism','talk.religion.misc','comp.graphics','sci.space']
remove = ('headers','footers','quotes')
data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42, remove=remove)
data_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42, remove=remove)
y_train,y_test = data_train.target, data_test.target


vectorizationModels = (
    (HashingVectorizer(stop_words='english', non_negative=True,n_features=2**16),'Hashing'),
    (TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english'),'Tfidf'))

rfesvc = SVC(kernel="linear", C=1)
featureSelectionModels = (
    (SelectKBest(chi2, k=5),'Univariance'),
    (LinearSVC(C=0.1, penalty="l1", dual=False),'L1'),
    (ExtraTreesClassifier(),'Tree'),
    (RFE(estimator=rfesvc, n_features_to_select=10000, step=1),'RFE'))

classificationModels = (
    (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
    (Perceptron(n_iter=50), "Perceptron"),
    (KNeighborsClassifier(n_neighbors=10), "kNN"),
    (RandomForestClassifier(n_estimators=100), "Random forest"),
    (LinearSVC(loss='squared_hinge', penalty='l1', dual=False, tol=1e-3),"SVC-L1"),
    (LinearSVC(loss='squared_hinge', penalty='l2', dual=False, tol=1e-3),"SVC-L2"),
    (SGDClassifier(alpha=.0001, n_iter=50,penalty='l1'),'SGD-L1'),
    (SGDClassifier(alpha=.0001, n_iter=50,penalty='l2'),'SGD-L2'),
    (SGDClassifier(alpha=.0001, n_iter=50,penalty='elasticnet'),'SGD-ElasticNet'),
    (NearestCentroid(),'Nearest neighbor'),
    (MultinomialNB(alpha=.01),'NB1'),
    (BernoulliNB(alpha=.01),'NB2'))

def benchmark(classifier):
    classifier.fit(X_train,y_train)
    pred = classifier.predict(X_test)
    score = metrics.accuracy_score(y_test, pred)    
    return score

results = []
for (vectorizer,vectorizerName),(selector,selectorName),(classifier,classifierName) in list(itertools.product(vectorizationModels,featureSelectionModels,classificationModels)):
    # vectorize input
    if vectorizerName=='Hashing':
        X_train = vectorizer.transform(data_train.data)
    else:
        X_train = vectorizer.fit_transform(data_train.data)
    X_test = vectorizer.transform(data_test.data)
    print(X_train.shape)
    # feature selection
    X_train = selector.fit_transform(X_train,y_train)
    X_test = selector.transform(X_test)
    # classification model
    results.append((vectorizerName,selectorName,classifierName,benchmark(classifier)))


for x in results:
    print(x)



