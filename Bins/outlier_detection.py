'''
Outlier detection with one-class SVMs and robust covariance estimation.
One-class SVM is designed for novelty detection.
Robust covariance estimation is designed for outlier detection.
Code is modify from example in scikit-learn http://scikit-learn.org/stable/_downloads/plot_outlier_detection.py
Copyright @ Hongyu Su (hongyu.su@me.com)
'''

print (__doc__)
# scikit-learn package
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
# numpy
import numpy as np
# plot
import matplotlib.pyplot as plt
import matplotlib.font_manager
from scipy import stats

n_samples=400
outliers_fraction=0.25
clusters_separation=[0,1,2,3]

# define classifier
classifiers = {'one class SVM': svm.OneClassSVM(nu=0.95*outliers_fraction+0.05,kernel='rbf',gamma=0.1),
'covariance estimation':EllipticEnvelope(contamination=0.1)}

xx, yy = np.meshgrid(np.linspace(-7, 7, 500), np.linspace(-7, 7, 500))
n_inliers = int((1. - outliers_fraction) * n_samples)
n_outliers = int(outliers_fraction * n_samples)
ground_truth = np.ones(n_samples, dtype=int)
ground_truth[-n_outliers:] = 0


# Fit the problem with varying cluster separation
plt.figure(figsize=(20,10))

for i, offset in enumerate(clusters_separation):
    np.random.seed(42)
    # generate inlier, gaussian
    x = 0.3 * np.random.randn(0.25 * n_inliers, 1) - offset
    y = 0.3 * np.random.randn(0.25 * n_inliers, 1) - offset
    X1 = np.r_[x,y].reshape((2,x.shape[0])).T
    x = 0.3 * np.random.randn(0.25 * n_inliers, 1) - offset
    y = 0.3 * np.random.randn(0.25 * n_inliers, 1) + offset
    X2 = np.r_[x,y].reshape((2,x.shape[0])).T
    x = 0.3 * np.random.randn(0.25 * n_inliers, 1) + offset
    y = 0.3 * np.random.randn(0.25 * n_inliers, 1) - offset
    X3 = np.r_[x,y].reshape((2,x.shape[0])).T
    x = 0.3 * np.random.randn(0.25 * n_inliers, 1) + offset
    y = 0.3 * np.random.randn(0.25 * n_inliers, 1) + offset
    X4 = np.r_[x,y].reshape((2,x.shape[0])).T
    X = np.r_[X1, X2, X3, X4]

    # generate outlier, uniform
    X = np.r_[X, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]

    # Fit the model with the One-Class SVM
    for j, (clf_name, clf) in enumerate(classifiers.items()):
        # fit the data and tag outliers
        clf.fit(X)
        y_pred = clf.decision_function(X).ravel()
        threshold = stats.scoreatpercentile(y_pred,100 * outliers_fraction)
        y_pred = y_pred > threshold
        n_errors = (y_pred != ground_truth).sum()
        # plot the levels lines and the points
        Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        subplot = plt.subplot(2, 4, (j)*4+i+1)
        subplot.set_title("Outlier detection")
        subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),cmap=plt.cm.Blues_r)
        a = subplot.contour(xx, yy, Z, levels=[threshold],linewidths=2, colors='red')
        subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],colors='orange')
        b = subplot.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], c='white')
        c = subplot.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], c='black')
        subplot.axis('tight')
        subplot.legend(
            [a.collections[0], b, c],
            ['learned decision function', 'true inliers', 'true outliers'],
            prop=matplotlib.font_manager.FontProperties(size=8))
        subplot.set_xlabel("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))
        subplot.set_xlim((-7, 7))
        subplot.set_ylim((-7, 7))
    plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
plt.show()
