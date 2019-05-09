#!/usr/bin/python

import matplotlib.pyplot as plt
from prep_terrain_data import makeTerrainData
from class_vis import prettyPicture

features_train, labels_train, features_test, labels_test = makeTerrainData()


### the training data (features_train, labels_train) have both "fast" and "slow"
### points mixed together--separate them so we can give them different colors
### in the scatterplot and identify them visually
grade_fast = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==0]
bumpy_fast = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==0]
grade_slow = [features_train[ii][0] for ii in range(0, len(features_train)) if labels_train[ii]==1]
bumpy_slow = [features_train[ii][1] for ii in range(0, len(features_train)) if labels_train[ii]==1]


#### initial visualization
#plt.xlim(0.0, 1.0)
#plt.ylim(0.0, 1.0)
#plt.scatter(bumpy_fast, grade_fast, color = "b", label="fast")
#plt.scatter(grade_slow, bumpy_slow, color = "r", label="slow")
#plt.legend()
#plt.xlabel("bumpiness")
#plt.ylabel("grade")
#plt.show()
################################################################################


### your code here!  name your classifier object clf if you want the 
### visualization code (prettyPicture) to show you the decision boundary

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=8)
clf.fit(features_train,labels_train)
pred = clf.predict(features_test) 
from sklearn import metrics
acc = metrics.accuracy_score(pred , labels_test)
print "ANALYSIS USING K NEAREST NEIGHBOURS ALGO"
print(acc)

try:
    prettyPicture(clf, features_test, labels_test)
except NameError:
    pass

print "ANALYSIS USING NAIVE BAYES ALGO\n"
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
clf1 = GaussianNB()
clf1.fit(features_train , labels_train) 
pred1 = clf1.predict(features_test) 
acc1 = metrics.accuracy_score(pred1 , labels_test)
print(acc1)
try:
    prettyPicture(clf1, features_test, labels_test)
except NameError:
    pass

from sklearn.svm import SVC 
from sklearn import metrics
print "ANLAYSIS USING SVM ALGO\n"
clf2 = SVC(C = 10000 , kernel = "rbf" , gamma = "auto") 
clf2.fit(features_train , labels_train)
pred2 = clf2.predict(features_test) 
acc2 = metrics.accuracy_score(pred2 , labels_test) 
print(acc2) 
try:
    prettyPicture(clf1, features_test, labels_test)
except NameError:
    pass