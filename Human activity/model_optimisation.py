import pandas as pd 
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import GridSearchCV, ParameterGrid

# Import data
train = pd.read_csv('Data/train.csv', sep=',')
X_train = train.drop(['subject', 'Activity'], axis=1).values
y_train = train['Activity'].values

test = pd.read_csv('Data/test.csv', sep=',')
X_test = test.drop(['subject', 'Activity'], axis=1).values
y_test = test['Activity']

# LinearSVC
# Select hyperparameters
parameters = {'penalty':['l2'], 'C': np.logspace(-3,3,7)}
svc = LinearSVC()
clf1 = GridSearchCV(svc, parameters)
clf1.fit(X_train, y_train)
y_pred1 = clf1.predict(X_test)

# Linear Discriminant Analysis
parameters = {'solver':('svd', 'eigen'), 'shrinkage':(None,'auto')}
lda = LinearDiscriminantAnalysis()
clf2 = GridSearchCV(lda, parameters)
clf2.fit(X_train, y_train)
y_pred2 = clf2.predict(X_test)

# RidgeClassifier
parameters = {'alpha': np.logspace(-3,3,7)}
rclf = RidgeClassifier(normalize=True)
clf3 = GridSearchCV(rclf, parameters)
clf3.fit(X_train, y_train)
y_pred3 = clf3.predict(X_test)

# Metrics
print('LinearSVC - params :', clf1.best_params_, '- score :', round(clf1.score(X_test, y_test),3),
      '\nLDA - params :', clf2.best_params_, '- score :', round(clf2.score(X_test, y_test),3),
      '\nRidgeClf - params :', clf3.best_params_, '- score :', round(clf3.score(X_test, y_test),3))


# Confusion matrix for LDA
pd.crosstab(y_pred2, y_test)