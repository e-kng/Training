import pandas as pd 
from lazypredict.Supervised import LazyClassifier

train = pd.read_csv('Data/train.csv', sep=',')
X_train = train.drop(['subject', 'Activity'], axis=1).values
y_train = train['Activity'].values

test = pd.read_csv('Data/test.csv', sep=',')
X_test = test.drop(['subject', 'Activity'], axis=1).values
y_test = test['Activity']

clf = LazyClassifier(verbose=0, ignore_warnings=True)
models, predictions = clf.fit(X_train, X_test, y_train, y_test)
models