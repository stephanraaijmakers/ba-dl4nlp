import numpy as np
from sklearn.model_selection import KFold



X=np.array([[1],[2],[3],[4]])
y=np.array([0,1,1,0])

kf = KFold(n_splits=2, shuffle=True)

for train_index, test_index in kf.split(X):
     X_train=X[train_index]
     X_test=X[test_index]
     y_train=y[train_index]
     y_test=y[test_index]
     print(X_train,y_train,X_test,y_test)

     
