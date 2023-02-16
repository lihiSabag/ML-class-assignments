import numpy as np
from sklearn import svm  # To fit the svm classifier
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
data = pd.read_csv('spambase.data')

X = data.iloc[:, :-1]
Y = data.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3)

list_C = np.arange(100, 1000, 100)
score_train = np.zeros(len(list_C))
score_test = np.zeros(len(list_C))

# SVC with rbf kernel
svc = svm.SVC(C=1.0).fit(x_train, y_train)
score_train[0] = svc.score(x_train, y_train)
score_test[0]= svc.score(x_test, y_test)
count=0
for C in list_C:
    svc = svm.SVC(C=C)
    svc.fit(x_train, y_train)
    score_train[count] = svc.score(x_train, y_train)
    score_test[count]= svc.score(x_test, y_test)
    count = count + 1

matrix = np.matrix(np.c_[list_C, score_train, score_test])
models = pd.DataFrame(data = matrix, columns =
             ['C', 'Train Accuracy', 'Test Accuracy'])
print('The Accuracy score of SVM with RBF kernel :')
svc = svm.SVC(C=1.0).fit(x_train, y_train)
print(f'The Accuracy score with C=1.0 is {svc.score(x_test, y_test)}')
print(models.head(n=10))


# SVC with linear kernel
svc = svm.SVC(kernel='linear', C=1.0).fit(x_train, y_train)
y_pred = svc.predict(x_test)
print(f'The Accuracy score of linear SVM with C=1.0 is {accuracy_score(y_test, y_pred)}')

# SVC with polynomial (degree 3) kernel
poly_svc = svm.SVC(kernel='poly', degree=3, C=1.0).fit(x_train, y_train)
y_pred = poly_svc.predict(x_test)
print(f'The Accuracy score of Quadratic SVM with C=1.0 is {accuracy_score(y_test, y_pred)}')



