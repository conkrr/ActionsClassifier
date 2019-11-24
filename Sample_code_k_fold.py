import pandas as pd
from plotly.offline import iplot, plot
import plotly.graph_objs as go
import plotly.figure_factory as ff
from scipy import stats
import numpy as np
import math
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from itertools import product

df = pd.read_table("data_3_6.csv", sep=",", names=["x1", "x2", "y"])  # <-- FIX THIS to get the correct file and labels


X = df[["x1", "x2"]]
y = df["y"]

scores_gnb = []
scores_lda = []
scores_qda = []
gnb = GaussianNB()
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()
cv = KFold(n_splits=5, random_state=42, shuffle=True)
for train_index, test_index in cv.split(X):
    # print("Train Index: ", train_index, "\n")
    # print("Test Index: ", test_index)

    X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[test_index], y.iloc[train_index], y.iloc[test_index]

    # gnb
    gnb.fit(X_train, y_train)
    scores_gnb.append(gnb.score(X_test, y_test))

    # lda
    lda.fit(X_train, y_train)
    scores_lda.append(lda.score(X_test, y_test))

    #qda
    qda.fit(X_train, y_train)
    scores_qda.append(qda.score(X_test, y_test))


def print_matrix(m):
    print('\n'.join([''.join(['{:4}'.format(item) for item in row])
                     for row in m]))


print()
print("Score of GNB is: " + str(np.mean(scores_gnb)) + " with a std of: " + str(np.std(scores_gnb)))
print("Class Priors: ", gnb.class_prior_)
print("GNB confusion matrix is: ")
print_matrix(confusion_matrix(y, gnb.predict(X)))
print()

print("Score of LDA is: " + str(np.mean(scores_lda)) + " with a std of: " + str(np.std(scores_lda)))
print("LDA confusion matrix is: ")
print_matrix(confusion_matrix(y, lda.predict(X)))
print()

print("Score of QDA is: " + str(np.mean(scores_qda)) + " with a std of: " + str(np.std(scores_qda)))
print("QDA confusion matrix is: ")
print_matrix(confusion_matrix(y, qda.predict(X)))
print()

