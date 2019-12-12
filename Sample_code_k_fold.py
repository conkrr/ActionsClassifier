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
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

# parameters

hl_size = 100

df = pd.read_table("ActionsData/trainData.csv", sep=",", header=0)
df.drop(columns=df.columns[0], axis=1, inplace=True)

test_data = pd.read_table("ActionsData/testData.csv", sep=",", header=0)
test_data.drop(columns=test_data.columns[0], axis=1, inplace=True)


X = df
y = pd.read_table("ActionsData/trainLabel.csv", sep=",", header=0)
y.drop(columns=y.columns[0], axis=1, inplace=True)
y = y["class_id"]

# --- Dataset Info ---
print("{} total samples".format(len(X)))
print("{} unique features per sample".format(len(X.columns)))
print("{} unique classes".format(len(y.unique())))
print("Samples per class:\n", y.value_counts().sort_index())


# --- PCA ----

pca = PCA(n_components=10)
pca.fit(X)


# --- K-fold Naive Bayes, LDA, QDA ---

scores_gnb = []
scores_lda = []
scores_qda = []
scores_qda_red = []
scores_nn = []
gnb = GaussianNB()
lda = LinearDiscriminantAnalysis()
qda = QuadraticDiscriminantAnalysis()
qda_reduced = QuadraticDiscriminantAnalysis()
nn = MLPClassifier(hidden_layer_sizes=(hl_size, hl_size, hl_size), activation="logistic", verbose=False)
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

    #qda reduced
    qda_reduced.fit(pca.transform(X_train), y_train)
    scores_qda_red.append(qda_reduced.score(pca.transform(X_test), y_test))

    #nn
    nn.fit(X_train, y_train)
    scores_nn.append(nn.score(X_test, y_test))


def print_matrix(m):
    print('\n'.join([''.join(['{:4}'.format(item) for item in row])
                     for row in m]))


print()
print("Score of GNB is: " + str(np.mean(scores_gnb)) + " with a std of: " + str(np.std(scores_gnb)))
print("Scores: ", scores_gnb)
print("Class Priors: ", gnb.class_prior_)
print("GNB confusion matrix is: ")
print_matrix(confusion_matrix(y, gnb.predict(X)))
print()

print("Score of LDA is: " + str(np.mean(scores_lda)) + " with a std of: " + str(np.std(scores_lda)))
print("Scores: ", scores_lda)
print("LDA confusion matrix is: ")
print_matrix(confusion_matrix(y, lda.predict(X)))
print()

print("Score of QDA is: " + str(np.mean(scores_qda)) + " with a std of: " + str(np.std(scores_qda)))
print("Scores: ", scores_qda)
print("QDA confusion matrix is: ")
print_matrix(confusion_matrix(y, qda.predict(X)))
print()

print("Score of QDA REDUCED is: " + str(np.mean(scores_qda_red)) + " with a std of: " + str(np.std(scores_qda_red)))
print("Scores: ", scores_qda_red)
print("QDA REDUCED confusion matrix is: ")
print_matrix(confusion_matrix(y, qda_reduced.predict(pca.transform(X))))
print()

print("Score of neural net is: " + str(np.mean(scores_nn)) + " with a std of: " + str(np.std(scores_nn)))
print("Scores: ", scores_nn)
print("NN confusion matrix is: ")
print_matrix(confusion_matrix(y, nn.predict(X)))
print()

# print("Test Set Predictions:")
# print(qda.predict(test_data))
