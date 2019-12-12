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

# load data

df = pd.read_table("ActionsData/trainData.csv", sep=",", header=0)
df.drop(columns=df.columns[0], axis=1, inplace=True)

test_data = pd.read_table("ActionsData/testData.csv", sep=",", header=0)
test_data.drop(columns=test_data.columns[0], axis=1, inplace=True)

y = pd.read_table("ActionsData/trainLabel.csv", sep=",", header=0)
y.drop(columns=y.columns[0], axis=1, inplace=True)
y = y["class_id"]

df["y"] = y  # integrate y into dataframe

df = df.sample(frac=1).reset_index(drop=True)  # shuffle the data

# parameters

max_iter = 1000
num_of_hidden_l = 4
hl_size = 100
split = 0.7
n_samples = len(df)
split_index = int(n_samples * split)

# df = (df-df.min())/(df.max()-df.min())  # normalize

# train and test sets

training_set = df.iloc[:split_index, :]
train_X = df.iloc[:split_index, :-1]
# train_X = np.array(train_X, np.float32).reshape([-1, 1])
train_Y = df.iloc[:split_index, -1]
# train_Y = np.array(train_Y, np.int).reshape([-1, 1])

testing_set = df.iloc[split_index:, :]
test_X = df.iloc[split_index:, :-1]
# test_X = np.array(test_X, np.float32).reshape([-1, 1])
test_Y = df.iloc[split_index:, -1]
# test_Y = np.array(test_Y, np.int).reshape([-1, 1])


# --- Dataset Info ---
print("{} total samples".format(len(df)))
print("{} unique features per sample".format(len(df.columns) - 1))
print("{} unique classes".format(len(y.unique())))
print("Samples per class:\n", y.value_counts().sort_index())

layers = []
for i in range(num_of_hidden_l):
    layers.append(hl_size)

nn = MLPClassifier(hidden_layer_sizes=layers, activation="logistic", verbose=True, max_iter=max_iter)
nn.fit(train_X, train_Y)
print("Testing accuracy of nn: ", nn.score(test_X, test_Y))

# accuracy per class
for a_class in y.unique():
    data_temp = testing_set.loc[testing_set["y"] == a_class]
    accuracy = nn.score(data_temp.iloc[:, :-1], data_temp.iloc[:, -1])
    print("Accuracy for class {} is: {}".format(a_class, accuracy))
