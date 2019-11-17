import pandas as pd
from plotly.offline import iplot, plot
import plotly.graph_objs as go
import plotly.figure_factory as ff
from scipy import stats
import numpy as np
import math
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

x = pd.read_table("ActionsData/trainData.csv", sep=",", header=0)
x = x.drop(x.columns[0], axis=1)
y = pd.read_table("ActionsData/trainLabel.csv", sep=",", header=0)
y = y.drop(y.columns[0], axis=1)
x["y"] = y["class_id"]

print(x.head(5))
print(y.head(5))


gnb = GaussianNB()
y_pred = gnb.fit(x, y).predict(x)
x["y_pred_gnb"] = y_pred
x["isCorrectGNB"] = x.apply(lambda x: x["y_pred_gnb"] == x["y"], axis=1)
print("The accuracy of naive bayes sklearn is: {}".format(x.groupby("isCorrectGNB").count()[x.columns[0]].get(1) / len(x)))
