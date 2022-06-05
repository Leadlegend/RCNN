import os
import torch

from sklearn import svm
from sklearn.externals import joblib


class SVMModel:
    def __init__(self):
        self.clf = None

    def train(self, features, labels):
        self.clf = svm.LinearSVC()
        self.clf.fit(features, labels)

    def predict(self, x):
        assert self.clf is not None
        pred = self.clf.predict(x)
        return pred
