import os
import torch
import joblib

from sklearn import svm


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
