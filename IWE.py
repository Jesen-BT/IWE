import copy
import math
import collections
from river import base, linear_model
import numpy as np



def dict_mul(d1, d2):
    d3 = dict()
    for k in d1:
        if k in d2:
            d3[k] = d1[k] * d2[k]
    return d3

def dict_plu(d1, d2):
    d3 = dict()
    for k in d1:
        if k in d2:
            d3[k] = d1[k] + d2[k]
        else:
            d3[k] = d1[k]
    return d3

def dict_div(d1, d2):
    d3 = dict()
    for k in d2:
        if k in d1:
            d3[k] = d1[k] / d2[k]
    return d3


class WE(base.WrapperMixin, base.EnsembleMixin, base.Classifier):
    def __init__(self, model, n_models=10, window=100):
        super().__init__(copy.deepcopy(model) for _ in range(1))
        self.n_models = n_models
        self.model = model
        self.count = 0
        self.window = window



    @property
    def _wrapped_model(self):
        return self.model

    @classmethod
    def _unit_test_params(cls):
        return {"model": linear_model.LogisticRegression()}

    def learn_one(self, x, y):
        self.count = self.count + 1
        self.models[-1].learn_one(x, y)
        if self.count == self.window:
            new_model = copy.deepcopy(self.model)
            if len(self.models) == self.n_models:
                self.models.pop(0)
            self.models.append(new_model)
            self.count = 0
        return self

    def predict_proba_one(self, x):
        y_pred = collections.Counter()
        for classifier in self:
            y_pred.update(classifier.predict_proba_one(x))

        total = sum(y_pred.values())
        if total > 0:
            return {label: proba / total for label, proba in y_pred.items()}
        return y_pred


class IWE(WE):
    def __init__(self, model: base.Classifier, n_models=10, window=100):
        super().__init__(model=model, n_models=n_models, window=window)
        self.weight = []
        self.Lambda = []
        self.weight.append(1)
        self.Lambda.append(0)
        self.buff_x = []
        self.buff_y = []

    def learn_one(self, x, y):
        for i, model in enumerate(self):
            if self.models[i].predict_one(x) == y:
                W = self.window * math.exp(-self.Lambda[i])
                if W < 10:
                    W = 10
                self.weight[i] = self.weight[i] * (W - 1) / W + 1 / W
                self.Lambda[i] = self.Lambda[i] - 1
                if self.Lambda[i] < 0:
                    self.Lambda[i] = 0
            else:
                W = self.window * math.exp(-self.Lambda[i])
                if W < 10:
                    W = 10
                self.weight[i] = self.weight[i] * (W - 1) / W
                self.Lambda[i] = self.Lambda[i] + 1
            model.learn_one(x, y)

        self.count = self.count + 1
        self.buff_x.append(x)
        self.buff_y.append(y)
        if self.count == self.window:
            new_model = copy.deepcopy(self.model)
            for i in range(len(self.buff_x)):
                new_model.learn_one(self.buff_x[i], self.buff_y[i])
            if len(self.models) == self.n_models:
                index = self.weight.index(min(self.weight))
                self.models[index] = new_model
                self.weight[index] = 1
                self.Lambda[index] = 0
            else:
                self.models.append(new_model)
                self.weight.append(1)
                self.Lambda.append(0)
            self.count = 0
            self.buff_x = []
            self.buff_y = []
        return self

    def predict_proba_one(self, x):
        y_pred = collections.Counter()
        for classifier in self:
            pre = classifier.predict_proba_one(x)
            pre.update((key, value*self.weight[self.models.index(classifier)]) for key, value in pre.items())
            y_pred.update(pre)

        total = sum(y_pred.values())
        if total > 0:
            return {label: proba / total for label, proba in y_pred.items()}
        return y_pred

    def get_weight(self):
        return self.weight
