import copy
import math
import collections
from river import base
import numpy as np

from IWE import WE, dict_div, dict_mul, dict_plu

class IWE_M(WE):
    def __init__(self, model: base.Classifier, n_models=10, window=100):
        super().__init__(model=model, n_models=n_models, window=window)
        self.weight = []
        self.weight.append({})
        self.Lambda = []
        self.Lambda.append({})
        self.buff_x = []
        self.buff_y = []


    def learn_one(self, x, y):
        for i, model in enumerate(self):
            if self.models[i].predict_one(x) == y:
                k = -1
                c = 1
            else:
                k = 1
                c = 0
            if y in self.weight[i]:
                W = self.window*math.exp(-self.Lambda[i][y])
                if W < 10:
                    W = 10
                self.weight[i][y] = self.weight[i][y]*(W-1)/W + c/W
                self.Lambda[i][y] = self.Lambda[i][y] + k
                if k == 1:
                    if self.models[i].predict_one(x) in self.weight[i]:
                        self.weight[i][self.models[i].predict_one(x)] = self.weight[i][
                                                                            self.models[i].predict_one(x)] * (
                                                                                    W - 1) / W + c / W
                        self.Lambda[i][self.models[i].predict_one(x)] = self.Lambda[i][
                                                                            self.models[i].predict_one(x)] + k
                    else:
                        self.weight[i].update({self.models[i].predict_one(x): 1})
                        self.Lambda[i].update({self.models[i].predict_one(x): 0})

                if self.Lambda[i][y] < 0:
                    self.Lambda[i][y] = 0
            else:
                self.weight[i].update({y: 1})
                self.Lambda[i].update({y: 0})
            model.learn_one(x, y)

        self.count = self.count + 1
        self.buff_x.append(x)
        self.buff_y.append(y)
        if self.count == self.window:
            new_model = copy.deepcopy(self.model)
            for i in range(len(self.buff_x)):
                new_model.learn_one(self.buff_x[i], self.buff_y[i])
            if len(self.models) == self.n_models:
                inlist = []
                for i in range(len(self.weight)):
                    inlist.append(np.array(list(self.weight[i].values())).mean())
                index = inlist.index(min(inlist))
                self.models[index] = new_model
                self.weight[index] = dict()
                self.Lambda[index] = dict()
            else:
                self.models.append(new_model)
                self.weight.append(dict())
                self.Lambda.append(dict())
            self.count = 0
            self.buff_x = []
            self.buff_y = []
        return self

    def predict_proba_one(self, x):
        y_pred = collections.Counter()
        total = {}
        for i in range(len(self.models)):
            total = dict_plu(self.weight[i], total)

        for i, model in enumerate(self):
            pre = model.predict_proba_one(x)
            standweight = dict_div(self.weight[i], total)
            pre = dict_mul(pre, standweight)
            y_pred.update(pre)
        total = sum(y_pred.values())
        if total > 0:
            return {label: proba / total for label, proba in y_pred.items()}
        return y_pred

    def get_weight(self):
        return self.weight