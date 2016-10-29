#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from copy import deepcopy
from sklearn.tree import DecisionTreeRegressor
from sklearn.base import ClassifierMixin, BaseEstimator


class BinaryBoostingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators, lr=0.1, max_depth=3):
        self.base_regressor = DecisionTreeRegressor(criterion='friedman_mse',
                                                    splitter='best',
                                                    max_depth=max_depth)
        self.lr = lr
        self.n_estimators = n_estimators
        self.feature_importances_ = None
        self.estimators_ = []
        self.coefficients = []

    def loss_grad(self, original_y, pred_y):
        # Вычислите градиент на кажом объекте
        ### YOUR CODE ###
        grad = None
        grad = self.lr * (pred_y - original_y)

        return grad

    def fit(self, X, original_y):
        # Храните базовые алгоритмы тут
        self.estimators_ = []
        self.coefficients = []

        for i in range(self.n_estimators):
            grad = self.loss_grad(original_y, self._predict(X))
            # Настройте базовый алгоритм на градиент, это классификация или регрессия?
            ### YOUR CODE ###
            estimator = None
            estimator = deepcopy(self.base_regressor)
            estimator.fit(X, -grad)

            ### END OF YOUR CODE
            self.estimators_.append(estimator)
            self.coefficients.append(1)

            sc = np.dot(original_y, self._predict(X))
            md = np.dot(self._predict(X), self._predict(X))
            self.coefficients[-1] = sc / md

        grad = self.loss_grad(original_y, self._predict(X))
        self.out_ = self._outliers(grad)
        self.feature_importances_ = self._calc_feature_imps()

        return self

    def _predict(self, X):
        # Получите ответ композиции до применения решающего правила
        ### YOUR CODE ###
        y_pred = None

        if len(self.estimators_) == 0:
            return np.zeros((X.shape[0]))
        
        h = [est.predict(X) for est in self.estimators_]
        y_pred = np.zeros((X.shape[0]))
        
        for i in np.arange(len(h)):
            y_pred += self.coefficients[i] * h[i]

        return y_pred

    def predict(self, X):
        # Примените к self._predict решающее правило
        ### YOUR CODE ###
        y_pred = None

        y_pred = np.sign(self._predict(X))

        return y_pred

    def _outliers(self, grad):
        # Топ-10 объектов с большим отступом
        ### YOUR CODE ###
        _outliers = None

        args = np.argsort(grad)
        _outliers = np.hstack((args[:10], args[-10:]))

        return _outliers

    def _calc_feature_imps(self):
        # Посчитайте self.feature_importances_ с помощью аналогичных полей у базовых алгоритмов
        f_imps = None
        ### YOUR CODE ###
        f_imps = np.zeros_like(self.estimators_[0].feature_importances_)
        
        for i in np.arange(len(self.estimators_)):
            f_imps += self.estimators_[i].feature_importances_*self.coefficients[i]


        return f_imps/len(self.estimators_)
