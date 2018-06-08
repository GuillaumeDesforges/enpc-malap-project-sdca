import unittest
import numpy as np

from engine.estimators.logistic_regression import LogisticRegression
from engine.optimizers.sgd_logistic import LogisticSGD
from engine.optimizers.sdca_logistic import LogisticSDCA


class TestLogisticSGD(unittest.TestCase):
    def setUp(self):
        self.sgd = LogisticSGD(1, 1e-3)
        self.estimator = LogisticRegression(optimizer=self.sgd)
        np.random.seed(0)
        x1 = np.random.normal(loc=(-1, -1), scale=(1, 1), size=(10, 2))
        x2 = np.random.normal(loc=(1, 1), scale=(1, 1), size=(10, 2))
        self.x = np.concatenate([x1, x2])
        y1 = -np.ones(shape=10)
        y2 = np.ones(shape=10)
        self.y = np.concatenate([y1, y2])

    def testFit(self):
        np.random.seed(10)
        self.estimator.fit(self.x, self.y)
        # regression test
        self.assertListEqual(self.estimator.w.tolist(), [0.030821640301161652, 0.03813816690922319])

    def testPredict(self):
        w = [0.20140517, 0.26121764]
        self.estimator.w = w
        y_pred = self.estimator.predict(self.x)
        errors = y_pred != self.y
        error_rate = sum(errors)/len(errors)
        # regression test
        self.assertEqual(error_rate, 0.1)


class TestLogisticSDCA(unittest.TestCase):
    def setUp(self):
        self.sgd = LogisticSDCA(1)
        self.estimator = LogisticRegression(optimizer=self.sgd)
        np.random.seed(0)
        x1 = np.random.normal(loc=(-1, -1), scale=(1, 1), size=(10, 2))
        x2 = np.random.normal(loc=(1, 1), scale=(1, 1), size=(10, 2))
        self.x = np.concatenate([x1, x2])
        y1 = -np.ones(shape=10)
        y2 = np.ones(shape=10)
        self.y = np.concatenate([y1, y2])

    def testFit(self):
        np.random.seed(10)
        self.estimator.fit(self.x, self.y)
        # regression test
        self.assertListEqual(self.estimator.w.tolist(), [0.21104995073661634, 0.27655396212509703])

    def testPredict(self):
        w = [0.20140517, 0.26121764]
        self.estimator.w = w
        y_pred = self.estimator.predict(self.x)
        errors = y_pred != self.y
        error_rate = sum(errors)/len(errors)
        # regression test
        self.assertEqual(error_rate, 0.1)
