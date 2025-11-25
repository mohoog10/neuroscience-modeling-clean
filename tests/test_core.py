import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.registry.model_registry import ModelRegistry
from src.processing.datapreprocessor import DataPreprocessor
from src.manager.manager import Manager
from src.models.logistic_model import LogisticRegressionModel
from src.models.kmeansmodel import KMeansModel


class DummyInterface:
    def setup(self):
        return None

    def run(self, *_args, **_kwargs):
        return None


class TestRegistry(unittest.TestCase):
    def test_register_and_get(self):
        registry = ModelRegistry()
        registry.register_model("logistic", LogisticRegressionModel)
        self.assertTrue(registry.model_exists("logistic"))
        inst = registry.get_model("logistic")
        self.assertIsInstance(inst, LogisticRegressionModel)


class TestPreprocessor(unittest.TestCase):
    def test_split_dataframe(self):
        df = pd.DataFrame({
            "f1": [0.0, 1.0, 2.0, 3.0],
            "f2": [1.0, 2.0, 3.0, 4.0],
            "target": [0, 1, 0, 1]
        })
        pre = DataPreprocessor({"df": df, "target_column": "target"})
        X_train, y_train, *_ = pre.split()
        self.assertEqual(X_train.shape[1], 2)
        self.assertEqual(X_train.shape[0], y_train.shape[0])

    def test_load_all_csvs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            df = pd.DataFrame({"a": [0.0, 1.0], "target": [0, 1]})
            df.to_csv(os.path.join(tmpdir, "sample.csv"), index=False)
            pre = DataPreprocessor({"filepath": tmpdir, "target_column": "target"})
            X, y = pre.load_all_csvs()
            self.assertEqual(len(X), 1)
            self.assertEqual(len(y), 1)


class TestManager(unittest.TestCase):
    def test_logistic_pipeline(self):
        # Deterministic arrays to avoid class imbalance in random splits
        X_train = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        y_train = np.array([0, 1, 0, 1])
        X_val = np.array([[1.5, 2.5], [2.5, 3.5]])
        y_val = np.array([0, 1])
        X_test = np.array([[0.5, 1.5], [3.5, 4.5]])
        y_test = np.array([0, 1])

        df = pd.DataFrame({"f1": y_train, "f2": y_train, "target": y_train})
        pre = DataPreprocessor({"df": df, "target_column": "target"})
        registry = ModelRegistry()
        registry.register_model("logistic", LogisticRegressionModel)
        manager = Manager(DummyInterface(), registry, pre)

        cfg = {
            "name": "LogisticRegression",
            "type": "logistic",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 200,
            "fit_predict_on": "val",
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test
        }

        train_res = manager.run_pipeline("logistic", mode="train", config=cfg)
        self.assertIn("train_accuracy", train_res)
        pred_res = manager.run_pipeline("logistic", mode="predict", config=cfg)
        self.assertEqual(pred_res["n_samples"], X_val.shape[0])


class TestKMeansModel(unittest.TestCase):
    def test_train_predict(self):
        X = np.vstack([np.zeros((5, 2)), np.ones((5, 2))])
        cfg = {"n_clusters": 2, "scaler": False, "X_train": X}
        model = KMeansModel()
        model.build(cfg)
        train_res = model.train()
        self.assertIn("n_clusters", train_res)
        model.config["X_val"] = X
        val_res = model.validate()
        self.assertEqual(val_res["n_samples"], X.shape[0])


if __name__ == "__main__":
    unittest.main()
