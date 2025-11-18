import unittest
import pandas as pd
import numpy as np

from src.registry.model_registry import ModelRegistry
from src.processing.datapreprocessor import DataPreprocessor
from src.manager.manager import Manager
from src.models.logistic_model import LogisticRegressionModel
from src.models.kmeansmodel import KMeansModel


class DummyInterface:
    """Minimal interface stub satisfying Manager expectations."""

    def setup(self):
        return None

    def run(self, *_args, **_kwargs):
        return None


class TestRegistry(unittest.TestCase):
    def test_register_and_get_model(self):
        registry = ModelRegistry()
        registry.register_model("kmeans", KMeansModel)
        self.assertTrue(registry.model_exists("kmeans"))

        model_instance = registry.get_model("kmeans")
        self.assertIsInstance(model_instance, KMeansModel)


class TestPreprocessor(unittest.TestCase):
    def test_split_with_dataframe(self):
        df = pd.DataFrame(
            {
                "feature_1": [0.0, 1.0, 2.0, 3.0],
                "feature_2": [1.0, 2.0, 3.0, 4.0],
                "target": [0, 1, 0, 1],
            }
        )
        pre = DataPreprocessor({"df": df, "target_column": "target"})
        X_train, y_train, X_val, y_val, X_test, y_test = pre.split()

        self.assertEqual(X_train.shape[1], 2)
        # ensure splits are non-empty and y is aligned
        self.assertTrue(len(X_train) > 0 and len(y_train) > 0)
        self.assertEqual(X_train.shape[0], y_train.shape[0])
        self.assertEqual(X_val.shape[0], y_val.shape[0])
        self.assertEqual(X_test.shape[0], y_test.shape[0])


class TestManagerPipeline(unittest.TestCase):
    def test_logistic_pipeline(self):
        df = pd.DataFrame(
            {
                "feature_1": [0.0, 1.0, 2.0, 3.0],
                "feature_2": [1.0, 2.0, 3.0, 4.0],
                "target": [0, 1, 0, 1],
            }
        )
        pre = DataPreprocessor({"df": df, "target_column": "target"})
        X_train, y_train, X_val, y_val, X_test, y_test = pre.split()
        splits = {
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
        }

        registry = ModelRegistry()
        registry.register_model("logistic", LogisticRegressionModel)

        manager = Manager(interface=DummyInterface(), model_registry=registry, preprocessor=pre)
        cfg = {
            "name": "LogisticRegression",
            "type": "logistic",
            "C": 1.0,
            "solver": "lbfgs",
            "max_iter": 200,
        }
        runtime_cfg = {**cfg, **splits}

        train_res = manager.run_pipeline("logistic", mode="train", config=runtime_cfg)
        self.assertIn("train_accuracy", train_res)
        pred_res = manager.run_pipeline("logistic", mode="predict", config=runtime_cfg)
        self.assertEqual(pred_res["n_samples"], runtime_cfg["X_val"].shape[0])


if __name__ == "__main__":
    unittest.main()
