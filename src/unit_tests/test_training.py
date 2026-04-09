import configparser
import os
import unittest
import pandas as pd
import sys

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from train import MultiModel

config = configparser.ConfigParser()
config.read("config.ini")


class TestMultiModel(unittest.TestCase):

    def setUp(self) -> None:
        self.multi_model = MultiModel()

    def test_log_reg(self):
        """Тест обучения логистической регрессии"""
        result = self.multi_model.log_reg()
        self.assertEqual(result, True)
        # Проверяем, что модель сохранена
        self.assertTrue(os.path.isfile(self.multi_model.log_reg_path))

    def test_model_accuracy(self):
        """Тест что модель имеет приемлемую точность"""
        self.multi_model.log_reg()
        from sklearn.linear_model import LogisticRegression
        import pickle
        
        # Загружаем обученную модель
        with open(self.multi_model.log_reg_path, 'rb') as f:
            classifier = pickle.load(f)
        
        # Проверяем точность на тестовом наборе
        score = classifier.score(self.multi_model.X_test, self.multi_model.y_test)
        self.assertGreater(score, 0.7)  # Минимум 70% accuracy

    def test_model_predictions(self):
        """Тест что модель может делать предсказания"""
        self.multi_model.log_reg()
        from sklearn.linear_model import LogisticRegression
        import pickle
        
        with open(self.multi_model.log_reg_path, 'rb') as f:
            classifier = pickle.load(f)
        
        # Делаем предсказание на одном образце
        sample = self.multi_model.X_test.iloc[:1]
        prediction = classifier.predict(sample)
        self.assertEqual(len(prediction), 1)
        self.assertIn(prediction[0], ['M', 'R'])


if __name__ == "__main__":
    unittest.main()
