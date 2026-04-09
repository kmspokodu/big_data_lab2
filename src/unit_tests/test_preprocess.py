import configparser
import os
import unittest
import pandas as pd
import sys

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from preprocess import DataMaker

config = configparser.ConfigParser()
config.read("config.ini")


class TestDataMaker(unittest.TestCase):

    def setUp(self) -> None:
        self.data_maker = DataMaker()

    def test_get_data(self):
        """Тест загрузки и разделения данных на X и y"""
        result = self.data_maker.get_data()
        self.assertEqual(result, True)
        self.assertTrue(os.path.isfile(self.data_maker.X_path))
        self.assertTrue(os.path.isfile(self.data_maker.y_path))

    def test_split_data(self):
        """Тест разделения данных на train/test наборы"""
        result = self.data_maker.split_data()
        self.assertEqual(result, True)
        # Проверяем что все файлы созданы
        for path in self.data_maker.train_path + self.data_maker.test_path:
            self.assertTrue(os.path.isfile(path))

    def test_data_shapes(self):
        """Тест корректности размеров данных"""
        self.data_maker.split_data()
        
        X_train = pd.read_csv(self.data_maker.train_path[0], index_col=0)
        y_train = pd.read_csv(self.data_maker.train_path[1], index_col=0)
        X_test = pd.read_csv(self.data_maker.test_path[0], index_col=0)
        y_test = pd.read_csv(self.data_maker.test_path[1], index_col=0)
        
        # Sonar датасет имеет 60 признаков
        self.assertEqual(X_train.shape[1], 60)
        self.assertEqual(X_test.shape[1], 60)
        
        # Проверяем что train/test разделены примерно 80/20
        total = X_train.shape[0] + X_test.shape[0]
        train_ratio = X_train.shape[0] / total
        self.assertGreater(train_ratio, 0.75)
        self.assertLess(train_ratio, 0.85)

    def test_save_splitted_data(self):
        """Тест сохранения данных"""
        X_data = pd.read_csv(self.data_maker.X_path, index_col=0)
        result = self.data_maker.save_splitted_data(X_data, self.data_maker.X_path)
        self.assertEqual(result, True)
        self.assertTrue(os.path.isfile(self.data_maker.X_path))


if __name__ == "__main__":
    unittest.main()
