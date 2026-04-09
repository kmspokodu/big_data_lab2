import configparser
import os
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import sys
import traceback

from logger import Logger

SHOW_LOG = True


class MultiModel():

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")
        
        # Читаем препроцессированные данные
        self.X_train = pd.read_csv(
            self.config["SPLIT_DATA"]["X_train"], index_col=0)
        self.y_train = pd.read_csv(
            self.config["SPLIT_DATA"]["y_train"], index_col=0)
        self.X_test = pd.read_csv(
            self.config["SPLIT_DATA"]["X_test"], index_col=0)
        self.y_test = pd.read_csv(
            self.config["SPLIT_DATA"]["y_test"], index_col=0)
        
        # Преобразуем в правильный формат
        self.y_train = self.y_train.values.ravel()
        self.y_test = self.y_test.values.ravel()
        
        self.project_path = os.path.join(os.getcwd(), "experiments")
        self.log_reg_path = os.path.join(self.project_path, "log_reg.sav")
        self.rand_forest_path = os.path.join(
            self.project_path, "rand_forest.sav")
        self.knn_path = os.path.join(self.project_path, "knn.sav")
        self.svm_path = os.path.join(self.project_path, "svm.sav")
        self.gnb_path = os.path.join(self.project_path, "gnb.sav")
        self.d_tree_path = os.path.join(self.project_path, "d_tree.sav")
        self.log.info("MultiModel is ready")

    def log_reg(self, predict=False) -> bool:
        classifier = LogisticRegression(max_iter=1000, random_state=42)
        try:
            classifier.fit(self.X_train, self.y_train)
        except Exception:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        if predict:
            y_pred = classifier.predict(self.X_test)
            acc = accuracy_score(self.y_test, y_pred)
            self.log.info(f"LOG_REG Accuracy: {acc:.4f}")
            print(classification_report(self.y_test, y_pred))
        params = {'path': self.log_reg_path}
        return self.save_model(classifier, self.log_reg_path, "LOG_REG", params)

    def save_model(self, classifier, path: str, name: str, params: dict) -> bool:
        self.config[name] = params
        os.remove('config.ini')
        with open('config.ini', 'w') as configfile:
            self.config.write(configfile)
        with open(path, 'wb') as f:
            pickle.dump(classifier, f)

        self.log.info(f'{path} is saved')
        return os.path.isfile(path)


if __name__ == "__main__":
    multi_model = MultiModel()
    multi_model.log_reg(predict=True)
