import argparse
import configparser
from datetime import datetime
import os
import json
import pandas as pd
import pickle
from sklearn.metrics import classification_report
import shutil
import sys
import time
import traceback
import yaml

from logger import Logger

SHOW_LOG = True


class Predictor():

    def __init__(self) -> None:
        logger = Logger(SHOW_LOG)
        self.config = configparser.ConfigParser()
        self.log = logger.get_logger(__name__)
        self.config.read("config.ini")
        self.parser = argparse.ArgumentParser(description="Predictor")
        self.parser.add_argument("-m",
                                 "--model",
                                 type=str,
                                 help="Select model",
                                 required=True,
                                 default="LOG_REG",
                                 const="LOG_REG",
                                 nargs="?",
                                 choices=["LOG_REG"])
        self.parser.add_argument("-t",
                                 "--tests",
                                 type=str,
                                 help="Select tests",
                                 required=True,
                                 default="smoke",
                                 const="smoke",
                                 nargs="?",
                                 choices=["smoke", "func"])
        self.X_train = pd.read_csv(
            self.config["SPLIT_DATA"]["X_train"], index_col=0)
        self.y_train = pd.read_csv(
            self.config["SPLIT_DATA"]["y_train"], index_col=0)
        self.X_test = pd.read_csv(
            self.config["SPLIT_DATA"]["X_test"], index_col=0)
        self.y_test = pd.read_csv(
            self.config["SPLIT_DATA"]["y_test"], index_col=0)
        
        
        self.y_train = self.y_train.values.ravel()
        self.y_test = self.y_test.values.ravel()
        
        self.log.info("Predictor is ready")

    def predict(self) -> bool:
        args = self.parser.parse_args()
        try:
            classifier = pickle.load(
                open(self.config[args.model]["path"], "rb"))
        except FileNotFoundError:
            self.log.error(traceback.format_exc())
            sys.exit(1)
        if args.tests == "smoke":
            try:
                score = classifier.score(self.X_test, self.y_test)
                y_pred = classifier.predict(self.X_test)
                self.log.info(f'{args.model} Accuracy: {score:.4f}')
                print(f'{args.model} has {score:.4f} score')
                print(classification_report(self.y_test, y_pred))
            except Exception:
                self.log.error(traceback.format_exc())
                sys.exit(1)
            self.log.info(
                f'{self.config[args.model]["path"]} passed smoke tests')
        elif args.tests == "func":
            tests_path = os.path.join(os.getcwd(), "tests")
            exp_path = os.path.join(os.getcwd(), "experiments")
            for test in os.listdir(tests_path):
                with open(os.path.join(tests_path, test)) as f:
                    try:
                        data = json.load(f)
                        X = pd.json_normalize(data, record_path=['X'])
                        y = pd.json_normalize(data, record_path=['y']).values.ravel()
                        score = classifier.score(X, y)
                        print(f'{args.model} has {score:.4f} score')
                    except Exception:
                        self.log.error(traceback.format_exc())
                        sys.exit(1)
                    self.log.info(
                        f'{self.config[args.model]["path"]} passed func test {f.name}')
                    exp_data = {
                        "model": args.model,
                        "model params": dict(self.config.items(args.model)),
                        "tests": args.tests,
                        "score": str(score),
                        "X_test path": self.config["SPLIT_DATA"]["X_test"],
                        "y_test path": self.config["SPLIT_DATA"]["y_test"],
                    }
                    date_time = datetime.fromtimestamp(time.time())
                    str_date_time = date_time.strftime("%Y_%m_%d_%H_%M_%S")
                    exp_dir = os.path.join(exp_path, f'exp_{test[:6]}_{str_date_time}')
                    os.mkdir(exp_dir)
                    with open(os.path.join(exp_dir,"exp_config.yaml"), 'w') as exp_f:
                        yaml.safe_dump(exp_data, exp_f, sort_keys=False)
                    shutil.copy(os.path.join(os.getcwd(), "logfile.log"), os.path.join(exp_dir,"exp_logfile.log"))
                    shutil.copy(self.config[args.model]["path"], os.path.join(exp_dir,f'exp_{args.model}.sav'))
        return True


if __name__ == "__main__":
    predictor = Predictor()
    predictor.predict()
