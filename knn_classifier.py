import numpy as np
from typing import TextIO


class KNNClassifier:

    def __init__(self):
        self.data = None
        self.label = None

    def fit(self, training_data, training_label):
        self.data: np.ndarray = training_data
        self.label: np.ndarray = training_label

    def predict(self, test_data):
        pass

    def check_accuracy(self):
        pass

    @staticmethod
    def __split_textio(file: TextIO):
        lines = file.readlines()

        data = np.array(
            [list(map(float, line.split(",")[:-1])) for line in lines]
        )

        label = np.array(
            [int(line.split(",")[-1]) for line in lines]
        )

        return data, label

    @staticmethod
    def split_train_file(train_file):
        try:
            with open(train_file, "r") as file:
                return KNNClassifier.__split_textio(file)

        except Exception as e:
            print(type(e).__name__, e)
            return None
