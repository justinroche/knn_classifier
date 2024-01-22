import numpy as np


class KNNClassifier:

    def __init__(self):
        self.data = None
        self.label = None

    @staticmethod
    def split_train_file(train_file):
        try:
            with open(train_file, "r") as file:
                lines = file.readlines()

                data = np.array(
                    [list(map(float, line.split(",")[:-1])) for line in lines]
                )

                label = np.array(
                    [int(line.split(",")[-1]) for line in lines]
                )

        except Exception as e:
            print(type(e).__name__, e)

        return data, label

    def fit(self, train_data, train_label):
        self.data = train_data
        self.label = train_label

    def predict(self, test_data):
        pass

    def check_accuracy(self):
        pass
