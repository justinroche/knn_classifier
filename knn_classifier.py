import numpy as np
from statistics import mode


class KNNClassifier:

    def __init__(self):
        self.fit_data = None
        self.fit_label = None

    def fit(self, training_data: np.ndarray, training_label: np.ndarray):
        self.fit_data: np.ndarray = training_data
        self.fit_label: np.ndarray = training_label

    def predict(self, test_data, k: int):

        if self.fit_data is None or self.fit_label is None:
            print("Classifier is untrained; cannot predict test data.")
            return None

        predictions: np.ndarray = np.zeros(len(test_data), dtype=int)

        for i, datum in enumerate(test_data):
            distances: np.ndarray = np.linalg.norm(datum - self.fit_data, axis=1)
            sorted_indices: np.ndarray = np.argsort(distances)
            votes: list = [self.fit_label[i] for i in sorted_indices[:k]]
            predictions[i] = mode(votes)

        return predictions

    @staticmethod
    def check_accuracy(predicted_labels: np.ndarray, actual_labels: np.ndarray) -> float:
        if len(predicted_labels) != len(actual_labels):
            print("Unequal arrays; cannot check accuracy.")
            return -1

        count: int = np.sum(predicted_labels == actual_labels)

        for i, (predicted, actual) in enumerate(zip(predicted_labels, actual_labels)):
            print(f"Index: {i}, Prediction: {predicted}, Actual: {actual}")

        print(f"Correctly predicted {count} out of {len(predicted_labels)} labels.")

        return count / len(predicted_labels)

    @staticmethod
    def split_data_label_from_file(file_path: str, delimiter: str = ", "):

        try:
            with open(file_path, "r") as fileio:
                lines: list[str] = fileio.readlines()

                # Data array contains all but last column
                data = np.array(
                    [list(map(float, line.split(delimiter)[:-1])) for line in lines]
                )

                # Label array contains last column
                label = np.array(
                    [int(line.split(",")[-1]) for line in lines]
                )

                return data, label

        except Exception as e:
            print(type(e).__name__, e)
            return None
