from knn_classifier import KNNClassifier


def main():
    classifier = KNNClassifier()

    data, label = KNNClassifier.split_train_file("wine_train.txt")
    classifier.fit(data, label)


if __name__ == "__main__":
    main()
