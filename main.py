from knn_classifier import KNNClassifier


def main():

    classifier = KNNClassifier()

    train_data, train_label = KNNClassifier.split_data_label_from_file("wine_train.txt")
    test_data, test_label = KNNClassifier.split_data_label_from_file("wine_test.txt")

    classifier.fit(train_data, train_label)

    test_label_guesses = classifier.predict(test_data, 200)
    accuracy: float = KNNClassifier.check_accuracy(test_label_guesses, test_label)
    print(accuracy)


if __name__ == "__main__":
    main()
