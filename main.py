from knn_classifier import KNNClassifier


def main():

    # Create a new KNNClassifier instance.
    classifier = KNNClassifier()

    # Split the train and test files into data and label arrays.
    # Specify the delimiter.
    train_data, train_label = KNNClassifier.split_data_label_from_file("data/new_train_1.txt", ",")
    test_data, test_label = KNNClassifier.split_data_label_from_file("data/new_test_1.txt", ",")

    # Fit the classifier to the train data and labels.
    classifier.fit(train_data, train_label)

    # Use the fitted classifier to predict the labels of the test data.
    # Specify the k value used in the prediction algorithm.
    test_label_predictions = classifier.predict(test_data, 2)

    # Check the accuracy of the predicted labels.
    accuracy: float = KNNClassifier.check_accuracy(test_label_predictions, test_label)
    print(accuracy)


if __name__ == "__main__":
    main()
