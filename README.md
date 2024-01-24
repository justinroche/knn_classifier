# k-Nearest Neighbors Classifier
**A k-Nearest Neighbors supervised classifier in Python 3.11.7**

*Developed by Justin Roche*

---

## Importing the package
```python
from knn_classifier import KNNClassifier
```

## Using the package

### Formatting the input data
Start by setting up your training and testing data. This classifier must be fitted before making any predictions.

Populate a text file with your data, following this format:
> 1.4, 4.0, 15.0, 1.1, 1.6, 5.0\
> 1.2, 4.0, 22.2, 2.2, 3.0, 6.0

Each record contains a list of its data points, each separated by a comma and a space, with the record's label as the final item in the list. In the above examples, the labels are 5.0 and 6.0, respectively.

1. Split the data and labels from a file (if necessary).
```python
train_data, train_label = KNNClassifier.split_data_label_from_file("train.txt")
test_data, test_label = KNNClassifier.split_data_label_from_file("test.txt")
```
The static method `split_data_label_from_file` returns two numpy arrays. The first contains all columns except the last, and the second contains the last column. This method assumes that the labels of the file are contained in the last column.

2. Initialize a `KNNClassifier` object.
```python
knn_classifier = KNNClassifier()
```