#
# In this example, I use the official data from the Iris flower classification:
# Https://en.wikipedia.org/wiki/Iris_(plant)
# The Sklearn provides an official source of data that should be integrated with Wikipedia.
# This sample will first list the "features" and the "labels" that will be classified as "classification",
# which is the iris of what species it belongs to.
#
from sklearn import tree
from sklearn.datasets import load_iris

IND_CNT = 120

iris_dataset = load_iris()

# List features and labels.
print("=" * IND_CNT)
print("Iris dataset including")
print("=" * IND_CNT)
print("Features: {}".format(iris_dataset.feature_names))
print("Labels: {}".format(iris_dataset.target_names))

# List what the iris_dataset is, these're information about iris_dataset.
print("=" * IND_CNT)
print("Iris dataset object properties")
print("=" * IND_CNT)
print("iris_dataset.feature_names: {}, iris_dataset.target_names: {}".format(
    type(iris_dataset.feature_names), type(iris_dataset.target_names)))
print("dataset: {}, dataset.data: {}, dataset.target: {}".format(
    type(iris_dataset), type(iris_dataset.data), type(iris_dataset.target)))
print("dataset.data[x]: {}, dataset.target[x]: {}".format(
    type(iris_dataset.data[0]), type(iris_dataset.target[0])))

# List all iris.
# Lists the target name represented by the label and label for the feature, which is the name of the flower.
print("=" * IND_CNT)
print("Iris list: Iris with features -> label / name of type")
print("=" * IND_CNT)
for i in range(len(iris_dataset.target)):
    print("{} -> {}/{}".format(iris_dataset.data[i],
                               iris_dataset.target[i], iris_dataset.target_names[iris_dataset.target[i]]))


# Generate some training data. Remove some data from original iris_dataset.data might be good for this sample.
import numpy as np
from numpy import delete

# Remove elements by these positions in collection.
positions_of_collection_to_delete = [4, 6, 66]

# Make train-data and labels
# Remove rows on iris_dataset.data and copy for training data. 0 -> rows.
train_data = delete(iris_dataset.data, positions_of_collection_to_delete, 0)
# axis = None for remove elements in 1-D array.
train_labels = delete(iris_dataset.target, positions_of_collection_to_delete)

# Train the model.
decision_tree_classifier = tree.DecisionTreeClassifier()
decision_tree_classifier.fit(train_data, train_labels)

# Make test-data to predict.
test_data = iris_dataset.data[positions_of_collection_to_delete]
test_labels = iris_dataset.target[positions_of_collection_to_delete]
print("test_data:")
print("{}".format(test_data))
print("test_labels:")
print("{}".format(test_labels))

#
# Run predication on the test-data.
#


def predict(predicted):
    predicted_label = decision_tree_classifier.predict([predicted])
    print(
        "Predict: {} -> {}/{}".format(predicted,
                                      predicted_label[0], iris_dataset.target_names[predicted_label[0]]))


for test_object in test_data:
    predict(test_object)

#
# To compute accuracy of predication.
#


from sklearn.metrics import accuracy_score
predication_labels = decision_tree_classifier.predict(test_data)
print("Predict: {}".format(predication_labels))
accuracy = accuracy_score(test_labels, predication_labels)
print("Accuracy of predication: {:.4}%".format(accuracy * 100))
