#
# In this example, I use the official data from the Iris flower classification:
# Https://en.wikipedia.org/wiki/Iris_(plant)
# This version is tensorflow solution which provides an official source of data that should be integrated with Wikipedia.
# This sample will first list the "features" and the "labels" that will be classified as "classification",
# which is the iris of what species it belongs to.
#
import config
from tensorflow.contrib import learn
from tensorflow.contrib import layers
import numpy as np
from numpy import delete

train_steps = 1500

# Get data for training.
iris_dataset = learn.datasets.load_iris()

# Prepare train-data
train_x = iris_dataset.data
train_y = iris_dataset.target
feature_count = train_x.shape[1]  # Four features.
feature_columns = [layers.real_valued_column("", dimension=feature_count)]

positions_of_collection_to_delete = config.positions_of_collection_to_delete
# Remove rows on iris_dataset.data and copy for training data. 0 -> rows.
train_x = delete(iris_dataset.data, positions_of_collection_to_delete, 0)
# axis = None for remove elements in 1-D array.
train_y = delete(iris_dataset.target, positions_of_collection_to_delete)


# Prepare test-data
test_x = iris_dataset.data[positions_of_collection_to_delete]
test_y = iris_dataset.target[positions_of_collection_to_delete]

# Train model
dnn_classifier = learn.DNNClassifier(feature_columns=feature_columns,
                                     hidden_units=[10, 10, 10],
                                     n_classes=3)
dnn_classifier.fit(train_x, train_y, steps=train_steps)

# Accuracy
evl = dnn_classifier.evaluate(test_x, test_y)
print("\n👍 evaluate:\nloss: {:>5.2f}\naccuracy: {:>5.2f}\nglobal_step: {:>5.2f}\n\n".format(
    evl["loss"], evl["accuracy"], evl["global_step"]))

# Predication
try_data = np.array(
    [[5.1, 2.5, 3., 1.1], [6.1, 3., 4.6, 1.4], [5.6, 3., 4.1, 1.3]])
print("\n👍 predict:\n{}\n\nPredict:\n{}\n\n".format(
    try_data,
    list(dnn_classifier.predict(try_data))
))
