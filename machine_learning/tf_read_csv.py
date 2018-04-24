#
# Use a small collection of CSV data to display data from
# reading, transforming, training, evaluating training, and finally predict.
#
# Ref: https://www.tensorflow.org/get_started/datasets_quickstart
#
import tensorflow as tf
import pandas as pd

FILE = "data.csv"
SEP = "=" * 100

dataframe = pd.read_csv(FILE, dtype={
    'Name': str, 'Width': float, "Height": float})

# Get CSV content
print(SEP)
print(dataframe)


# Get labels
print(SEP)
print("👉 Pop labels")
labels = dataframe.pop("Name")
print(labels)
print(type(labels))

print(SEP)
print("👉 After popping labels")
print(dataframe)


# Convert columns to map.
print(SEP)
print("👉 dict_dataframe")
dict_dataframe = dict(dataframe)
print(dict_dataframe)

print("👉 dict_dataframe[Width]")
print(dict_dataframe["Width"])

print("👉 dict_dataframe[Height]")
print(dict_dataframe["Height"])

# !!!! TensorSliceDataset
print(SEP)
print("👉 TensorSliceDataset")
dataset = tf.data.Dataset.from_tensor_slices((dict_dataframe, labels))
print(dataset)


# shuffle
print(SEP)
sample_count = dataframe.shape[0]
print("👉 Shuffle TensorSliceDataset: data-count: {}".format(sample_count))
dataset = dataset.shuffle(sample_count + 1).repeat().batch(2)
print(dataset)

# One-shot
print(SEP)
print("👉 One-shot")
train_features, train_labels = dataset.make_one_shot_iterator().get_next()
print((train_features, train_labels))


# Train example and predict
import numpy as np


def _input_data_(dataframe, labels):
    dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    sample_count = dataframe.shape[0]
    dataset = dataset.shuffle(sample_count + 1).repeat().batch(2)
    return dataset.make_one_shot_iterator().get_next()


feature_cols = [
    tf.feature_column.numeric_column("Width"),
    tf.feature_column.numeric_column("Height")
]

model = tf.estimator.LinearClassifier(
    feature_cols, n_classes=3, label_vocabulary=["a", "b", "c"])
model.train(steps=50, input_fn=lambda: _input_data_(dataframe, labels))

print(SEP)
print("👉 evaluate")
evaluate = model.evaluate(
    steps=50, input_fn=lambda: _input_data_(dataframe, labels))
print(evaluate)


print(SEP)
print("👉 predict")

test_width_cols = np.array(
    [10, 6, 2, 11], dtype=np.float32)
test_height_cols = np.array(
    [23, 66, 50, 25], dtype=np.float32)
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"Width": test_width_cols, "Height": test_height_cols},
    num_epochs=1,
    shuffle=False)

predict_res = list(
    model.predict(input_fn=predict_input_fn))
for res in predict_res:
    print(res["classes"][0])
