#
# Use a small collection of CSV data to display data from
# reading, transforming, training, evaluating training, and finally predict.
#
# Ref: https://www.tensorflow.org/get_started/datasets_quickstart
#
import tensorflow as tf
import pandas as pd


STEPS = 50000

FILE = "data.csv"
SEP = "=" * 140

dataframe = pd.read_csv(FILE, dtype={
    'Name': str, 'Width': int, "Height": int})

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

# Try drop function to remove useless columns.
print(SEP)
dataframe = dataframe[pd.notnull(dataframe['Length'])]
print("👉 find 'Length': ")
print(dataframe)
dataframe = dataframe[pd.notnull(dataframe['Dense'])]
print("👉 find 'Dense': ")
print(dataframe)
dataframe = dataframe.drop(["Length", "Dense"], axis=1)
print("👉 Use drop()")
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

# One-hot
print(SEP)
print("👉 One-hot")
train_features, train_labels = dataset.make_one_shot_iterator().get_next()
print((train_features, train_labels))


# Train example and predict
import numpy as np


def _input_data_(dataframe, labels):
    dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    sample_count = dataframe.shape[0]
    dataset = dataset.shuffle(sample_count + 1).repeat().batch(2)
    return dataset.make_one_shot_iterator().get_next()


def train_and_predict():
    feature_cols = [
        tf.feature_column.numeric_column("Width"),
        tf.feature_column.numeric_column("Height")
    ]
    model = tf.estimator.LinearClassifier(
        feature_cols, n_classes=3, label_vocabulary=["a", "b", "c"])
    model.train(steps=2000, input_fn=lambda: _input_data_(dataframe, labels))
    print(SEP)
    print("👉 evaluate")
    evaluate = model.evaluate(
        steps=STEPS, input_fn=lambda: _input_data_(dataframe, labels))
    print(evaluate)
    print(SEP)
    print("👉 predict")
    test_width_cols = np.array(
        [10, 4, 6, 2, 11], dtype=np.int32)
    test_height_cols = np.array(
        [23, 56, 66, 50, 25], dtype=np.int32)
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"Width": test_width_cols, "Height": test_height_cols},
        shuffle=False)
    predict_res = list(
        model.predict(input_fn=predict_input_fn))
    for res in predict_res:
        print(
            "🙏  Probability：{:<5.2f} -> {}".format(max(res["probabilities"]), res["classes"][0]))


train_and_predict()

