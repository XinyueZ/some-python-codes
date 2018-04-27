#
# Use a small collection of CSV data to display data from
# reading, transforming, training, evaluating training, and finally predict.
#
# Ref: https://www.tensorflow.org/get_started/datasets_quickstart
#
import tensorflow as tf
import pandas as pd
from tensorflow import keras
import numpy as np
from sklearn.preprocessing import LabelEncoder

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

# Make feature list, remove duplicates and keep order.
_labels_ = list()
fn = lambda: [x for x in labels if not (x in _labels_ or _labels_.append(x))]
fn()
print(_labels_)

print(labels)
print(type(labels))

# Use sklearn utility to convert label strings to numbered index.
print(SEP)
print("👉 Use sklearn utility to convert label strings to numbered index")

encoder = LabelEncoder()
label_nums = encoder.fit_transform(labels)
print(label_nums)

# Convert to one-hot standard.
print("👉 Convert labels into one-hot standard")
one_hot_labels = keras.utils.to_categorical(label_nums, np.max(label_nums) + 1)
print(one_hot_labels)

# Use tf.keras.preprocessing.text to convert label strings to numbered index.
print("👉 Use tf.keras.preprocessing.text to convert label strings to numbered index.")
print("{} -> {}".format(" ".join(labels), keras.preprocessing.text.one_hot(" ".join(labels), 26)))

# Word processing
print("👉 Word processing")
tokenize = keras.preprocessing.text.Tokenizer(num_words=100, char_level=False)
tokenize.fit_on_texts(labels)
print("- indexing...")
print(tokenize.texts_to_sequences(labels))
print(keras.preprocessing.sequence.pad_sequences(tokenize.texts_to_sequences(labels)))
print("- matrix and one-hot...")
print(tokenize.texts_to_matrix(labels))

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

# Iterator
print(SEP)
print("👉 Iterator")
train_features, train_labels = dataset.make_one_shot_iterator().get_next()
print((train_features, train_labels))


# Train example and predict

def _input_data_(dataframe, labels):
    def __input_fn__():
        dataset = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
        sample_count = dataframe.shape[0]
        dataset = dataset.shuffle(sample_count + 1).repeat().batch(2)
        return dataset.make_one_shot_iterator().get_next()

    return __input_fn__


def train_and_predict():
    feature_cols = [
        tf.feature_column.numeric_column("Width"),
        tf.feature_column.numeric_column("Height")
    ]
    model = tf.estimator.LinearClassifier(
        feature_cols, n_classes=np.max(label_nums) + 1,
        label_vocabulary=_labels_)
    model.train(steps=2000, input_fn=_input_data_(dataframe, labels))
    print(SEP)
    print("👉 evaluate")
    evaluate = model.evaluate(
        steps=STEPS, input_fn=_input_data_(dataframe, labels))
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


if __name__ == '__main__':
    train_and_predict()
