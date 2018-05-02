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
import nltk

STEPS = 50000

FILE = "data.csv"
SEP = "=" * 140


def fit_on_texts(dataset):
    lines = dataset.tolist()
    lex = []
    for line in lines:
        words = nltk.word_tokenize(line.lower())
        lex += words

    lemmatizer = nltk.WordNetLemmatizer()
    lex = [lemmatizer.lemmatize(word) for word in lex]
    _lex_ = list()
    fn = lambda: [x for x in lex if not (x in _lex_ or _lex_.append(x))]
    fn()
    return _lex_


def texts_to_matrix(lex, dataset):
    feature_list = []
    lines = dataset.tolist()
    for line in lines:
        words = nltk.word_tokenize(line.lower())
        lemmatizer = nltk.WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        features = np.zeros(len(lex) + 1)  # (len(max(lines, key=len)))
        for word in words:
            if word in lex:
                features[lex.index(word) + 1] = 1

        feature_list.append(list(features))
    return np.array(feature_list)


def texts_to_pad_sequences(lex, lines, padding_right=False, padding_left=False, padding_symbol=0):
    feature_list = []
    for line in lines:
        line_index = []
        words = nltk.word_tokenize(line.lower())
        lemmatizer = nltk.WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        for word in words:
            line_index.append(lex.index(word) + 1)

        n = len(lex) + 1  # len(max(lines, key=len)),
        if padding_right and not padding_left:
            pad_array = nltk.ngrams(line_index,
                                    n=n,  # len(max(lines, key=len)),
                                    pad_right=True,
                                    right_pad_symbol=padding_symbol)
            pad_array_to_list = list(pad_array)
            feature_list.append(pad_array_to_list[0])
        elif padding_left and not padding_right:
            pad_array = nltk.ngrams(line_index,
                                    n=n,
                                    pad_left=True,
                                    left_pad_symbol=padding_symbol)
            pad_array_to_list = list(pad_array)
            feature_list.append(pad_array_to_list[-1])
        elif padding_right and padding_left:
            pad_array = nltk.ngrams(line_index,
                                    n=n,
                                    pad_right=True,
                                    pad_left=True,
                                    right_pad_symbol=padding_symbol,
                                    left_pad_symbol=padding_symbol)
            pad_array_to_list = list(pad_array)
            feature_list.append(pad_array_to_list[-n // 2])
        else:
            feature_list.append(line_index)
    return np.array(feature_list)


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
print("- matrix...")
print(tokenize.texts_to_matrix(labels))

# Word processing other ways
print("👉 Word processing with nltk")
lexicon = fit_on_texts(labels)
print("- lexicon...")
print(lexicon)
print("- indexing...")
print(texts_to_pad_sequences(lexicon, labels, padding_right=True))
print(texts_to_pad_sequences(lexicon, labels, padding_left=True))
print(texts_to_pad_sequences(lexicon, labels, padding_right=True, padding_left=True))
print(texts_to_pad_sequences(lexicon, labels, padding_right=False, padding_left=False))
print("- matrix...")
print(texts_to_matrix(lexicon, labels))

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
