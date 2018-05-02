import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np

BATCH_SZ = 128
STEPS = 10
HIDDEN_UNITS = [500, 500, 500]
CUT_DATA = .5


def make_dataset(batch_sz, x, y=None, shuffle=False, shuffle_buffer_size=1000):
    """Create a slice Dataset from a pandas DataFrame and labels"""

    def input_fn():
        if y is not None:
            dataset = tf.data.Dataset.from_tensor_slices((dict(x), y))
        else:
            dataset = tf.data.Dataset.from_tensor_slices(dict(x))
        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size).batch(batch_sz).repeat()
        else:
            dataset = dataset.batch(batch_sz)
        return dataset.make_one_shot_iterator().get_next()

    return input_fn


def main():
    # Download data
    URL = "https://storage.googleapis.com/sara-cloud-ml/wine_data.csv"
    path = tf.keras.utils.get_file(URL.split('/')[-1], URL)

    # Clean data
    data = pd.read_csv(path)
    data = data.dropna()
    train_size = int(len(data) * CUT_DATA)
    data = data[:train_size]
    data = data[pd.notnull(data["price"])]

    variety_threshold = 500  # Anything that occurs less than this will be removed.
    value_counts = data["variety"].value_counts()
    to_remove = value_counts[value_counts <= variety_threshold].index
    data.replace(to_remove, np.nan, inplace=True)
    data = data[pd.notnull(data["variety"])]

    # Only the variety, description and price columns would be used.
    data = data.drop(
        ["country", "designation", "points", "province", "region_1", "region_2", "winery"], axis=1)

    # Prepare train and test data
    random_seed = None
    np.random.seed(random_seed)
    x_train = data.sample(frac=0.7, random_state=random_seed)
    x_test = data.drop(x_train.index)

    # Extract the labels from dataset
    y_train = x_train.pop("price")
    y_test = x_test.pop("price")

    # To get total classes of wine-varieties
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    x_variety_train = encoder.fit_transform(x_train["variety"])
    num_classes = (np.max(x_variety_train) + 1).item()
    print("Classes of variety for train: {}".format(num_classes))
    x_variety_test = encoder.fit_transform(x_test["variety"])
    print("Classes of variety for test: {}".format((np.max(x_variety_test) + 1).item()))

    # Input wine-varieties
    from sklearn.preprocessing import LabelBinarizer
    encoder = LabelBinarizer()
    x_variety_train = encoder.fit_transform(x_train["variety"])
    x_variety_test = encoder.fit_transform(x_test["variety"])
    # x_variety_train = keras.utils.to_categorical(x_variety_train, num_classes)
    # x_variety_test = keras.utils.to_categorical(x_variety_test, num_classes)

    # Input description
    vocab_size = 12000
    max_seq_length = 170
    tokenize = keras.preprocessing.text.Tokenizer(num_words=vocab_size, char_level=False)
    tokenize.fit_on_texts(x_train["description"])
    x_bow_desc_train = tokenize.texts_to_matrix(x_train["description"])
    x_bow_desc_test = tokenize.texts_to_matrix(x_test["description"])

    x_seq_desc_train = tokenize.texts_to_sequences(x_train["description"])
    x_seq_desc_test = tokenize.texts_to_sequences(x_test["description"])
    x_seq_desc_train = keras.preprocessing.sequence.pad_sequences(
        x_seq_desc_train, maxlen=max_seq_length, padding="post", dtype=np.int32)
    x_seq_desc_test = keras.preprocessing.sequence.pad_sequences(
        x_seq_desc_test, maxlen=max_seq_length, padding="post", dtype=np.int32)

    x_train = pd.Series(
        {
            "variety": x_variety_train,
            "description": x_bow_desc_train.astype(int),
            "embed_description": x_seq_desc_train
        })
    print(x_train)
    x_test = pd.Series(
        {
            "variety": x_variety_test,
            "description": x_bow_desc_test.astype(int),
            "embed_description": x_seq_desc_test
        })
    print(x_test)

    # Define feature-columns
    variety_column = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_identity(
            "variety", num_classes
        ))

    description_column = tf.feature_column.indicator_column(
        tf.feature_column.categorical_column_with_identity(
            "description", max_seq_length
        ))

    embedding_description_column = tf.feature_column.embedding_column(
        categorical_column=tf.feature_column.categorical_column_with_hash_bucket(
            key="embed_description",
            hash_bucket_size=int((max_seq_length * vocab_size) ** 0.25),
            dtype=tf.int32
        ),
        dimension=8)

    wide_columns = [variety_column, description_column]
    deep_columns = [embedding_description_column]

    # Train model
    run_config = tf.estimator.RunConfig(save_checkpoints_secs=1)
    model = tf.estimator.DNNLinearCombinedRegressor(
        linear_optimizer="Adam",
        dnn_optimizer="Adam",
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=HIDDEN_UNITS,
        config=run_config,
        model_dir='models/wine_price')

    tf.logging.set_verbosity(tf.logging.INFO)
    print("🏃....")
    model.train(
        steps=STEPS,
        input_fn=make_dataset(BATCH_SZ, x_train, y_train,
                              shuffle=True))
    print("🙏 ")
    evaluate = model.evaluate(
        steps=STEPS,
        input_fn=make_dataset(BATCH_SZ, x_test, y_test,
                              shuffle=False))
    print("💪 ")
    print(evaluate)


if __name__ == '__main__':
    main()
