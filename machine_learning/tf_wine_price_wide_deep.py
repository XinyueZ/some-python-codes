import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer


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
    # _CSV_COLUMN_DEFAULTS = [[""], [""], [""],
    #                         [0.], [0.],
    #                         [""], [""], [""], [""]]
    # data = tf.decode_csv(data, record_defaults=_CSV_COLUMN_DEFAULTS)
    data.head()
    data = data.dropna()
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
    encoder = LabelEncoder()
    variety_train = encoder.fit_transform(x_train["variety"])
    num_classes = (np.max(variety_train) + 1).item()
    print("Classes of variety: {}".format(num_classes))

    # Define feature-columns
    vocab_size = 12000
    embedding_dimension = vocab_size * 0.25
    variety_categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
        'variety', hash_bucket_size=vocab_size)
    description_categorical_column = tf.feature_column.categorical_column_with_hash_bucket(
        'description', hash_bucket_size=vocab_size)
    embedding_description_column = tf.feature_column.embedding_column(
        categorical_column=description_categorical_column,
        dimension=embedding_dimension)

    wide_columns = tf.feature_column.indicator_column(
        variety_categorical_column) + tf.feature_column.indicator_column(
        description_categorical_column)
    deep_columns = [embedding_description_column]

    # Train model
    hidden_units = [100, 75, 50, 25]
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))
    model = tf.estimator.DNNLinearCombinedRegressor(
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        config=run_config)

    model.train(
        steps=10,
        input_fn=make_dataset(128, x_train, y_train,
                              shuffle=True))


if __name__ == '__main__':
    main()
