import tensorflow as tf
import pandas as pd
import numpy as np
import nltk
import sys

CUT_DATA = .5  # Cut part of data from original.
BATCH_SZ = 128  # Train batch size.
EPOCHS = 30  # Epochs of train looπp.
STEPS = 100  # Steps of train loop.
MODE_DIR = "models/wine_price"
HIDDEN_UNITS = [1000, 1000, 1000]  # Units of each nn layers.
RUN_CONFIG = tf.estimator.RunConfig(save_checkpoints_secs=1)
LINEAR_OPTIMIZER = "Adam"
DNN_OPTIMIZER = "Adam"


def __print__(s):
    print(s, sep=' ', end='\r', flush=True)


# Replacement of Tokenizer and fit_on_texts of keras.
def fit_on_texts(dataset):
    """
    :param dataset: Series which provides lexicon source.
    :return: A list of all lexicons which already have avoided duplicated words.
    """
    lines = dataset.tolist()
    lex = []
    prec = 1
    total = len(lines)
    for line in lines:
        words = nltk.word_tokenize(line.lower())
        lex += words
        __print__("👉 {}: {:.2f}%".format("Make lexicon", (prec / total) * 100))
        prec += 1

    lemmatizer = nltk.WordNetLemmatizer()
    lex = [lemmatizer.lemmatize(word) for word in lex]

    print("Count words...")
    word_count = nltk.Counter(lex)
    lex = []
    for word in word_count:
        if 2000 > word_count[word] > 20:
            lex.append(word)

    print("Move duplicated words...")
    _lex_ = list()
    fn = lambda: [x for x in lex if not (x in _lex_ or _lex_.append(x))]
    fn()
    print("lexicon count: {}".format(len(_lex_)))
    return _lex_


# Replacement of texts_to_matrix of keras.
def texts_to_matrix(lex, dataset):
    """
    :param lex: A list of all lexicons which already have avoided duplicated words.
    :param dataset: Series which wants to be converted into a Numpy matrix.
    :return: A Numpy matrix.
    """
    feature_list = []
    lines = dataset.tolist()
    prec = 1
    total = len(lines)
    for line in lines:
        words = nltk.word_tokenize(line.lower())
        lemmatizer = nltk.WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        features = np.zeros(len(lex) + 1)
        # (len(max(lines, key=len)))  Instead vocab_size = 12000 a hyperparameter.
        for word in words:
            if word in lex:
                features[lex.index(word) + 1] = 1

        feature_list.append(list(features))
        __print__("👉 {}: {:.2f}%".format("Make matrix", (prec / total) * 100))
        prec += 1
    return np.array(feature_list)


# Replacement of texts_to_sequences and pad_sequences of keras.
def texts_to_pad_sequences(lex, dataset, padding_right=False, padding_left=False, padding_symbol=0):
    """
    :param lex:  A list of all lexicons which already have avoided duplicated words.
    :param dataset: Series is a list of lines of text content. Each line of text will be converted
                    into a sequence of integers based on the lexicons.
    :param padding_right:  'post' pad  after each sequence.
    :param padding_left: 'pre' pad  before  sequence.
    :param padding_symbol: String, 'pre' or 'post': remove values from sequences larger than maxlen,
                either at the beginning or at the end of the sequences.
    :return: List of lines of padded "text"s in integers.
    """
    feature_list = []
    lines = dataset.tolist()
    prec = 1
    total = len(lines)
    for line in lines:
        line_index = []
        words = nltk.word_tokenize(line.lower())
        lemmatizer = nltk.WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words]

        for word in words:
            if word in lex:
                line_index.append(lex.index(word) + 1)
            else:
                line_index.append(0)

        n = len(lex) + 1  # len(max(lines, key=len)),  Instead vocab_size = 12000 a hyperparameter.
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

        __print__("👉 {}: {:.2f}%".format("Make sequences", (prec / total) * 100))
        prec += 1
    return np.array(feature_list)


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

    print("data size: {}".format(len(data)))

    # Only the variety, description and price columns would be used.
    data = data.drop(
        ["country", "designation", "points", "province", "region_1", "region_2", "winery"], axis=1)

    # Prepare train and test data
    random_seed = None
    np.random.seed(random_seed)
    x_train = data.sample(frac=0.7, random_state=random_seed)
    x_test = data.drop(x_train.index)
    print("train size: {}".format(len(x_train)))
    print("test size: {}".format(len(x_test)))

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

    # Input description
    lexicon = fit_on_texts(data["description"])
    suggest_max_line = len(lexicon)
    x_bow_desc_train = texts_to_matrix(lexicon,
                                       x_train["description"])
    x_bow_desc_test = texts_to_matrix(lexicon,
                                      x_test["description"])
    x_seq_desc_train = texts_to_pad_sequences(lexicon,
                                              x_train["description"],
                                              padding_right=True)
    x_seq_desc_test = texts_to_pad_sequences(lexicon,
                                             x_test["description"],
                                             padding_right=True)

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
            "description", suggest_max_line + 1
        ))

    embedding_description_column = tf.feature_column.embedding_column(
        categorical_column=tf.feature_column.categorical_column_with_hash_bucket(
            key="embed_description",
            hash_bucket_size=int(suggest_max_line ** 0.25),
            dtype=tf.int32
        ),
        dimension=8)

    wide_columns = [variety_column, description_column]
    deep_columns = [embedding_description_column]

    # Train model
    model = tf.estimator.DNNLinearCombinedRegressor(
        linear_optimizer=LINEAR_OPTIMIZER,
        dnn_optimizer=DNN_OPTIMIZER,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=HIDDEN_UNITS,
        config=RUN_CONFIG,
        model_dir=MODE_DIR)

    # variety_x_description = tf.feature_column.crossed_column(
    #     ["variety", "description"], suggest_max_line
    # )
    #
    # # Train model
    # model = tf.estimator.DNNRegressor(
    #     feature_columns=variety_x_description,
    #     hidden_units=HIDDEN_UNITS,
    #     model_dir=MODE_DIR,
    #     optimizer=LINEAR_OPTIMIZER,
    #     config=RUN_CONFIG
    # )

    tf.logging.set_verbosity(tf.logging.INFO)
    for _ in range(EPOCHS):
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
    print("🌞 ")


if __name__ == '__main__':
    main()
