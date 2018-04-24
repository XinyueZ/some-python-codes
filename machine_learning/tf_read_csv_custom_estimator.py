#
# Define custom estimator(model).
#
# Use a small collection of CSV data to display data from
# reading, transforming, training, evaluating training, and finally predict.
#
# Ref: https://www.tensorflow.org/get_started/datasets_quickstart
#
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer

LAYERS = [50, 50, 50, 50, 50]
STEPS = 50000


def custom_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""

    net = tf.feature_column.input_layer(features, params['feature_columns'])
    for units in params['hidden_units']:
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    predicted_classes = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'classes': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=labels, logits=logits))

    label_classes = tf.argmax(labels, 1)
    accuracy = tf.metrics.accuracy(
        labels=label_classes, predictions=predicted_classes)
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    assert mode == tf.estimator.ModeKeys.TRAIN

    train_steps = tf.train.get_global_step()
    learning_rate = tf.train.exponential_decay(
        0.0025, train_steps, 100000, 0.96, staircase=False)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss, global_step=train_steps)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


FILE = "data.csv"
SEP = "=" * 140

dataframe = pd.read_csv(FILE, dtype={
    'Name': str, 'Width': int, "Height": int})

_labels_ = ["a", "b", "c"]

# Get CSV content
print(SEP)
print(dataframe)


# Get labels
print(SEP)
print("👉 Pop labels")
labels = dataframe.pop("Name")
print(labels)
print(type(labels))

# Because label in strings might not be supported.
# Convert label in strings to numbers.
print("👉 To one-hot")
encoder = LabelBinarizer()
labels = encoder.fit_transform(labels)
print(labels)

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
def _input_data_(frame, labels):
    dataset = tf.data.Dataset.from_tensor_slices((dict(frame),  labels))
    sample_count = frame.shape[0]
    dataset = dataset.shuffle(sample_count + 1).repeat().batch(4)
    return dataset.make_one_shot_iterator().get_next()


feature_cols = [
    tf.feature_column.numeric_column("Width"),
    tf.feature_column.numeric_column("Height")
]

model = tf.estimator.Estimator(
    model_fn=custom_model,
    params={
        'feature_columns': feature_cols,
        'hidden_units': LAYERS,
        'n_classes': 3,
    })
model.train(steps=STEPS, input_fn=lambda: _input_data_(dataframe, labels))

print(SEP)
print("👉 evaluate")
evaluate = model.evaluate(
    steps=50, input_fn=lambda: _input_data_(dataframe, labels))
print(evaluate)


print(SEP)
print("👉 predict")

test_width_cols = np.array(
    [10, 4,  6, 2, 11, 1], dtype=np.int32)
test_height_cols = np.array(
    [23, 56,  66, 50, 25, 1], dtype=np.int32)
predict_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"Width": test_width_cols, "Height": test_height_cols},
    shuffle=False)

predict_res = list(
    model.predict(input_fn=predict_input_fn))


for res in predict_res:
    print(
        "🙏  Probability：{:<5.2f} -> {}".format(max(res["probabilities"]),  _labels_[res["classes"][0]]))
